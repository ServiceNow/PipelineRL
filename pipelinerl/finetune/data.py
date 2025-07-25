from functools import partial
from typing import Any, Callable, Iterable, Sequence

import datasets
from omegaconf import DictConfig
import torch
import transformers
from datasets.arrow_dataset import Dataset
from datasets.combine import interleave_datasets
from datasets.fingerprint import Hasher
from datasets.load import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator
import os

from pipelinerl.finetune.utils import create_sentinel_example
from pipelinerl.rollouts import TrainingText

from .context import get_accelerator, logger
from .rl import RL_DATA_COLUMNS, prepare_rl_fields
from .types import DataArgs, DataPartArgs, PipelineBatchEncoding

datasets.builder.has_sufficient_disk_space = (
    lambda needed_bytes, directory=".": True
)  # hack for NFS filesystem with 0 disk space reported

# -100 is the default "ignore_index" in nn.CrossEntropyLoss,
# see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
MASKED_TOKEN_ID = -100


def save_samples(training_samples: list[TrainingText], jsonl_filename: str):
    assert jsonl_filename.endswith(".jsonl"), f"Filename {jsonl_filename} must end with .jsonl"
    with open(jsonl_filename, "w") as f:
        for sample in training_samples:
            f.write(sample.model_dump_json() + "\n")


def load_samples(file: str) -> list[TrainingText]:
    samples = []
    with open(file) as f:
        for line in f.readlines():
            samples.append(TrainingText.model_validate_json(line))
    return samples


def mask_labels(
    input_ids: Sequence[int],
    offset_mapping: Iterable[tuple[int, int]],
    predicted_spans: Iterable[Iterable[int]],
    masked_token_id: int = MASKED_TOKEN_ID,
) -> tuple[list[int], list[int]]:
    """
    This function creates labels from a sequence of input ids by masking
    the tokens that do not have any overlap with the character spans that
    are designated for prediction. The labels can then be used to train
    a model to predict everything except the masked tokens.

    The function also returns a list of midpoints for splitting the
    labels into a source and a target. The source is the part of the
    labels that is used to predict the target. There is one midpoint
    for each span that is designated for prediction. Each midpoint is
    the index of the first token that overlaps with the corresponding
    span.

    Args:
        input_ids (Sequence[int]): A sequence of token ids.
        offset_mapping (Iterable[tuple[int, int]]): The offset mapping
            returned by the tokenizer.
        predicted_spans (Iterable[Iterable[int]]): The character spans
            that are designated for prediction. The spans are given as
            a sequence of two-element sequences, where the first element
            is the beginning of the span (inclusive) and the second
            element is the end of the span (not inclusive).

    Returns:
        tuple[list[int], list[int]]: A tuple of masked labels and
            corresponding midpoints for splitting the labels into
            a source and a target.
    """
    labels = [masked_token_id] * len(input_ids)
    midpoints = []
    # TODO: Make this O(n_tokens) instead of O(n_tokens * n_spans)
    for span_begin, span_end in predicted_spans:
        midpoint_found = False
        for i, (offset_begin, offset_end) in enumerate(offset_mapping):
            # visual inspection of the results shows that this is the correct way to check
            if offset_begin < span_end and span_begin < offset_end:
                if not midpoint_found:
                    midpoints.append(i)
                    midpoint_found = True
                labels[i] = input_ids[i]
    return labels, midpoints


def validate_spans(text: str, predicted_spans: list[tuple[int, int]]) -> None:
    """Make sure the spans are valid, don't overlap, and are in order."""
    for start, end in predicted_spans:
        if start < 0 or end > len(text):
            raise ValueError(f"Span {start}:{end} is out of bounds for text {text!r}")
        if start > end:
            raise ValueError(f"Span {start}:{end} is invalid")
    for (start1, end1), (start2, end2) in zip(predicted_spans, predicted_spans[1:]):
        # Make sure the second span starts after the first one ends.
        if start2 < end1:
            raise ValueError(
                f"Spans {start1}:{end1} ({text[start1:end1]!r}) and {start2}:{end2} ({text[start2:end2]!r}) overlap"
            )


def preprocess_fn(
    entry: dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    is_rl: bool = False,
) -> dict[str, Any]:
    if "input_ids" in entry and entry["input_ids"]:
        # build the encoding dict from the given tokenization
        encoding = {
            "input_ids": entry["input_ids"],
            "labels": entry["labels"],
            "attention_mask": [1] * len(entry["input_ids"])
        }
    else:
        # tokenize text to build the encoding dict
        tokenizer_output = tokenizer(
            entry["text"],
            return_offsets_mapping=True,
            max_length=seq_length,
            truncation=True,
        )
        # Convert BatchEncoding to dict
        encoding = dict(tokenizer_output)
        if "predicted_spans" in entry:
            predicted_spans = entry["predicted_spans"]
        else:
            text_length = len(entry["text"])
            predicted_chars = entry.get("n_predicted", text_length)
            predicted_spans = [(text_length - predicted_chars, text_length)]
        validate_spans(entry["text"], predicted_spans)
        encoding["labels"], _ = mask_labels(
            encoding["input_ids"],  # type: ignore
            encoding["offset_mapping"],  # type: ignore
            predicted_spans,
        )
    if is_rl:
        encoding = prepare_rl_fields(
            encoding,
            entry["reward"],
            entry["logprobs"],
            entry["ref_logprobs"],
        )
    
    # Preserve visual fields if they exist
    if "pixel_values" in entry:
        encoding["pixel_values"] = entry["pixel_values"]
    if "image_thw" in entry:
        encoding["image_thw"] = entry["image_thw"]
    
    return encoding


def collate(
    examples: list[dict[str, list[int]]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    label_mask_value: int = MASKED_TOKEN_ID,
    pad_to_multiple_of: int = 16,
) -> PipelineBatchEncoding:
    # turn list of dicts with the same keys into a dict of lists
    example_dict = {key: [example[key] for example in examples] for key in examples[0].keys()}
    seq_length = max(len(i) for i in example_dict["input_ids"])
    if seq_length % pad_to_multiple_of:
        seq_length += pad_to_multiple_of - (seq_length % pad_to_multiple_of)
    result = {}
    
    # Visual feature fields that should be stacked, not padded
    if "visual_features" in example_dict and isinstance(example_dict["visual_features"][0], dict):
        for k, seq_list in example_dict["visual_features"][0].items():
            if k == "image_grid_thw":
                # image_grid_thw should remain as a list
                result[k] = seq_list
            else:
                # Other visual fields like pixel_values can be stacked as tensors
                valid_tensors = [torch.tensor(seq) for seq in seq_list]
                result[k] = torch.stack(valid_tensors)
    
    for k, seq_list in example_dict.items():
        if k == "model_version":
            continue
        if any(isinstance(seq, (str, dict)) for seq in seq_list):
            logger.debug(f"Skipping key '{k}' - contains str/dict sequences")
            continue
        # Check if any sequence contains strings or dicts
        if any(isinstance(item, (str, dict)) for seq in seq_list if isinstance(seq, list) for item in seq):
            logger.debug(f"Skipping key '{k}' - sequences contain str/dict items")
            continue
        else:
            # Handle sequence data: pad as usual
            padded_sequences = []
            pad_value = label_mask_value if k == "labels" else (0.0 if k in RL_DATA_COLUMNS else 0)
            for seq in seq_list:
                if seq is None:
                    continue  # Skip None sequences, e.g. visual features when absent
                if not isinstance(seq, list):
                    seq = [seq]
                padding = [pad_value] * (seq_length - len(seq))
                padded = (seq + padding) if tokenizer.padding_side == "right" else (padding + seq)
                padded_sequences.append(padded)
            result[k] = torch.tensor(padded_sequences)
    result["model_version"] = min([example.get("model_version", 0) for example in examples])
    result["is_packed"] = False 
    return PipelineBatchEncoding(**result)


def collate_packed(
    examples: list[dict[str, list[int]]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_parallel: int,
    label_pad_value: int = MASKED_TOKEN_ID,
) -> PipelineBatchEncoding:
    # pre-compute total length and create tensors in one go
    total_length = sum(len(example["input_ids"]) for example in examples)
    if total_length % seq_parallel != 0:
        padding = seq_parallel - (total_length % seq_parallel)
        sentinel_model_version = max(example["model_version"] for example in examples)
        sentinel_example = create_sentinel_example(padding, tokenizer=tokenizer, model_version=sentinel_model_version)
        examples = examples + [sentinel_example]
        total_length = sum(len(example["input_ids"]) for example in examples)
    else: 
        padding = 0

    # create a single tensor for sequence boundaries
    seq_boundaries = torch.zeros(len(examples) + 1, dtype=torch.int)
    seq_boundaries[1:] = torch.tensor([len(example["input_ids"]) for example in examples]).cumsum(0)

    # preallocate all tensors at once
    base_tensors = {
        "input_ids": torch.empty(1, total_length, dtype=torch.long),
        "labels": torch.empty(1, total_length, dtype=torch.long),
        "attention_mask": torch.ones(1, total_length, dtype=torch.long),  # initialize to 1s
        "position_ids": torch.empty(1, total_length, dtype=torch.long),
    }

    # initialize lists for extra keys
    extra_keys = [col for col in RL_DATA_COLUMNS if col in examples[0]]
    extra_lists = {key: [] for key in extra_keys}

    for i, example in enumerate(examples):
        start_idx = seq_boundaries[i].item()
        end_idx = seq_boundaries[i + 1].item()
        seq_len = end_idx - start_idx

        base_tensors["input_ids"][0, start_idx:end_idx] = torch.tensor(example["input_ids"], dtype=torch.long)

        # use arange to fill position_ids
        base_tensors["position_ids"][0, start_idx:end_idx] = torch.arange(seq_len)

        # process labels
        example_labels = torch.tensor(example["labels"], dtype=torch.long)
        if i > 0 and seq_len > 0:
            example_labels[0] = label_pad_value
        base_tensors["labels"][0, start_idx:end_idx] = example_labels

        # handle extra keys
        for key in extra_keys:
            value = example[key]
            if isinstance(value, (list, tuple)):
                extra_lists[key].extend(value)
            else:
                extra_lists[key].append(value)

    extra_tensors = default_data_collator([{k: extra_lists[k] for k in extra_keys}], return_tensors="pt")

    result = {**base_tensors, **extra_tensors}
    result["model_version"] = min([example.get("model_version", 0) for example in examples])
    result["is_packed"] = True 
    result["seq_boundaries"] = seq_boundaries
    result["padding"] = padding
    return PipelineBatchEncoding(**result)


def create_dataloader(
    data_parts: list[DataPartArgs] | list[TrainingText],
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch_size: int,
    seq_length: int,
    is_rl: bool = False,
    rng: torch.Generator | None = None,
    shuffle: bool = False,
    rl_data_callback: Callable | None = None,
    n_examples: int | None = None,
) -> DataLoader:
    preprocess = partial(preprocess_fn, seq_length=seq_length, tokenizer=tokenizer, is_rl=is_rl)
    columns = ["input_ids", "labels", "attention_mask"]
    if is_rl:
        columns += RL_DATA_COLUMNS

    logger.info(f"Instantiated preprocess function hash {Hasher.hash(preprocess)}")
    collate_fn = partial(
        collate,
        tokenizer=tokenizer,
    )
    logger.info(f"Instantiated collate_fn hash {Hasher.hash(collate_fn)}")

    datasets_list = []
    weights = []
    stop = False
    for part in data_parts:
        if isinstance(part, TrainingText):
            dataset_part = Dataset.from_list([s.model_dump() for s in data_parts])
            weights.append(1.0)
            stop = True
        else:
            # The path must point to the directory containing the data files
            # for one split of interest. `load_dataset` will automatically call
            # this split "train".
            dataset_part = load_dataset(part.path, split="train", data_files=part.files)
            assert isinstance(dataset_part, Dataset)
            weights.append(part.weight)

        logger.info(f"Raw data part size: {dataset_part.num_rows}")
        logger.info(f"Raw data part fingerprint: {dataset_part._fingerprint}")

        num_proc = (os.cpu_count() // get_accelerator().num_processes) or 1
        dataset_part = dataset_part.map(
            preprocess,
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=num_proc,
        )
        dataset_part = dataset_part.with_format(columns=columns)

        logger.info(f"Preprocessed data part fingerprint: {dataset_part._fingerprint}")
        datasets_list.append(dataset_part)
        if stop:
            break
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    data = interleave_datasets(
        datasets_list,
        probabilities=probs,
        stopping_strategy="all_exhausted",
        seed=rng.initial_seed() if rng is not None else None,
    )
    logger.info(f"Merged data size: {data.num_rows}")
    logger.info(f"Merged data fingerprint: {data._fingerprint}")

    if rl_data_callback is not None:
        get_accelerator().wait_for_everyone()
        data = rl_data_callback(dataset=data, columns=columns, collate_fn=collate_fn)

    if n_examples:
        data = data.select(range(n_examples))

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=rng,
    )


def prepare_dataloaders(
    args: DictConfig,
    data_args: DataArgs,
    tokenizer: transformers.PreTrainedTokenizerBase,
    rl_data_callback: Callable | None,
    dataloader_rng: torch.Generator | None,
    is_rl: bool = False,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    _create_dataloader = partial(
        create_dataloader,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        rng=dataloader_rng,
        rl_data_callback=rl_data_callback,
        is_rl=is_rl,
    )

    # Load dataset and dataloader
    train_dataloader = _create_dataloader(
        data_parts=data_args.data_parts_train,
        batch_size=args.train_batch_size,
        n_examples=args.n_examples,
        shuffle=True,
    )

    eval_dataloader = (
        _create_dataloader(
            data_parts=data_args.data_parts_valid,
            batch_size=args.valid_batch_size,
            shuffle=False,
        )
        if data_args.data_parts_valid
        else None
    )

    dev_dataloader = (
        _create_dataloader(
            data_parts=data_args.data_parts_dev,
            batch_size=args.valid_batch_size,
            shuffle=False,
        )
        if data_args.data_parts_dev
        else None
    )

    return train_dataloader, eval_dataloader, dev_dataloader
