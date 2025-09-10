import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
from tapeagents.llms import TrainableLLM

from pipelinerl.finetune.logging_ import flatten_dict_config, init_wandb
from pipelinerl.rollouts import TrainingText
from pipelinerl.streams import (
    SingleStreamSpec,
    StreamSpec,
    StreamWriter,
    set_streams_backend,
    write_to_streams,
)
from pipelinerl.utils import setup_logging

logger = logging.getLogger(__name__)

HF_DATASET = "ServiceNow-AI/long-context-pipeline-rl"

# Constants for tokenization
MASKED_TOKEN_ID = -100


class SFTDataLoop:
    """
    A data processing loop that converts existing datasets into TrainingText format
    and writes them to streams for supervised fine-tuning.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        llm: TrainableLLM,
        data_stream: StreamSpec,
        stats_stream: StreamSpec,
        is_training: bool = True,
    ) -> None:
        self.data_stream = data_stream
        self.stats_stream = stats_stream
        self.llm = llm
        self.cfg = cfg
        self.is_training = is_training
        self.debug_mode = bool(cfg.debug.mode)
        
        # Initialize stats tracking
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.loop_start_time = -1
        
        logger.info(f"Initialized {'train' if self.is_training else 'test'} SFT data loop")

    def init_stats(self):
        """Initialize statistics tracking."""
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.loop_start_time = time.time()

    def process_dataset_sample(self, sample: dict) -> List[TrainingText]:
        """
        Process a single dataset sample into TrainingText format.
        
        Args:
            sample: A dictionary containing the dataset sample
            
        Returns:
            List of TrainingText objects ready for training
        """
        training_texts = []
        
        # Extract text content from the sample
        # This assumes the dataset has 'text' field - adjust based on your dataset format
        if 'text' in sample:
            text = sample['text']
        elif 'prompt' in sample and 'response' in sample:
            # For datasets with separate prompt/response fields
            text = sample['prompt'] + sample['response']
        else:
            # Try to find any text field
            text_fields = [k for k, v in sample.items() if isinstance(v, str) and len(v) > 10]
            if text_fields:
                text = sample[text_fields[0]]
            else:
                logger.warning(f"Could not find text content in sample: {sample}")
                return []
        
        # Tokenize the text
        try:
            # Use the LLM's tokenizer to tokenize the text
            tokenized = self.llm.tokenizer(
                text,
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.get('max_seq_length', 2048)
            )
            
            input_ids = tokenized['input_ids'][0].tolist()
            
            # For SFT, we typically want to predict the entire sequence
            # So we set n_predicted to the length of the sequence
            n_predicted = len(input_ids)
            
            # Create labels (same as input_ids for SFT)
            labels = input_ids.copy()
            
            # Create the TrainingText object
            training_text = TrainingText(
                text=text,
                n_predicted=n_predicted,
                input_ids=input_ids,
                labels=labels,
                finished=True,  # Assume finished for SFT data
                prompt_tokens=0,  # For SFT, we typically train on the full sequence
                output_tokens=n_predicted,
                metadata={
                    'source': 'sft_data',
                    'dataset': sample.get('dataset', 'unknown'),
                    'sample_id': sample.get('id', 'unknown'),
                }
            )
            
            training_texts.append(training_text)
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            return []
        
        return training_texts

    def update_stats(self, training_texts: List[TrainingText], dataset_name: str = "sft_data"):
        """Update statistics with processed training texts."""
        for training_text in training_texts:
            # Track basic metrics
            self.stats['text_length'][dataset_name]['all'].append(len(training_text.text))
            self.stats['input_tokens'][dataset_name]['all'].append(len(training_text.input_ids))
            self.stats['output_tokens'][dataset_name]['all'].append(training_text.output_tokens)
            self.stats['finished'][dataset_name]['all'].append(training_text.finished)

    def run(self, dataset: List[dict]):
        """
        Main processing loop that processes the dataset and writes to streams.
        
        Args:
            dataset: List of dataset samples to process
        """
        self.init_stats()
        
        published_samples = 0
        batch_size = self.cfg.get('batch_size', 100)
        
        logger.info(f"Starting SFT data processing with {len(dataset)} samples")
        
        with (
            write_to_streams(self.data_stream, "a") as data_stream_writer,
            write_to_streams(self.stats_stream, "a") as stats_writer,
        ):
            # Process dataset in batches
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                batch_training_texts = []
                
                # Process each sample in the batch
                for sample in batch:
                    training_texts = self.process_dataset_sample(sample)
                    batch_training_texts.extend(training_texts)
                
                if batch_training_texts:
                    # Convert to dict format for streaming
                    text_dumps = [text.model_dump() for text in batch_training_texts]
                    
                    # Write to data stream
                    data_stream_writer.write(text_dumps)
                    
                    # Update stats
                    self.update_stats(batch_training_texts)
                    
                    published_samples += len(batch_training_texts)
                    
                    logger.info(
                        f"Published {len(batch_training_texts)} samples to {self.data_stream}, "
                        f"total {published_samples} samples so far"
                    )
                
                # Publish stats periodically
                if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(dataset):
                    self.publish_stats(stats_writer, published_samples)
        
        logger.info(f"Finished processing {published_samples} samples")

    def publish_stats(self, stats_writer: StreamWriter, published_samples: int):
        """Publish statistics to the stats stream."""
        split_name = "test_" if not self.is_training else ""
        
        stats = defaultdict(float)
        
        # Calculate aggregated statistics
        for metric_name, dict_of_stats_per_metric in self.stats.items():
            for dataset_name, group_stats in dict_of_stats_per_metric.items():
                if group_stats['all']:
                    values = group_stats['all']
                    stats[f"{split_name}{metric_name}_mean"] = sum(values) / len(values)
                    stats[f"{split_name}{metric_name}_min"] = min(values)
                    stats[f"{split_name}{metric_name}_max"] = max(values)
                    stats[f"{split_name}{metric_name}_count"] = len(values)
        
        # Add loop-level stats
        stats.update({
            f"{split_name}published_samples": published_samples,
            f"{split_name}time_since_start": time.time() - self.loop_start_time,
        })
        
        # Write stats to stream
        stats_writer.write(dict(stats))
        
        # Reset stats for next iteration
        self.init_stats()


def run_sft_data_loop(cfg: DictConfig):
    """
    Main entry point for running the SFT data processing loop.
    
    Args:
        cfg: Hydra configuration object
    """
    set_streams_backend(**cfg.streams)
    
    # Set seed for reproducibility
    random.seed(cfg.seed)
    
    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path / "sft_data", "sft_data")
    logger.info(f"Current dir: {os.getcwd()}, experiment root dir: {cfg.output_dir}")
    
    if cfg.wandb.use_wandb:
        run = init_wandb(cfg, exp_path / "sft_data", flatten_dict_config(cfg))
        if run is None:
            raise ValueError("Failed to initialize wandb run")
    
    # Load dataset using the same pattern as actor
    dataset_loader = hydra.utils.get_method(cfg.dataset_loader)
    dataset_loader_params = cfg.get('dataset_loader_params', {})
    
    train_dataset = dataset_loader(cfg.train_dataset_names, **dataset_loader_params)
    test_dataset = dataset_loader(cfg.test_dataset_names, **dataset_loader_params)
    
    if cfg.train_subset:
        train_dataset = train_dataset[cfg.train_subset.begin : cfg.train_subset.end]
    
    logger.info(f"Loaded {len(train_dataset)} training samples")
    logger.info(f"Loaded {len(test_dataset)} test samples")
    
    # Initialize LLM for tokenization
    llm_urls = str(cfg.me.llm_urls).split("+")
    llm_url = llm_urls[0]  # Use first URL for tokenization
    
    finetune_model_path = exp_path / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        model_path = finetune_model_path
    else:
        model_path = cfg.model_path
    
    llm = TrainableLLM(
        base_url=llm_url,
        model_name=str(model_path),
        tokenizer_name=str(model_path),
        parameters=cfg.llm.parameters,
        use_cache=False,
        collect_logprobs=False,  # Not needed for SFT data processing
        observe_llm_calls=False,
    )
    
    # Create stream specifications
    stats_stream = SingleStreamSpec(exp_path=exp_path, topic="sft_stats")
    test_stats_stream = SingleStreamSpec(exp_path=exp_path, topic="sft_stats_test")
    data_stream = SingleStreamSpec(exp_path=exp_path, topic="sft_data")
    test_data_stream = SingleStreamSpec(exp_path=exp_path, topic="sft_data_test")
    
    # Create and run training data loop
    train_loop = SFTDataLoop(
        data_stream=data_stream,
        stats_stream=stats_stream,
        llm=llm,
        cfg=cfg,
        is_training=True,
    )
    
    train_loop.run(dataset=train_dataset)
    
    # Create and run test data loop if test dataset exists
    if test_dataset:
        test_loop = SFTDataLoop(
            data_stream=test_data_stream,
            stats_stream=test_stats_stream,
            llm=llm,
            cfg=cfg,
            is_training=False,
        )
        
        test_loop.run(dataset=test_dataset)
    
    logger.info("SFT data processing completed")