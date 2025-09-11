import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from pipelinerl.finetune.logging_ import flatten_dict_config, init_wandb
from pipelinerl.rollouts import TrainingText
from pipelinerl.streams import (
    SingleStreamSpec,
    StreamSpec,
    StreamWriter,
    set_streams_backend,
    write_to_streams,
)
import datasets
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
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        data_stream: StreamSpec,
        stats_stream: StreamSpec,
        is_training: bool = True,
    ) -> None:
        self.data_stream = data_stream
        self.stats_stream = stats_stream
        self.tokenizer = tokenizer
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

    def process_dataset_sample(self, input_text, output_text) -> TrainingText:
        """
        Process a single dataset sample into TrainingText format.
        
        Args:
            sample: A dictionary containing the dataset sample
            
        Returns:
            List of TrainingText objects ready for training
        """
        
        # Extract text content from the sample
        # This assumes the dataset has 'text' field - adjust based on your dataset format
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        text_ids = self.tokenizer.encode(input_text + output_text, add_special_tokens=False)
        training_text = TrainingText(
            text=input_text + output_text,
            n_predicted=len(output_text),
            reward=0.0,
            logprobs=[],
            ref_logprobs=[],
            input_ids=text_ids,
            labels=[MASKED_TOKEN_ID] * len(input_ids) + text_ids[len(input_ids):],
            group_id=None,
            finished=True,
            prompt_tokens=len(input_ids),
            output_tokens=len(text_ids) - len(input_ids),
            visual_features=None,
            metadata={}
        )
        
        return training_text

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
        batch_size = 1 #self.cfg.get('batch_size', 100)
        
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
                for input_text, output_text in zip(batch['inputs_pretokenized'], batch['targets_pretokenized']):
                    training_text = self.process_dataset_sample(input_text, output_text)
                    batch_training_texts.append(training_text)
                
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
    dataset_dict = datasets.load_dataset(HF_DATASET, 'short_context_replay')
    
    # Access the training split
    train_dataset = dataset_dict['train']
    
    logger.info(f"Loaded {len(train_dataset)} training samples")
    
    # Create stream specifications
    stats_stream = SingleStreamSpec(exp_path=exp_path, topic="sft_stats")
    data_stream = SingleStreamSpec(exp_path=exp_path, topic="sft_data")
    
    # Create and run training data loop
    train_loop = SFTDataLoop(
        data_stream=data_stream,
        stats_stream=stats_stream,
        tokenizer=AutoTokenizer.from_pretrained(cfg.model_path),
        cfg=cfg,
        is_training=True,
    )
    
    train_loop.run(dataset=train_dataset)
    


@hydra.main(version_base=None, config_path="../conf", config_name="sft")
def main(cfg: DictConfig):
    run_sft_data_loop(cfg)

if __name__ == "__main__":
    main()