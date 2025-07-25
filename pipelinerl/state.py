import logging
import threading
import time
from pathlib import Path

from pydantic import TypeAdapter

from pipelinerl.finetune_loop import (
    TRAINER_TOPIC,
    TrainerMessage,
    WeightUpdateSuccess,
    SamplesProcessed,
)
from pipelinerl.streams import SingleStreamSpec, read_stream

logger = logging.getLogger(__name__)


class TrainerState:
    def __init__(self, exp_path: Path):
        self.exp_path = exp_path
        self.propagated_weight_version: int | None = None
        self.samples_processed: int | None = None

    def debug_mode_init(self):
        self.propagated_weight_version = 0
        self.samples_processed = 0

    def start_listening(self):
        stream = SingleStreamSpec(exp_path=self.exp_path, topic=TRAINER_TOPIC)

        def listen():
            with read_stream(stream) as reader:
                for line in reader.read():
                    message = TypeAdapter(TrainerMessage).validate_python(line)
                    if isinstance(message, WeightUpdateSuccess):
                        self.propagated_weight_version = message.version
                    if isinstance(message, SamplesProcessed):
                        self.samples_processed = message.samples_processed

        self._thread = threading.Thread(target=listen)
        self._thread.start()
    
    def wait_for_processed_samples(self):
        while self.samples_processed is None:
            logger.info("Waiting for the trainer to declare the number of processed samples")
            time.sleep(1)
        return self.samples_processed

    def wait_for_model_version(self):
        while self.propagated_weight_version is None:
            logger.info("Waiting for the trainer to declare the initial weight version")
            time.sleep(1)
        return self.propagated_weight_version
