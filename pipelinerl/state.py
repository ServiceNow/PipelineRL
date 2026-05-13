import logging
import threading
import time
from pathlib import Path

from pipelinerl.streams import SingleStreamSpec, read_stream

logger = logging.getLogger(__name__)

# Keep this in sync with pipelinerl.finetune_loop.TRAINER_TOPIC without importing
# the full finetune stack (which imports DeepSpeed).
TRAINER_TOPIC = "weight_update_request"


class TrainerState:
    def __init__(self, exp_path: Path):
        self.exp_path = exp_path
        self.propagated_weight_version: int | None = None
        self.samples_processed: int | None = None
        self.training_done: bool = False
        self._training_done_event = threading.Event()

    def debug_mode_init(self):
        self.propagated_weight_version = 0
        self.samples_processed = 0
        self.training_done = True
        self._training_done_event.set()

    def start_listening(self):
        stream = SingleStreamSpec(exp_path=self.exp_path, topic=TRAINER_TOPIC)

        def listen():
            with read_stream(stream) as reader:
                for line in reader.read():
                    if not isinstance(line, dict):
                        continue
                    kind = line.get("kind")
                    if kind == "weight_update_success":
                        version = line.get("version")
                        if version is not None:
                            self.propagated_weight_version = int(version)
                    if kind == "samples_processed":
                        samples = line.get("samples_processed")
                        if samples is not None:
                            self.samples_processed = int(samples)
                    if kind == "training_done":
                        self.training_done = True
                        self._training_done_event.set()

        self._thread = threading.Thread(target=listen, daemon=True)
        self._thread.start()

    def wait_for_training_done(self, timeout: float | None = None) -> bool:
        return self._training_done_event.wait(timeout=timeout)

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
