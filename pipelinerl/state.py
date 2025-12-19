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
    TrainingDone,
)
from pipelinerl.streams import SingleStreamSpec, read_stream

logger = logging.getLogger(__name__)

# Fast-LLM event stream name (must match fast-llm config events.redis.stream_key)
FAST_LLM_EVENTS_STREAM = "fast_llm_events"


class TrainerState:
    def __init__(self, exp_path: Path, use_fast_llm: bool = False):
        self.exp_path = exp_path
        self.use_fast_llm = use_fast_llm
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
        if self.use_fast_llm:
            self._start_listening_fast_llm()
        else:
            self._start_listening_legacy()

    def _start_listening_legacy(self):
        """Listen to legacy PipelineRL trainer messages."""
        stream = SingleStreamSpec(exp_path=self.exp_path, topic=TRAINER_TOPIC)

        def listen():
            with read_stream(stream) as reader:
                for line in reader.read():
                    message = TypeAdapter(TrainerMessage).validate_python(line)
                    if isinstance(message, WeightUpdateSuccess):
                        self.propagated_weight_version = message.version
                    if isinstance(message, SamplesProcessed):
                        self.samples_processed = message.samples_processed
                    if isinstance(message, TrainingDone):
                        self.training_done = True
                        self._training_done_event.set()

        self._thread = threading.Thread(target=listen, daemon=True)
        self._thread.start()

    def _start_listening_fast_llm(self):
        """Listen to Fast-LLM trainer events directly from Redis."""
        import orjson
        import redis
        from pipelinerl.streams import RedisConfig, _backend, connect_to_redis

        # Fast-LLM event stream config (must match fast-llm config)
        stream_key = FAST_LLM_EVENTS_STREAM  # "fast_llm_events"
        payload_key = b"event"  # Fast-LLM uses "event" as payload key

        def listen():
            assert isinstance(_backend, RedisConfig)
            r = connect_to_redis(_backend)
            last_id = "0-0"

            logger.info(f"Listening for Fast-LLM events on Redis stream '{stream_key}'")

            while True:
                # Read from stream (blocking)
                result = r.xread({stream_key: last_id}, count=1, block=1000)

                if not result:
                    continue

                for stream_name, messages in result:
                    for msg_id, msg_data in messages:
                        last_id = msg_id

                        # Fast-LLM sends: {payload_key: orjson.dumps({type: "...", step: N})}
                        if payload_key not in msg_data:
                            logger.warning(f"Fast-LLM event missing '{payload_key.decode()}' field: {msg_data}")
                            continue

                        try:
                            event = orjson.loads(msg_data[payload_key])
                        except Exception as e:
                            logger.error(f"Failed to parse Fast-LLM event: {e}")
                            continue

                        event_type = event.get("type")
                        step = event.get("step")

                        if event_type == "initial_weights_step":
                            logger.info(f"Received initial_weights_step event: step={step}")
                            self.propagated_weight_version = step
                            # Initial step also sets samples_processed to 0
                            if self.samples_processed is None:
                                self.samples_processed = 0
                        elif event_type == "weights_ready":
                            logger.info(f"Received weights_ready event: step={step}")
                            self.propagated_weight_version = step
                        elif event_type == "training_finished":
                            logger.info("Received training_finished event")
                            self.training_done = True
                            self._training_done_event.set()
                        else:
                            logger.warning(f"Unknown Fast-LLM event type: {event_type}")

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
