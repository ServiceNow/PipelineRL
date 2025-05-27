import logging

from accelerate import Accelerator

logger = logging.getLogger(__name__)

# step_scheduler_with_optimizer=False prevents the scheduler
# from being stepped multiple times in the multi-gpu setting.
# (The default behavior in AcceleratedScheduler when split_batches=False is to
#   step() "num_processes" times, because they expect the lr schedule to
#   depend on processed samples/epochs, not completed_steps)

_accelerator = None


def get_accelerator():
    global _accelerator
    if _accelerator is None:
        _accelerator = Accelerator(step_scheduler_with_optimizer=False)
    return _accelerator
