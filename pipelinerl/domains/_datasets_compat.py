import inspect

from datasets import load_dataset as _hf_load_dataset

# `trust_remote_code` was removed from datasets>=4. Wrap load_dataset so callers can keep
# passing it: forward it when the installed version accepts it, drop it otherwise.
_LOAD_DATASET_ACCEPTS_TRUST_REMOTE_CODE = "trust_remote_code" in inspect.signature(_hf_load_dataset).parameters


def load_dataset(*args, **kwargs):
    if not _LOAD_DATASET_ACCEPTS_TRUST_REMOTE_CODE:
        kwargs.pop("trust_remote_code", None)
    return _hf_load_dataset(*args, **kwargs)
