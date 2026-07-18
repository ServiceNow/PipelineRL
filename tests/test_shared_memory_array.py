from multiprocessing.managers import SharedMemoryManager

import pytest

from pipelinerl.shared_memory_array import EntrySizeExceeded, SharedMemoryQueue


def test_put_oversized_raises_typed_error_with_exact_sizes():
    with SharedMemoryManager() as smm:
        queue = SharedMemoryQueue(smm, max_size=1, max_entry_size=1024)

        with pytest.raises(EntrySizeExceeded) as exc_info:
            queue.put(b"x" * 4096)

        assert exc_info.value.size > 1024
        assert exc_info.value.max_size == 1024
        assert isinstance(exc_info.value, ValueError)


def test_put_unpicklable_propagates_and_returns_reserved_slot():
    with SharedMemoryManager() as smm:
        queue = SharedMemoryQueue(smm, max_size=1, max_entry_size=1_000_000)

        with pytest.raises(Exception) as exc_info:
            queue.put(lambda value: value)

        assert not isinstance(exc_info.value, EntrySizeExceeded)
        queue.put(b"ok")
        assert queue.get() == b"ok"


def test_repeated_oversized_puts_do_not_leak_reserved_slots():
    with SharedMemoryManager() as smm:
        queue = SharedMemoryQueue(smm, max_size=2, max_entry_size=1024)

        for _ in range(5):
            with pytest.raises(EntrySizeExceeded):
                queue.put(b"x" * 4096)

        for _ in range(3):
            queue.put(b"a")
            queue.put(b"bb")
            assert queue.get() == b"a"
            assert queue.get() == b"bb"
