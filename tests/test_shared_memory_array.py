from multiprocessing.managers import SharedMemoryManager

import pytest

from pipelinerl.shared_memory_array import EntrySizeExceeded, SharedMemoryQueue


def test_put_oversized_raises_entry_size_exceeded():
    with SharedMemoryManager() as smm:
        q = SharedMemoryQueue(smm, max_size=2, max_entry_size=1024)
        with pytest.raises(EntrySizeExceeded):
            q.put(b"x" * 4096)


def test_entry_size_exceeded_carries_sizes_and_is_valueerror():
    with SharedMemoryManager() as smm:
        q = SharedMemoryQueue(smm, max_size=1, max_entry_size=1024)
        with pytest.raises(EntrySizeExceeded) as excinfo:
            q.put(b"x" * 4096)
        err = excinfo.value
        assert err.max_size == 1024
        assert err.size > 1024
        assert isinstance(err, ValueError)  # existing ValueError callers keep working


def test_put_unpicklable_propagates_and_returns_slot():
    # A non-oversize failure (unpicklable value) must still propagate (not be
    # swallowed as a drop) AND still return the reserved slot so the queue stays
    # usable. Guards against the broad-swallow failure mode.
    with SharedMemoryManager() as smm:
        q = SharedMemoryQueue(smm, max_size=1, max_entry_size=1_000_000)
        with pytest.raises(Exception) as excinfo:
            q.put(lambda x: x)  # lambdas are not picklable
        assert not isinstance(excinfo.value, EntrySizeExceeded)
        # Slot was returned: a normal put/get still works.
        q.put(b"ok")
        assert q.get() == b"ok"


def test_put_oversized_does_not_leak_slot():
    # The free slot is reserved before serialization. A leaked slot per failed put
    # would exhaust free_slots after max_size oversize puts and then block forever
    # (the deadlock-not-crash failure mode). Attempt more oversize puts than the
    # queue has slots, then confirm normal traffic still flows repeatedly.
    with SharedMemoryManager() as smm:
        q = SharedMemoryQueue(smm, max_size=2, max_entry_size=1024)
        for _ in range(5):  # more than max_size
            with pytest.raises(EntrySizeExceeded):
                q.put(b"x" * 4096)
        for _ in range(3):
            q.put(b"a")
            q.put(b"bb")
            assert q.get() == b"a"
            assert q.get() == b"bb"
