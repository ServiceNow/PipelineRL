import pickle
import struct
from multiprocessing import Queue
from multiprocessing.managers import SharedMemoryManager, SyncManager
from typing import Any
from queue import Empty, Full


class SharedMemoryArray:
    """
    A class that manages an array of Python objects in shared memory.

    Objects are serialized using pickle and stored in a shared memory buffer.
    The array has a fixed number of entries with a maximum size per entry.
    """

    def __init__(self, smm: SharedMemoryManager, num_entries: int, max_entry_size: int):
        """
        Initialize a shared array.

        Args:
            num_entries: Number of entries in the array
            max_entry_size: Maximum size in bytes for each entry when serialized
        """
        if num_entries <= 0:
            raise ValueError("Number of entries must be positive")
        if max_entry_size <= 0:
            raise ValueError("Maximum entry size must be positive")

        self.num_entries = num_entries
        self.max_entry_size = max_entry_size

        # Each entry has:
        # - 4 bytes for the actual data size (to know how much to unpickle)
        # - max_entry_size bytes for the serialized data
        self.entry_size = 4 + max_entry_size

        # Create shared memory buffer
        buffer_size = self.num_entries * self.entry_size
        self.shared_mem = smm.SharedMemory(size=buffer_size)

        # Initialize all entries as empty
        for i in range(num_entries):
            self._set_entry_size(i, 0)

        self._max_actual_entry_size = 0

    def get_memory_size(self) -> int:
        """Get the size of the shared memory buffer."""
        return self.shared_mem.size

    def _get_entry_offset(self, index: int) -> int:
        """Calculate the byte offset for a given index."""
        if not 0 <= index < self.num_entries:
            raise IndexError(f"Index {index} out of range (0-{self.num_entries - 1})")
        return index * self.entry_size

    def _get_entry_size(self, index: int) -> int:
        """Get the actual size of data stored at the given index."""
        offset = self._get_entry_offset(index)
        return struct.unpack("I", self.shared_mem.buf[offset : offset + 4])[0]

    def _set_entry_size(self, index: int, size: int) -> None:
        """Set the size information for an entry."""
        if size > self.max_entry_size:
            raise ValueError(f"Data size ({size} bytes) exceeds maximum entry size ({self.max_entry_size} bytes)")
        offset = self._get_entry_offset(index)
        struct.pack_into("I", self.shared_mem.buf, offset, size)

    def __getitem__(self, index: int) -> Any:
        """Get the object at the given index."""
        size = self._get_entry_size(index)
        self._max_actual_entry_size = max(self._max_actual_entry_size, size)
        if size == 0:
            return None

        offset = self._get_entry_offset(index) + 4  # Skip size field
        data = self.shared_mem.buf[offset : offset + size]
        return pickle.loads(data)

    def __setitem__(self, index: int, value: Any) -> None:
        """Set the object at the given index."""
        # Serialize the value
        data = pickle.dumps(value)
        size = len(data)

        if size > self.max_entry_size:
            raise ValueError(
                f"Serialized object size ({size} bytes) exceeds maximum entry size ({self.max_entry_size} bytes)"
            )

        # Write the data
        offset = self._get_entry_offset(index) + 4  # Skip size field
        # Use memoryview for faster copying
        self.shared_mem.buf[offset : offset + size] = data
        # Update the size
        self._set_entry_size(index, size)
        self._max_actual_entry_size = max(self._max_actual_entry_size, size)

    def __len__(self) -> int:
        """Return the number of entries in the array."""
        return self.num_entries
    
    def max_actual_entry_size(self) -> int:
        """Return the maximum size of an entry written to the array."""
        return self._max_actual_entry_size


class SharedMemoryQueue:
    """
    A fixed-size queue backed by shared memory.
    
    Uses a SharedMemoryArray for storage and multiprocessing Queues to track available and filled slots.
    Items are stored in the shared memory array and slot indices are managed via the queues.
    """

    def __init__(self, smm: SharedMemoryManager, max_size: int, max_entry_size: int):
        """
        Initialize a shared memory queue.

        Args:
            smm: SharedMemoryManager instance
            max_size: Maximum number of items the queue can hold
            max_entry_size: Maximum size in bytes for each item when serialized
        """
        self.max_size = max_size
        self.shared_array = SharedMemoryArray(smm, max_size, max_entry_size)
        
        # Queue to track available slots (indices)
        self.free_slots = Queue(maxsize=max_size)
        
        # Queue to track filled slots (indices)
        self.content_slots = Queue(maxsize=max_size)
        
        # Initialize with all slots available
        for i in range(max_size):
            self.free_slots.put(i)

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        """
        Put an item into the queue.

        Args:
            item: The item to add to the queue
            block: Whether to block if the queue is full
            timeout: Timeout for blocking operations
        """
        # Get an available slot
        try:
            slot_index = self.free_slots.get(block=block, timeout=timeout)
        except Empty:
            raise Full()
        
        # Store the item in the shared array
        self.shared_array[slot_index] = item
        
        # Add slot to filled slots queue
        self.content_slots.put(slot_index)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """
        Get an item from the queue.

        Args:
            block: Whether to block if the queue is empty
            timeout: Timeout for blocking operations

        Returns:
            The item retrieved from the queue
        """
        # Get a filled slot
        slot_index = self.content_slots.get(block=block, timeout=timeout)
        
        # Retrieve the item from the shared array
        item = self.shared_array[slot_index]
        
        # Return slot to available pool
        self.free_slots.put(slot_index)
        
        return item
    
    def get_memory_size(self) -> int:
        """Get the size of the shared memory buffer."""
        return self.shared_array.get_memory_size()
    
    def full(self) -> bool:
        """Check if the queue is full."""
        return self.free_slots.empty()
    
    def qsize(self) -> int:
        """Get the current size of the queue."""
        return self.content_slots.qsize()
    
    def max_actual_entry_size(self) -> int:
        """Get the maximum size of an entry written to the queue."""
        return self.shared_array.max_actual_entry_size()