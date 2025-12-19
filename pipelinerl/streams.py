from abc import ABC, abstractmethod
import pickle
import numpy
import orjson
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Iterator, Literal, Self, TextIO
import redis
import torch
from pydantic import BaseModel
import redis.exceptions


logger = logging.getLogger(__name__)

# If we try to read too often, the code will be to slow. Too rarely, and delays will be too big.
_REREAD_DELAY = 0.1
# Every time we recheck if a stream is createed we print a warning, best not to do it too often.
_RECHECK_DELAY = 3.0


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379


_backend: RedisConfig | Literal["files"] | None = None


def set_streams_backend(backend: Literal["redis"] | Literal["files"], **kwargs):
    """Set the backend for the streams. Currently only redis is supported."""
    global _backend
    if _backend is not None:
        raise ValueError("Backend already set. Cannot change it.")
    if backend == "redis":
        _backend = RedisConfig(**kwargs)
    elif backend == "files":
        _backend = "files"
    else:
        raise ValueError(f"Invalid backend: {backend}. Only 'redis' and 'files' are supported.")


def raise_if_backend_not_set():
    """Raise an error if the backend is not set. This is used to check if the backend is set before using it."""
    if _backend is None:
        raise ValueError("Backend not set. Please call set_streams_backend() first.")


class SingleStreamSpec(BaseModel):
    exp_path: Path
    topic: str
    instance: int = 0
    partition: int = 0

    def __str__(self):
        return f"{self.topic}/{self.instance}/{self.partition}"


class StreamRangeSpec(BaseModel):
    exp_path: Path
    topic: str
    instance: int = 0
    partition_range: tuple[int, int]

    def __str__(self):
        return f"{self.topic}/{self.instance}/{self.partition_range[0]}-{self.partition_range[1]}"


# Inferfaces


class StreamWriter(ABC):
    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def write(self, data: Any, partition: int | None = None):
        pass


class StreamReader(ABC):
    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def read(self) -> Iterator[Any]:
        pass


# Redis-based streaming


def connect_to_redis(config: RedisConfig):
    """Connect to the Redis server. Unlimited retries."""
    while True:
        try:
            logger.debug(f"Trying to connect to Redis server at {config.host}:{config.port}")
            client = redis.Redis(host=config.host, port=config.port)
            client.ping()
            logger.info(f"Connected to Redis server")
            return client
        except (redis.exceptions.TimeoutError, redis.ConnectionError) as e:
            logger.warning(f"Waiting for Redis server ({type(e)}). Retrying in 5 seconds.")
            time.sleep(5)


class RedisStreamWriter(StreamWriter):
    def __init__(self, stream: SingleStreamSpec, mode: Literal["w", "a"] = "a"):
        self.stream = stream
        assert isinstance(_backend, RedisConfig)
        self._stream_name = str(self.stream)
        self._redis = connect_to_redis(_backend)
        if mode == "a":
            # If we are appending, we need to get the last index from the stream
            # and start from there.
            last_entry = self._redis.xrevrange(self._stream_name, count=1)
            if last_entry:
                assert isinstance(last_entry, list) and len(last_entry) == 1
                entry_id, entry = last_entry[0]
                self._index = int(entry["index".encode()].decode()) + 1
            else:
                self._index = 0
        elif mode == "w":
            # If we are writing, we need to start from 0. If there's any data for this stream,
            # we should crash, cause overwriting is a bad idea.
            last_entry = self._redis.xrevrange(str(self.stream), count=1)
            if last_entry:
                raise ValueError(f"Stream {self.stream} already exists. Cannot overwrite it.")
            self._index = 0
        else:
            raise ValueError(f"Invalid mode: {mode}. Only 'w' and 'a' are supported.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._redis.close()

    def write(self, data, partition: int | None = None):
        if partition is not None:
            raise ValueError()
        if isinstance(data, BaseModel):
            data = data.model_dump()
        data = pickle.dumps(data)
        self._redis.xadd(self._stream_name, {"index": self._index, "data": data}, maxlen=1000000, approximate=True)
        self._index += 1


class RedisStreamReader(StreamReader):
    def __init__(self, stream: SingleStreamSpec):
        self.stream = stream
        assert isinstance(_backend, RedisConfig)
        self._redis = connect_to_redis(_backend)
        self._stream_name = str(self.stream)
        self._last_id = 0
        self._index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._redis.close()

    def read(self):
        block = int(_REREAD_DELAY * 1000)
        while True:
            # Read from the stream
            response = self._redis.xread({self._stream_name: self._last_id}, count=1, block=block)
            if response:
                assert isinstance(response, list) and len(response) == 1
                stream_name, result = response[0]
                assert stream_name.decode("utf-8") == self._stream_name
                assert isinstance(result, list) and len(result) == 1
                entry_id, entry = result[0]
                if int(entry[b"index"].decode("utf-8")) != self._index:
                    raise ValueError(f"Index mismatch: expected {self._index}, got {entry['index']}")
                self._last_id = entry_id
                self._index += 1
                yield pickle.loads(entry[b"data"])


class RedisSharedStreamWriter(StreamWriter):
    """Redis writer that supports multiple producers appending to a single stream."""

    def __init__(
        self,
        stream: SingleStreamSpec,
        mode: Literal["w", "a"] = "a",
        *,
        writer_id: str | None = None,
        maxlen: int = 1_000_000,
    ):
        self.stream = stream
        assert isinstance(_backend, RedisConfig)
        self._redis = connect_to_redis(_backend)
        self._stream_name = str(self.stream)
        self._counter_key = f"stream:{self._stream_name}:next_index"
        self._writer_id = str(writer_id) if writer_id is not None else None
        self._maxlen = maxlen

        if mode not in {"w", "a"}:
            raise ValueError(f"Invalid mode: {mode}. Only 'w' and 'a' are supported.")

        if mode == "w":
            last_entry = self._redis.xrevrange(self._stream_name, count=1)
            if last_entry:
                raise ValueError(f"Stream {self.stream} already exists. Cannot overwrite it.")
            self._redis.delete(self._counter_key)
            self._redis.set(self._counter_key, -1)
        else:
            if not self._redis.exists(self._counter_key):
                last_entry = self._redis.xrevrange(self._stream_name, count=1)
                if last_entry:
                    _, entry = last_entry[0]
                    raw_index = entry.get(b"index")
                    next_index = int(raw_index.decode("utf-8")) + 1 if raw_index else 0
                else:
                    next_index = 0
                self._redis.set(self._counter_key, next_index - 1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._redis.close()

    def write(self, data, partition: int | None = None):
        # Note: partition is ignored for shared streams - all data goes to a single stream
        # This is intentional for Fast-LLM integration where Fast-LLM handles its own sharding
        serialized = _serialize_with_orjson(data)
        entry_index = self._redis.incr(self._counter_key)
        record: dict[str, Any] = {
            "index": str(entry_index),
            "data": serialized,
            "ts": f"{time.time():.6f}",
        }
        if self._writer_id is not None:
            record["writer"] = self._writer_id
        self._redis.xadd(self._stream_name, record, maxlen=self._maxlen, approximate=True)


class RedisSharedStreamReader(StreamReader):
    """Redis reader that validates fan-in ordering for a shared stream."""

    def __init__(self, stream: SingleStreamSpec, *, fail_on_gap: bool = True):
        self.stream = stream
        assert isinstance(_backend, RedisConfig)
        self._redis = connect_to_redis(_backend)
        self._stream_name = str(self.stream)
        self._last_id = 0
        self._expected_index: int | None = None
        self._fail_on_gap = fail_on_gap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._redis.close()

    def _update_expected_index(self, entry: dict[bytes, bytes]):
        raw_index = entry.get(b"index")
        if raw_index is None:
            return

        index_value = int(raw_index.decode("utf-8"))
        if self._expected_index is None:
            self._expected_index = index_value
        elif index_value != self._expected_index:
            message = (
                f"Index mismatch for shared stream {self.stream}: expected {self._expected_index}, got {index_value}"
            )
            if self._fail_on_gap:
                raise ValueError(message)
            logger.warning(message)
            self._expected_index = index_value

        self._expected_index += 1

    def read(self):
        block = int(_REREAD_DELAY * 1000)
        while True:
            response = self._redis.xread({self._stream_name: self._last_id}, count=1, block=block)
            if not response:
                continue

            stream_name, result = response[0]
            assert stream_name.decode("utf-8") == self._stream_name
            assert isinstance(result, list) and len(result) == 1
            entry_id, entry = result[0]
            self._last_id = entry_id
            self._update_expected_index(entry)

            payload = entry.get(b"data")
            if payload is None:
                raise ValueError(f"Shared stream entry missing 'data' field: {entry}")

            yield orjson.loads(payload)


class RoundRobinRedisStreamWriter(StreamWriter):
    # TODO: share the connection across writers

    def __init__(self, streams: StreamRangeSpec, mode: Literal["w", "a"] = "a"):
        self.streams = streams
        self._next_stream = 0
        self._writers = [
            RedisStreamWriter(
                SingleStreamSpec(
                    exp_path=self.streams.exp_path,
                    topic=self.streams.topic,
                    instance=self.streams.instance,
                    partition=i,
                ),
                mode=mode,
            )
            for i in range(*self.streams.partition_range)
        ]

    def __enter__(self):
        for writer in self._writers:
            writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for writer in self._writers:
            writer.__exit__(exc_type, exc_value, traceback)

    def write(self, data, partition: int | None = None):
        if partition is not None:
            # Write to specific partition
            if partition < 0 or partition >= len(self._writers):
                raise ValueError(f"Invalid partition {partition}. Must be between 0 and {len(self._writers) - 1}")
            self._writers[partition].write(data)
        else:
            # Use round-robin
            self._writers[self._next_stream].write(data)
            self._next_stream = (self._next_stream + 1) % len(self._writers)


# File-based streaming


def stream_dir(exp_path: Path, topic: str, instance: int, partition: int) -> Path:
    return exp_path / "streams" / topic / str(instance) / str(partition)


def stream_file(stream_dir: Path, shard_id: int) -> Path:
    return stream_dir / f"{shard_id}.jsonl"


StreamSpec = SingleStreamSpec | StreamRangeSpec


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    if isinstance(value, numpy.ndarray):
        return value

    if isinstance(value, numpy.generic):
        return value.item()

    if isinstance(value, dict):
        return {key: _to_json_ready(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]

    return value


def _serialize_with_orjson(data: Any) -> bytes:
    return orjson.dumps(_to_json_ready(data), option=orjson.OPT_SERIALIZE_NUMPY)


class FileStreamWriter(StreamWriter):
    def __init__(self, stream: SingleStreamSpec, mode: Literal["w", "a"] = "a"):
        self.stream = stream
        self.mode = mode

    def __enter__(self):
        # TODO: sharding
        _file_dir = stream_dir(self.stream.exp_path, self.stream.topic, self.stream.instance, self.stream.partition)
        os.makedirs(_file_dir, exist_ok=True)
        self._file_path = stream_file(_file_dir, 0)
        self._file = open(self._file_path, self.mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    def write(self, data, partition: int | None = None):
        if partition is not None:
            raise ValueError()
        # Textual streams are so useful, that we try hard to jsonify the given object.
        payload = _serialize_with_orjson(data)
        self._file.write(payload.decode("utf-8"))
        self._file.write("\n")
        self._file.flush()


def read_jsonl_stream(f: TextIO, retry_delay: float = _REREAD_DELAY) -> Iterator[Any]:
    position = f.tell()

    while True:
        line = f.readline()

        # Handle line ending
        if line.endswith("\n"):
            try:
                yield json.loads(line)
                position = f.tell()
            except json.JSONDecodeError as e:
                e.msg += f" (position {position})"
                e.position = position  # type: ignore
                raise e
        else:
            f.seek(position)
            time.sleep(retry_delay)
            continue


class FileStreamReader(StreamReader):
    def __init__(self, stream: SingleStreamSpec):
        self.stream = stream

    def __enter__(self):
        _file_dir = stream_dir(self.stream.exp_path, self.stream.topic, self.stream.instance, self.stream.partition)
        # TODO: support sharding
        self._file_path = stream_file(_file_dir, 0)
        # wait until the file is created with a delay of 3.0 seconds
        # and a logger warning
        while not os.path.exists(self._file_path):
            logger.warning(f"Waiting for {self.stream} to be created")
            time.sleep(_RECHECK_DELAY)
        self._file = open(self._file_path, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    def read(self):
        retry_time = 0.01
        cur_retries = 0
        max_retries = 10
        while True:
            try:
                for line in read_jsonl_stream(self._file):
                    yield line
                    cur_retries = 0
            except json.JSONDecodeError as e:
                # Sometimes when the stream file is being written to as the as time as we reading it,
                # we get lines like \0x00\0x00\0x00\0x00\0x00\0x00\0x00\0x00 that break the JSON decoder.
                # We have to reopen the file and seek to the previous position to try again.
                if cur_retries < max_retries:
                    logger.warning(
                        f"Could not decode JSON from {self.stream}, might have run into end of the file. Will reopen the file and retry ({cur_retries}/{max_retries}), starting from position {e.position})"
                    )  # type: ignore
                    time.sleep(retry_time)
                    self._file.close()
                    self._file = open(self._file_path, "r")
                    self._file.seek(e.position)
                    retry_time *= 2
                    cur_retries += 1
                    continue
                else:
                    logger.error(f"Error reading stream {self.stream}, giving up after {max_retries} retries")
                    raise e


class RoundRobinFileStreamWriter(StreamWriter):
    def __init__(self, streams: StreamRangeSpec, mode: Literal["w", "a"] = "a"):
        self.streams = streams
        self._next_stream = 0
        self._writers = [
            FileStreamWriter(
                SingleStreamSpec(
                    exp_path=self.streams.exp_path,
                    topic=self.streams.topic,
                    instance=self.streams.instance,
                    partition=i,
                ),
                mode=mode,
            )
            for i in range(*self.streams.partition_range)
        ]

    def __enter__(self):
        for writer in self._writers:
            writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for writer in self._writers:
            writer.__exit__(exc_type, exc_value, traceback)

    def write(self, data, partition: int | None = None):
        if partition is not None:
            # Write to specific partition
            if partition < 0 or partition >= len(self._writers):
                raise ValueError(f"Invalid partition {partition}. Must be between 0 and {len(self._writers) - 1}")
            self._writers[partition].write(data)
        else:
            # Use round-robin
            self._writers[self._next_stream].write(data)
            self._next_stream = (self._next_stream + 1) % len(self._writers)


# Below are the public stream APIs. Easy to replace files with Redis or another pubsub system.


def read_stream(stream: SingleStreamSpec, *, shared: bool = False, fail_on_gap: bool = True) -> StreamReader:
    """Start reading the stream from the beginning.

    When ``shared`` is True, multiple producers are assumed to append to the same
    Redis stream and the reader will validate ordering using the stored index
    metadata.
    """
    raise_if_backend_not_set()
    if not isinstance(stream, SingleStreamSpec):
        raise ValueError(f"Invalid stream spec: {stream}")
    if isinstance(_backend, RedisConfig):
        if shared:
            return RedisSharedStreamReader(stream, fail_on_gap=fail_on_gap)
        return RedisStreamReader(stream)
    elif _backend == "files":
        if shared:
            raise ValueError("Shared stream mode is only supported with the Redis backend")
        return FileStreamReader(stream)
    else:
        assert False


def write_to_streams(
    streams: StreamSpec,
    mode: Literal["w", "a"] = "a",
    *,
    shared: bool = False,
    writer_id: str | None = None,
) -> StreamWriter:
    """Append to the end of the stream.

    Set ``shared`` to True when multiple producers must append to the same Redis
    stream and ServiceNow/Fast-LLM will perform downstream sharding.
    """
    raise_if_backend_not_set()
    if not isinstance(streams, (SingleStreamSpec, StreamRangeSpec)):
        raise ValueError(f"Invalid stream spec: {streams}")
    if isinstance(_backend, RedisConfig):
        if isinstance(streams, SingleStreamSpec):
            if shared:
                return RedisSharedStreamWriter(streams, mode, writer_id=writer_id)
            return RedisStreamWriter(streams, mode)
        elif isinstance(streams, StreamRangeSpec):
            if shared:
                raise ValueError("Shared Redis streams only support SingleStreamSpec inputs")
            return RoundRobinRedisStreamWriter(streams, mode)
        else:
            assert False
    elif _backend == "files":
        if shared:
            raise ValueError("Shared stream mode is only supported with the Redis backend")
        if isinstance(streams, SingleStreamSpec):
            return FileStreamWriter(streams, mode)
        elif isinstance(streams, StreamRangeSpec):
            return RoundRobinFileStreamWriter(streams, mode)
        else:
            assert False
    else:
        assert False
