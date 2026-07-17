"""Pytest configuration and fixtures for vllm1 tests."""

import os
import pytest
import torch
import tempfile
from pathlib import Path
import subprocess
import sys

from pipelinerl.vllm1 import EngineManager


@pytest.fixture(scope="session")
def model_name():
    """Model to use for testing."""
    return "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def sample_prompts():
    """Sample prompts for generation testing."""
    return [
        "Write a haiku about coding:",
        "The capital of France is",
        "In a galaxy far away,",
    ]


@pytest.fixture(scope="session")
def simple_prompt():
    """Single simple prompt for deterministic testing."""
    return "The capital of France is"


@pytest.fixture(scope="session")
def num_gpus():
    """Number of GPUs available."""
    return torch.cuda.device_count()


@pytest.fixture(scope="session")
def require_2_gpus(num_gpus):
    """Skip test if less than 2 GPUs available."""
    if num_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")


@pytest.fixture(scope="session")
def require_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("Test requires GPU")


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def shared_test_dir():
    """Session-scoped shared directory for test data that persists across tests.

    Use this for data that needs to be shared between tests (like perturbed weights).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def distributed_init_method(temp_dir):
    """File-based init method for distributed testing."""
    return f"file://{temp_dir}/dist_init"


@pytest.fixture(scope="session")
def shared_distributed_init_method(shared_test_dir):
    """Session-scoped file-based init method for tests that share data."""
    return f"file://{shared_test_dir}/dist_init"


@pytest.fixture(scope="session")
def cache_dir():
    """Directory for caching downloaded models."""
    cache_path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


@pytest.fixture
def vllm_server_port():
    """Port for vLLM server in tests."""
    # Use a high port to avoid conflicts
    return 8765


@pytest.fixture
def generation_config():
    """Configuration for deterministic generation."""
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 50,
        "seed": 42,
    }


@pytest.fixture
def vllm_engine_factory_2gpu(model_name):
    """Factory fixture that defaults to 2 GPUs.

    Usage:
        async with vllm_engine_factory_2gpu() as manager:
            # Uses 2 GPUs by default
            # Access engine via manager.engine
            ...
    """
    def _factory(tensor_parallel_size: int = 2, **kwargs):
        """Create engine with 2 GPUs by default."""
        import argparse

        args = argparse.Namespace(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            disable_log_stats=True,
            enable_log_requests=False,
            **kwargs
        )

        return EngineManager.create_engine(args)

    return _factory


@pytest.fixture
def vllm_engine_factory(model_name):
    """Factory fixture for creating vLLM engines.

    Usage in tests:
        async with vllm_engine_factory() as manager:
            # use manager.engine for generation
            ...
        # automatic cleanup

    Or with custom config:
        async with vllm_engine_factory(tensor_parallel_size=2) as manager:
            # use manager.engine with 2 GPUs
            ...

    Or if you need engine_config:
        async with vllm_engine_factory() as manager:
            # access manager.engine, manager.engine_config, manager.args
            ...
    """
    def _factory(tensor_parallel_size: int = 1, **kwargs):
        """Create engine context manager with test defaults.

        Args:
            tensor_parallel_size: Number of GPUs
            **kwargs: Additional attributes for args object

        Returns:
            Async context manager for EngineManager
        """
        import argparse

        # Create minimal args object with required attributes for AsyncEngineArgs.from_cli_args()
        args = argparse.Namespace(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            disable_log_stats=True,
            enable_log_requests=False,
            # Apply any additional kwargs
            **kwargs
        )

        print("args: ", args)

        return EngineManager.create_engine(args)

    return _factory


@pytest.fixture
def distributed_trainer_helper():
    """Path to the distributed trainer helper script."""
    return Path(__file__).parent / "distributed_trainer_helper.py"


@pytest.fixture
def vllm_engine_helper():
    """Path to the vLLM engine helper script."""
    return Path(__file__).parent / "vllm_engine_helper.py"
