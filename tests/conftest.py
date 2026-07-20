"""Pytest configuration and fixtures for vllm1 tests."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def model_name():
    """Model to use for testing."""
    return "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def simple_prompt():
    """Single simple prompt for deterministic testing."""
    return "The capital of France is"


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

        from pipelinerl.vllm1 import EngineManager

        # Create minimal args object with required attributes for AsyncEngineArgs.from_cli_args()
        args = argparse.Namespace(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            disable_log_stats=True,
            enable_log_requests=False,
            # Apply any additional kwargs
            **kwargs
        )

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
