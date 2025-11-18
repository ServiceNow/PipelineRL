import contextlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Union

import joblib
import numpy as np
import psutil
import requests
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

from tapeagents.config import is_debug_mode

logger = logging.getLogger(__name__)


def generate_cuda_device_strings(total_gpus: int, gpus_per_model: int) -> List[str]:
    """
    Generate a list of CUDA device strings for assigning GPUs to models.

    Args:
    - total_gpus (int): The total number of GPUs available.
    - gpus_per_model (int): The number of GPUs required per model.

    Returns:
    - List[str]: A list of strings, each representing the CUDA devices for a model.
    """
    assert (
        total_gpus % gpus_per_model == 0
    ), f"Total GPUs ({total_gpus}) must be divisible by GPUs per model ({gpus_per_model})"

    cuda_device_strings = []
    for start_gpu in range(0, total_gpus, gpus_per_model):
        end_gpu = start_gpu + gpus_per_model
        cuda_devices = ",".join(str(i) for i in range(start_gpu, end_gpu))
        cuda_device_strings.append(cuda_devices)
    return cuda_device_strings


class VLLMServiceManager:
    def __init__(
        self,
        exp_path: Path,
        service_name: str,
        model_name_or_path: Union[str, Path],
        port: int = 8080,
        verbose: bool = True,
        cuda_device: str = "0",
        host: str = "localhost",
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.service_name = service_name
        self.stdout_file_prefix = str(exp_path / f"{service_name}_stdout")
        self.stderr_file_prefix = str(exp_path / f"{service_name}_stderr")
        self.port = port
        self.ports = []
        self.processes = []
        pipeline_parallel_size = kwargs.get("--pipeline-parallel-size", 1)
        tensor_parallel_size = kwargs.get("--tensor-parallel-size", 1)
        self.gpus_per_model_instance = (pipeline_parallel_size) * (tensor_parallel_size)
        logger.info(f"Using {self.gpus_per_model_instance} GPUs per model instance")
        self.verbose = verbose
        self.cuda_device = cuda_device
        self.host = host
        self.kwargs = kwargs
        self.process: Optional[subprocess.Popen] = None
        self.open_files: list[TextIO] = []
        self.stats = {}

    def get_base_urls(self) -> list[str]:
        return [f"http://127.0.0.1:{port}" for port in self.ports]

    def _terminate_with_children(self, process_id: int) -> None:
        try:
            process = psutil.Process(process_id)
            children = process.children(recursive=True)
            # terminate child processes
            for child in children:
                child.terminate()
            _, still_alive = psutil.wait_procs(children, timeout=3)
            for child in still_alive:
                child.kill()

            # terminate parent process
            process.terminate()
            process.wait(timeout=3)
        except psutil.NoSuchProcess:
            logger.warning(f"Could not terminate process with PID: {process_id}, not found")
        except psutil.AccessDenied:
            logger.error(f"Insufficient privileges to terminate process with PID: {process_id}")
        except Exception as e:
            logger.error(f"An error occurred while terminating process: {e}")

    def _wait_for_service(self, process: subprocess.Popen, url, headers=None, timeout=120) -> bool:
        start_time = time.time()
        logger.info(f"-> Waiting for service at {url}")
        while True:
            try:
                response = requests.get(url, headers=headers, timeout=1)
                if response.status_code == 200:
                    logger.info(f"-> Service at {url} is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass

            if time.time() - start_time > timeout:
                logger.error(f"-> Timeout waiting for service at {url}")
                return False

            if process.poll() is not None:
                logger.error("-> Service process has terminated")
                return False
            time.sleep(1.0)

    def _start_service(self) -> None:
        """
        Launch multiple LLMs in parallel.

        Args:
        - model_starter (Callable): A function that starts a single LLM.
        - start_port (int): The port number to start the LLMs on.
        - num_gpu_required (int): The number of GPUs required per model.

        Returns:
        - Tuple[List[subprocess.Popen], List[int]]: A tuple containing the list of assistant processes and the list of ports

        """

        threads = []

        for i, device_number in enumerate(
            generate_cuda_device_strings(torch.cuda.device_count(), self.gpus_per_model_instance)
        ):
            port = self.port + i
            # start_llm(device_number, port, assistant_procs, ports)
            thread = threading.Thread(target=self._start_llm, args=(device_number, port))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=2, min=10))
    def _start_llm(self, cuda_device, port):
        kwargs_str = " ".join([f"--{k} {v}" for k, v in self.kwargs.items()]) if self.kwargs else ""

        cmd = (
            f"OUTLINES_CACHE_DIR=/tmp/.outlines_{cuda_device}_{int(time.time())} "
            f"CUDA_VISIBLE_DEVICES={cuda_device} "
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {self.model_name_or_path} "
            f"--port {port} "
            f"--seed {cuda_device[0]} "
            "--disable-frontend-multiprocessing "
            "--dtype bfloat16 "
            f"{kwargs_str}"
        )

        if self.gpus_per_model_instance > 1:
            cmd = "VLLM_WORKER_MULTIPROC_METHOD=spawn " + cmd

        if self.verbose:
            logger.info(f"Server launcher cmd: {cmd}")

        process_args = {
            "shell": True,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

        if self.verbose:
            stdout_path = self.stdout_file_prefix + f"_{cuda_device}.log"
            stderr_path = self.stderr_file_prefix + f"_{cuda_device}.log"
            logger.info(f"See vLLM outputs in {stdout_path} and {stderr_path}")
            stdout_file = open(stdout_path, "w")
            stderr_file = open(stderr_path, "w")
            process_args["stdout"] = stdout_file
            process_args["stderr"] = stderr_file
            self.open_files.extend([stdout_file, stderr_file])

        try:
            process = subprocess.Popen(cmd, **process_args)
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            raise e

        vllm_url = f"http://{self.host}:{port}/health"
        headers = {"User-Agent": "vLLM Client"}

        start_waiting = time.time()
        if self._wait_for_service(process, vllm_url, headers=headers, timeout=8000):
            logger.info(f"{self.service_name} {self.model_name_or_path} model loaded on port {port}")
        else:
            self._cleanup()
            raise Exception("Failed to start the service")
        end_waiting = time.time()
        self.stats["starting_time"] = end_waiting - start_waiting
        self.ports.append(port)
        self.processes.append(process)

    def _cleanup(self) -> None:
        logger.info(f"Killing {len(self.processes)} vLLM processes")
        threads = []
        for process in self.processes:
            logger.info(f"Terminating process with command {process.args}")
            thread = threading.Thread(target=self._terminate_with_children, args=(process.pid,))
            threads.append(thread)
            thread.start()
        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        for f in self.open_files:
            f.close()

    def __enter__(self) -> "VLLMServiceManager":
        if is_debug_mode():
            logger.info("Running in debug mode, skipping service start")
            self.ports = [8080]
            return self
        self._start_service()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if is_debug_mode():
            logger.info("Running in debug mode, skipping service cleanup")
            return None
        self._cleanup()

    def get_stats(self):
        return self.stats


def setup_logging(output_dir):
    print(f"Setting up logging to {output_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

    # Define log file paths
    info_log = output_dir / "info.log"
    debug_log = output_dir / "debug.log"
    error_log = output_dir / "error.log"

    # Clear any existing handlers
    logger = logging.getLogger()  # get root logger
    logger.handlers = []  # Clear existing handlers
    logger.setLevel(logging.DEBUG)  # Ensure all levels are captured at the root level

    # Create file handlers for each log level
    info_handler = logging.FileHandler(info_log)
    info_handler.setLevel(logging.INFO)

    debug_handler = logging.FileHandler(debug_log)
    debug_handler.setLevel(logging.DEBUG)

    error_handler = logging.FileHandler(error_log)
    error_handler.setLevel(logging.ERROR)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)

    # Create formatters and set them to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%d/%m/%Y %H:%M:%S")

    info_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(info_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(error_handler)
    logger.addHandler(stdout_handler)


def load_state(state_path):
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    else:
        return {"iteration": 0}


def save_state(state, state_path):
    with open(state_path, "w") as f:
        json.dump(state, f)


def clean_up(target_path: Path, state: Dict, state_path: str | Path) -> None:
    os.makedirs(target_path, exist_ok=True)

    def remove_dir(directory: Path):
        if directory.exists() and directory.is_dir():
            shutil.rmtree(directory)

    # Reset the state iteration steps
    state["iteration"] = 0
    save_state(state, state_path)

    logger.info("Cleaning up checkpoints and training state")
    # list of log files to erase
    log_files = list(target_path.glob("*.log"))

    for file in log_files:
        if file.exists():
            # erase the content but not the file
            with open(file, "w"):
                pass
            logger.info(f"{file} erased.")

    # List of directories to remove
    directories = [
        target_path / "llm_calls.sqlite",
        target_path / "dialogue_trace.log",
        target_path / "rollouts",
        target_path / "tapes",
        target_path / "conf",
        target_path / "finetune" / "current",
        target_path / "finetune" / "logs",
        target_path / "finetune" / "intermediate",
        target_path / "finetune" / "training_state",
    ]

    for directory in directories:
        remove_dir(directory)
        logger.info(f"{directory} removed.")


def calculate_stats(stats: Dict[str, list]) -> Dict[str, float]:
    return {
        "max": max([max(stats) for stats in stats.values() if stats]),  # max among the max of each group
        "min": min([min(stats) for stats in stats.values() if stats]),  # min among the min of each group
        "mean_max": np.mean([max(stats) for stats in stats.values() if stats]),  # mean of the max of each group
        "mean_min": np.mean([min(stats) for stats in stats.values() if stats]),  # mean of the min of each group
        "var": np.mean([np.var(stats) for stats in stats.values() if stats]),  # mean of the variance of each group
        "mean": np.mean([np.mean(stats) for stats in stats.values() if stats]),  # mean of the mean of each group
    }


def launch_training(config_dir: str, config_name: str, accelerate_cfg_path: str, use_deepspeed: bool = False) -> None:
    """
    Launch training process with proper GPU configuration and error handling.

    Args:
        config_dir (str): Path to the training config directory
        config_name (str): Name of the config file
        accelerate_cfg_path (str): Path to accelerate config
        use_deepspeed (bool, optional): Whether to use DeepSpeed. Defaults to False.

    Raises:
        ValueError: If no GPUs are available
        RuntimeError: If training process fails
    """
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("No GPUs available for finetuning")

    base_cmd = [
        "accelerate",
        "launch",
        "--mixed_precision=bf16",
        "--config_file",
        accelerate_cfg_path,
        "pipelinerl/domains/miniwob/run_finetune.py",
        "--config-dir",
        config_dir,
        "--config-name",
        config_name,
    ]

    if num_gpus > 1:
        if use_deepspeed:
            base_cmd[2:2] = [
                "--num_processes",
                str(num_gpus),
                "--use_deepspeed",
                "--deepspeed_config_file",
                "conf/accelerate/deepspeed_stage3_bf16.json",
            ]
        else:
            base_cmd[2:2] = [
                "--multi_gpu",
                "--num_processes",
                str(num_gpus),
            ]

    logger.info(f"Launching training with command: {' '.join(base_cmd)}")
    try:
        subprocess.run(
            base_cmd,
            check=True,  # Raises CalledProcessError if return code != 0
            text=True,
            capture_output=False,
        )

    except subprocess.CalledProcessError as e:
        # Capture both stdout and stderr for debugging
        error_msg = (
            f"Training process failed with exit code {e.returncode}\n" f"stdout: {e.stdout}\n" f"stderr: {e.stderr}"
        )
        raise RuntimeError(error_msg) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during training: {str(e)}") from e


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class StepRepeatMonitor:
    def __init__(self, last_step=None, repeat_count=0, max_repeats=4):
        self.last_step = last_step
        self.repeat_count = repeat_count
        self.max_repeats = max_repeats

    def should_stop(self, step) -> bool:
        if step == self.last_step:
            self.repeat_count += 1
            return self.repeat_count > self.max_repeats
        self.repeat_count = 0
        self.last_step = step
        return False
