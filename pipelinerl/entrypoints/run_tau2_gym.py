import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from pipelinerl.domains.tau2.client import (
    NEMO_GYM_SHA,
    NEMO_GYM_TITO_PATCH_SHA,
    TAU2_DATA_SHA,
    TAU2_RUNTIME_SHA,
    normalize_openai_base_url,
)

_GYM_APP_PATH = Path("responses_api_models/vllm_model/app.py")
_GYM_APP_BASE_SHA256 = "80daf9e3c9edc6954e323eb0408e70a2af5f8c08579b171ebaa07cdcf3c15ae1"
_GYM_APP_POST_PATCH_SHA256 = "0e0053c94051b8fcdb3cf5f4ec3e775adad8a24afff50326f5890c7bc54aa994"
_GYM_TITO_PATCH_PATH = (
    Path(__file__).resolve().parents[1] / "domains" / "tau2" / "patches" / "nemo_gym_strict_tito.patch"
)


def build_gym_config(
    *,
    policy_urls: list[str],
    policy_model_name: str,
    user_model_url: str,
    user_model_name: str,
    host: str,
    head_port: int,
    service_port_start: int,
    max_steps: int,
    policy_api_key_env: str,
    user_api_key_env: str,
    uses_reasoning_parser: bool,
) -> dict[str, Any]:
    normalized_policy_urls = [normalize_openai_base_url(url) for url in policy_urls]
    if not normalized_policy_urls or len(normalized_policy_urls) != len(set(normalized_policy_urls)):
        raise ValueError("At least one unique policy URL is required")
    normalized_user_url = normalize_openai_base_url(user_model_url)
    if normalized_user_url in normalized_policy_urls:
        raise ValueError("Tau2 user-model endpoint must differ from every policy endpoint")
    if user_model_name == policy_model_name:
        raise ValueError("Tau2 user-model name must differ from the policy model name")
    if not policy_api_key_env.isidentifier() or not user_api_key_env.isidentifier():
        raise ValueError("API key environment-variable names must be valid identifiers")

    num_policies = len(normalized_policy_urls)
    allocated_ports = [head_port, *range(service_port_start, service_port_start + 2 * num_policies + 1)]
    if len(allocated_ports) != len(set(allocated_ports)):
        raise ValueError("Gym head and service ports overlap")

    config: dict[str, Any] = {
        "default_host": host,
        "head_server": {"host": host, "port": head_port},
        "error_on_almost_servers": True,
        "pipelinerl_nemo_gym_sha": NEMO_GYM_SHA,
        "pipelinerl_gym_patch_sha": NEMO_GYM_TITO_PATCH_SHA,
        "pipelinerl_tau2_runtime_sha": TAU2_RUNTIME_SHA,
        "pipelinerl_tau2_data_sha": TAU2_DATA_SHA,
        "pipelinerl_policy_model_name": policy_model_name,
        "pipelinerl_user_model_url": normalized_user_url,
        "pipelinerl_user_model_name": user_model_name,
        "pipelinerl_tau2_user": {
            "responses_api_models": {
                "openai_model": {
                    "entrypoint": "app.py",
                    "host": host,
                    "port": service_port_start + num_policies,
                    "openai_base_url": normalized_user_url,
                    "openai_api_key": f"${{oc.env:{user_api_key_env}}}",
                    "openai_model": user_model_name,
                    "openai_default_headers": {},
                    "drop_input_reasoning_items": False,
                }
            }
        },
    }

    for index, policy_url in enumerate(normalized_policy_urls):
        proxy_name = f"pipelinerl_policy_{index}"
        agent_name = f"pipelinerl_tau2_agent_{index}"
        config[proxy_name] = {
            "responses_api_models": {
                "vllm_model": {
                    "entrypoint": "app.py",
                    "host": host,
                    "port": service_port_start + index,
                    "base_url": policy_url,
                    "api_key": f"${{oc.env:{policy_api_key_env},dummy}}",
                    "model": policy_model_name,
                    "return_token_id_information": True,
                    "uses_reasoning_parser": uses_reasoning_parser,
                }
            }
        }
        config[agent_name] = {
            "responses_api_agents": {
                "tau2": {
                    "entrypoint": "app.py",
                    "host": host,
                    "port": service_port_start + num_policies + 1 + index,
                    "model_server": {"type": "responses_api_models", "name": proxy_name},
                    "user_model_server": {
                        "type": "responses_api_models",
                        "name": "pipelinerl_tau2_user",
                    },
                    "num_workers": 1,
                    "user_llm_args": {},
                    "max_steps": max_steps,
                    "debug": False,
                    "print_step_counts": True,
                }
            }
        }
    return config


def assert_gym_checkout(gym_root: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(gym_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != NEMO_GYM_SHA:
        raise RuntimeError(f"NeMo Gym checkout is {head}, expected {NEMO_GYM_SHA}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_dedicated_gym_checkout(gym_root: Path, run_dir: Path) -> None:
    gym_root = gym_root.resolve()
    run_dir = run_dir.resolve()
    if gym_root == run_dir or not gym_root.is_relative_to(run_dir):
        raise RuntimeError(
            f"NeMo Gym checkout {gym_root} must be dedicated to this run and live inside {run_dir}"
        )


def apply_gym_tito_patch(
    gym_root: Path,
    patch_path: Path = _GYM_TITO_PATCH_PATH,
) -> None:
    patch_path = patch_path.resolve()
    if _sha256(patch_path) != NEMO_GYM_TITO_PATCH_SHA:
        raise RuntimeError("NeMo Gym strict-TITO patch content does not match its pinned SHA256")

    target = gym_root / _GYM_APP_PATH
    target_sha = _sha256(target)
    status = subprocess.check_output(
        ["git", "-C", str(gym_root), "status", "--porcelain=v1", "--untracked-files=no"],
        text=True,
    )
    tracked_paths = {line[3:] for line in status.splitlines()}
    if target_sha == _GYM_APP_POST_PATCH_SHA256:
        if tracked_paths != {str(_GYM_APP_PATH)}:
            raise RuntimeError("Patched NeMo Gym checkout contains unexpected tracked changes")
        return
    if target_sha != _GYM_APP_BASE_SHA256:
        raise RuntimeError("NeMo Gym vllm_model/app.py does not match the pinned base or patched content")
    if tracked_paths:
        raise RuntimeError("NeMo Gym checkout contains tracked changes before strict-TITO patching")

    subprocess.run(
        ["git", "-C", str(gym_root), "apply", "--check", str(patch_path)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(gym_root), "apply", str(patch_path)],
        check=True,
    )
    if _sha256(target) != _GYM_APP_POST_PATCH_SHA256:
        raise RuntimeError("NeMo Gym strict-TITO patched content has the wrong SHA256")


def prepare_tau2_data(gym_root: Path, env: dict[str, str]) -> None:
    """Initialize the shared Tau2 data once before parallel agents start."""
    python = gym_root / ".venv" / "bin" / "python"
    data_dir = gym_root / "responses_api_agents" / "tau2" / "tau2_data"
    code = (
        "from pathlib import Path; "
        "from responses_api_agents.tau2.source import ensure_tau2_data_dir; "
        f"ensure_tau2_data_dir(Path({str(data_dir)!r}))"
    )
    subprocess.run(
        [str(python), "-c", code],
        cwd=gym_root,
        env=env,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the pinned Tau2 Gym service cluster.")
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy-url", action="append", required=True)
    parser.add_argument("--policy-model-name", required=True)
    parser.add_argument("--user-model-url", required=True)
    parser.add_argument("--user-model-name", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--head-port", type=int, default=11000)
    parser.add_argument("--service-port-start", type=int, default=12000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--policy-api-key-env", default="PIPELINERL_LLM_TOKEN")
    parser.add_argument("--user-api-key-env", default="TAU2_USER_API_KEY")
    parser.add_argument(
        "--uses-reasoning-parser",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gym_root = args.gym_root.resolve()
    run_dir = args.run_dir.resolve()
    assert_dedicated_gym_checkout(gym_root, run_dir)
    assert_gym_checkout(gym_root)
    apply_gym_tito_patch(gym_root)
    if not os.environ.get(args.user_api_key_env):
        raise RuntimeError(f"Required user-model API key environment variable {args.user_api_key_env} is unset")

    gym_command = gym_root / ".venv" / "bin" / "gym"
    if not gym_command.exists():
        raise FileNotFoundError(f"NeMo Gym executable not found at {gym_command}; run uv sync in {gym_root}")

    config = build_gym_config(
        policy_urls=args.policy_url,
        policy_model_name=args.policy_model_name,
        user_model_url=args.user_model_url,
        user_model_name=args.user_model_name,
        host=args.host,
        head_port=args.head_port,
        service_port_start=args.service_port_start,
        max_steps=args.max_steps,
        policy_api_key_env=args.policy_api_key_env,
        user_api_key_env=args.user_api_key_env,
        uses_reasoning_parser=args.uses_reasoning_parser,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    output_config = run_dir / "tau2_gym_config.json"
    output_config.write_text(json.dumps(config, indent=2) + "\n")

    env = dict(os.environ)
    env["NEMO_GYM_TAU2_BENCH_REF"] = TAU2_RUNTIME_SHA
    env["NEMO_GYM_TAU2_BENCH_DATA_REF"] = TAU2_DATA_SHA
    prepare_tau2_data(gym_root, env)
    os.chdir(gym_root)
    os.execvpe(
        str(gym_command),
        [str(gym_command), "env", "start", "--config", str(output_config)],
        env,
    )


if __name__ == "__main__":
    main()
