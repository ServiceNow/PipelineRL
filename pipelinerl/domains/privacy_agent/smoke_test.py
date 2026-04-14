"""End-to-end smoke test for the privacy_agent domain.

This script exercises one real privacy_agent rollout against an OpenAI-compatible
LLM endpoint, typically a local vLLM server. It builds the required private-doc
index for the selected task if needed, runs the rollout, and saves a compact
JSON summary of the result.
"""


import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

import aiohttp
from omegaconf import OmegaConf

from pipelinerl.llm import TrainableLLM

from .build_local_indices import build_local_indices_for_tasks
from .dataset import DEFAULT_DATASET_NAME, load_problems
from .drbench.paths import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_BROWSECOMP_CORPUS,
    DEFAULT_BROWSECOMP_INDEX_GLOB,
    DEFAULT_CURATED_CHAINS_PATH,
    DEFAULT_TASK_DATA_ROOT,
)
from .rollouts import generate_privacy_agent_rollout
from .settings import PrivacyAgentSettings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one privacy_agent smoke rollout.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible LLM endpoint, without /v1 suffix.")
    parser.add_argument("--model", required=True, help="Model name served by the chat endpoint.")
    parser.add_argument("--tokenizer-name", default=None, help="Tokenizer name for local token counting.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--problem-index", type=int, default=0)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--chain-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-index-root", default=None)
    parser.add_argument("--task-data-root", default=str(DEFAULT_TASK_DATA_ROOT))
    parser.add_argument("--annotations-path", default=str(DEFAULT_ANNOTATIONS_PATH))
    parser.add_argument("--curated-path", default=str(DEFAULT_CURATED_CHAINS_PATH))
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--concurrent-actions", type=int, default=2)
    parser.add_argument("--capture-mode", default="planning_only")
    parser.add_argument("--answer-match-f1-threshold", type=float, default=0.75)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--max-index-workers", type=int, default=4)
    parser.add_argument("--embedding-provider", default="huggingface")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument(
        "--helper-service-url",
        default=os.getenv("PRIVACY_AGENT_HELPER_URL"),
        help="Remote privacy helper base URL, for example http://helper-host:8012",
    )
    parser.add_argument("--helper-timeout-s", type=float, default=30.0)
    parser.add_argument("--local-tools", action="store_true", help="Disable remote helper usage.")
    parser.add_argument("--browsecomp-enabled", action="store_true", default=False)
    parser.add_argument("--browsecomp-device", default="cuda:0")
    parser.add_argument("--browsecomp-index-glob", default=DEFAULT_BROWSECOMP_INDEX_GLOB)
    parser.add_argument("--browsecomp-corpus", default=DEFAULT_BROWSECOMP_CORPUS)
    parser.add_argument("--browsecomp-model-name", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _select_problem(args: argparse.Namespace) -> dict:
    problems = load_problems(
        [args.dataset_name],
        annotations_path=args.annotations_path,
        curated_path=args.curated_path,
        sample_size=20,
    )

    if args.chain_id:
        for problem in problems:
            if str(problem["chain_id"]) == str(args.chain_id):
                return problem
        raise ValueError(f"Could not find chain_id={args.chain_id} in dataset {args.dataset_name}.")

    if args.task_id:
        for problem in problems:
            if str(problem["task_id"]) == str(args.task_id):
                return problem
        raise ValueError(f"Could not find task_id={args.task_id} in dataset {args.dataset_name}.")

    if args.problem_index < 0 or args.problem_index >= len(problems):
        raise IndexError(
            f"problem-index {args.problem_index} is out of range for dataset {args.dataset_name} "
            f"with {len(problems)} examples."
        )
    return problems[args.problem_index]


def _make_settings(args: argparse.Namespace) -> PrivacyAgentSettings:
    local_index_root = args.local_index_root or str(Path.cwd() / ".privacy_agent" / "local_indices")
    return PrivacyAgentSettings(
        annotations_path=Path(args.annotations_path).expanduser(),
        curated_path=Path(args.curated_path).expanduser(),
        task_data_root=Path(args.task_data_root).expanduser(),
        local_index_root=Path(local_index_root).expanduser(),
        helper_service_url=args.helper_service_url,
        helper_timeout_s=args.helper_timeout_s,
        use_remote_embeddings=not args.local_tools,
        use_remote_local_search=not args.local_tools,
        use_remote_browsecomp=not args.local_tools,
        capture_mode=args.capture_mode,
        answer_match_f1_threshold=args.answer_match_f1_threshold,
        max_iterations=args.max_iterations,
        concurrent_actions=args.concurrent_actions,
        browsecomp_enabled=args.browsecomp_enabled,
        browsecomp_device=args.browsecomp_device,
        browsecomp_index_glob=args.browsecomp_index_glob,
        browsecomp_model_name=args.browsecomp_model_name,
        browsecomp_corpus=args.browsecomp_corpus,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        verbose=args.verbose,
        log_searches=False,
        log_prompts=False,
        log_generations=False,
    )


def _build_cfg(settings: PrivacyAgentSettings, output_dir: Path) -> object:
    return OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "privacy_agent": {
                "annotations_path": str(settings.annotations_path),
                "curated_path": str(settings.curated_path),
                "sample_size": settings.sample_size,
                "task_data_root": str(settings.task_data_root),
                "local_index_root": str(settings.local_index_root),
                "local_index_mmap_mode": settings.local_index_mmap_mode,
                "helper_service_url": settings.helper_service_url,
                "helper_timeout_s": settings.helper_timeout_s,
                "use_remote_embeddings": settings.use_remote_embeddings,
                "use_remote_local_search": settings.use_remote_local_search,
                "use_remote_browsecomp": settings.use_remote_browsecomp,
                "capture_mode": settings.capture_mode,
                "answer_match_f1_threshold": settings.answer_match_f1_threshold,
                "max_iterations": settings.max_iterations,
                "concurrent_actions": settings.concurrent_actions,
                "semantic_threshold": settings.semantic_threshold,
                "report_style": settings.report_style,
                "browsecomp_enabled": settings.browsecomp_enabled,
                "browsecomp_device": settings.browsecomp_device,
                "browsecomp_index_glob": settings.browsecomp_index_glob,
                "browsecomp_model_name": settings.browsecomp_model_name,
                "browsecomp_corpus": settings.browsecomp_corpus,
                "browsecomp_top_k": settings.browsecomp_top_k,
                "browsecomp_max_chars": settings.browsecomp_max_chars,
                "llm_provider": settings.llm_provider,
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model,
                "log_searches": settings.log_searches,
                "log_prompts": settings.log_prompts,
                "log_generations": settings.log_generations,
                "verbose": settings.verbose,
                "no_web": settings.no_web,
            },
        }
    )


async def _run_rollout(cfg, llm: TrainableLLM, problem: dict):
    async with aiohttp.ClientSession() as session:
        return await generate_privacy_agent_rollout(cfg, llm, problem, session)


def main() -> None:
    args = _parse_args()
    problem = _select_problem(args)
    settings = _make_settings(args)
    if (
        settings.use_remote_embeddings
        or settings.use_remote_local_search
        or settings.use_remote_browsecomp
    ) and not settings.helper_service_url:
        raise ValueError(
            "Remote helper mode is enabled. Set --helper-service-url or PRIVACY_AGENT_HELPER_URL, "
            "or pass --local-tools for local-only smoke testing."
        )

    if args.build_index:
        build_records = build_local_indices_for_tasks(
            task_ids=[problem["task_id"]],
            settings=settings,
            max_workers=args.max_index_workers,
            force=args.force_index,
        )
        for record in build_records:
            print(json.dumps(record, indent=2))

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(
        tempfile.mkdtemp(prefix="privacy-agent-smoke-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg(settings, output_dir)

    llm = TrainableLLM(
        model_name=args.model,
        tokenizer_name=args.tokenizer_name or args.model,
        base_url=args.base_url,
        parameters={"temperature": 0.0, "max_tokens": args.max_tokens},
    )

    result = asyncio.run(_run_rollout(cfg, llm, problem))
    metrics = result.metrics.model_dump()
    summary = {
        "problem": {
            "dataset": args.dataset_name,
            "problem_index": args.problem_index,
            "chain_id": problem["chain_id"],
            "task_id": problem["task_id"],
            "company": problem.get("company"),
        },
        "training_text_count": len(result.training_texts),
        "reward": metrics.get("reward"),
        "success": metrics.get("success"),
        "no_error": metrics.get("no_error"),
        "metrics": metrics,
        "first_training_text": (
            {
                "prompt_tokens": result.training_texts[0].prompt_tokens,
                "output_tokens": result.training_texts[0].output_tokens,
                "group_id": result.training_texts[0].group_id,
                "metadata": result.training_texts[0].metadata,
            }
            if result.training_texts
            else None
        ),
        "output_dir": str(output_dir),
    }

    summary_path = output_dir / "smoke_result.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
