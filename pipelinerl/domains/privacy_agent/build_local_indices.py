import argparse
import json
import os
import shutil
from pathlib import Path

from .dataset import DEFAULT_DATASET_NAME, load_problems
from .drbench.agent_tools.content_processor import ContentProcessor
from .drbench.agent_tools.local_document_tool import LocalDocumentIngestionTool
from .drbench.config import RunConfig
from .drbench.paths import DEFAULT_LOCAL_INDEX_ROOT
from .drbench.task_loader import Task
from .drbench.vector_store import VectorStore
from .resources import get_shared_helper_client
from .settings import PrivacyAgentSettings


def _write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_embedding_settings(settings: PrivacyAgentSettings) -> None:
    if settings.use_remote_embeddings:
        if not settings.helper_service_url:
            raise RuntimeError(
                "privacy_agent index building is configured to use the remote helper service for embeddings, "
                "but helper_service_url is not set."
            )
        return

    provider = (settings.embedding_provider or "").lower()
    if provider == "vllm" and not os.getenv("VLLM_EMBEDDING_URL"):
        raise RuntimeError(
            "privacy_agent index building was configured with embedding_provider=vllm, "
            "but VLLM_EMBEDDING_URL is not set. "
            "Use embedding_provider=huggingface for local index builds, or set a valid vLLM embedding endpoint."
        )


def build_local_indices_for_tasks(
    task_ids: list[str],
    settings: PrivacyAgentSettings,
    max_workers: int = 4,
    force: bool = False,
) -> list[dict]:
    """Build task-local private-document indices for the given tasks."""

    _validate_embedding_settings(settings)
    index_root = settings.local_index_root
    index_root.mkdir(parents=True, exist_ok=True)
    build_records: list[dict] = []
    helper_client = None
    if settings.use_remote_embeddings and settings.helper_service_url:
        helper_client = get_shared_helper_client(
            base_url=settings.helper_service_url,
            timeout_s=settings.helper_timeout_s,
        )

    for task_id in sorted(set(task_ids)):
        task_output_dir = index_root / task_id
        embeddings_path = task_output_dir / "embeddings.npy"
        if task_output_dir.exists() and not force and embeddings_path.exists():
            build_records.append(
                {
                    "task_id": task_id,
                    "output_dir": str(task_output_dir),
                    "skipped": True,
                }
            )
            continue

        if task_output_dir.exists() and force:
            for child in task_output_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        task_output_dir.mkdir(parents=True, exist_ok=True)

        run_cfg = RunConfig(
            model="privacy_agent_index_builder",
            run_dir=task_output_dir,
            log_searches=False,
            log_prompts=False,
            log_generations=False,
            no_web=True,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            report_style="concise_qa",
        )
        task = Task(Path(task_id) / "config", data_dir=settings.task_data_root)
        vector_store = VectorStore(
            run_config=run_cfg,
            storage_dir=str(task_output_dir),
            embedding_model=run_cfg.get_embedding_model(),
            embedding_provider=run_cfg.get_embedding_provider(),
            helper_client=helper_client,
        )
        workspace_dir = task_output_dir / "workspace"
        content_processor = ContentProcessor(
            workspace_dir=str(workspace_dir),
            model=run_cfg.model or "privacy_agent_index_builder",
            llm_adapter=None,
            run_config=run_cfg,
            vector_store=vector_store,
        )
        ingestion = LocalDocumentIngestionTool(content_processor=content_processor, max_workers=max_workers)
        stats = ingestion.ingest_paths(file_paths=task.get_local_files_list(), recursive=True)
        _write_metadata(
            task_output_dir / "metadata.json",
            {
                "task_id": task_id,
                "embedding_provider": run_cfg.get_embedding_provider(),
                "embedding_model": run_cfg.get_embedding_model(),
                "processed_files": stats.processed_files,
                "total_files": stats.total_files,
                "task_data_root": str(settings.task_data_root),
            },
        )
        build_records.append(
            {
                "task_id": task_id,
                "output_dir": str(task_output_dir),
                "processed_files": stats.processed_files,
                "total_files": stats.total_files,
                "skipped": False,
            }
        )

    return build_records


def _settings_from_args(args: argparse.Namespace) -> PrivacyAgentSettings:
    default_settings = PrivacyAgentSettings()
    return PrivacyAgentSettings(
        annotations_path=(
            Path(args.annotations_path).expanduser()
            if args.annotations_path
            else default_settings.annotations_path
        ),
        curated_path=(
            Path(args.curated_path).expanduser()
            if args.curated_path
            else default_settings.curated_path
        ),
        sample_size=args.sample_size,
        task_data_root=(
            Path(args.task_data_root).expanduser()
            if args.task_data_root
            else default_settings.task_data_root
        ),
        local_index_root=Path(args.index_root).expanduser(),
        helper_service_url=args.helper_service_url or default_settings.helper_service_url,
        helper_timeout_s=float(args.helper_timeout_s),
        use_remote_embeddings=bool(args.use_remote_embeddings),
        use_remote_local_search=False,
        use_remote_browsecomp=False,
        embedding_provider=args.embedding_provider or default_settings.embedding_provider,
        embedding_model=args.embedding_model or default_settings.embedding_model,
        no_web=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild local private-doc indices for the privacy_agent domain.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--index-root", default=str(DEFAULT_LOCAL_INDEX_ROOT))
    parser.add_argument("--task-data-root", default=None)
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--helper-service-url", default=None)
    parser.add_argument("--helper-timeout-s", type=float, default=30.0)
    parser.add_argument("--use-remote-embeddings", dest="use_remote_embeddings", action="store_true")
    parser.add_argument("--no-remote-embeddings", dest="use_remote_embeddings", action="store_false")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--annotations-path", default=None)
    parser.add_argument("--curated-path", default=None)
    parser.set_defaults(use_remote_embeddings=True)
    args = parser.parse_args()

    settings = _settings_from_args(args)

    problems = load_problems(
        [args.dataset_name],
        **settings.dataset_loader_kwargs(),
    )
    task_ids = sorted({problem["task_id"] for problem in problems})
    build_records = build_local_indices_for_tasks(
        task_ids=task_ids,
        settings=settings,
        max_workers=args.max_workers,
        force=args.force,
    )
    for record in build_records:
        if record.get("skipped"):
            print(f"Skipping existing index for {record['task_id']}: {record['output_dir']}")
        else:
            print(
                f"Built {record['task_id']}: {record['processed_files']}/{record['total_files']} files -> "
                f"{record['output_dir']}"
            )


if __name__ == "__main__":
    main()
