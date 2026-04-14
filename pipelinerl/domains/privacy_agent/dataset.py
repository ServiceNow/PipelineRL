import json
from pathlib import Path
from typing import Any

from .drbench.paths import DEFAULT_ANNOTATIONS_PATH, DEFAULT_CURATED_CHAINS_PATH

DOMAIN_NAME = "privacy_agent"
DEFAULT_DATASET_NAME = "seed20"
DEFAULT_SAMPLE_SIZE = 20


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_selected_curated_rows(
    annotations_path: str | Path,
    curated_path: str | Path,
    sample_size: int,
) -> list[dict]:
    annotations = _load_json(annotations_path)
    accepted_ids = [str(chain_id) for chain_id, payload in annotations.items() if payload.get("s") == "accepted"]

    curated_by_id: dict[str, dict] = {}
    for row in _load_jsonl(curated_path):
        chain_id = str(row.get("chain_id", ""))
        if chain_id:
            curated_by_id[chain_id] = row

    selected: list[dict] = []
    for chain_id in accepted_ids:
        row = curated_by_id.get(chain_id)
        if row is None:
            continue
        selected.append(row)
        if len(selected) >= sample_size:
            break
    return selected


def _flatten_problem(row: dict, idx: int, dataset_name: str) -> dict:
    chain = dict(row.get("original_chain") or {})
    metadata = dict(chain.get("metadata") or {})
    task_id = str(row.get("task_id") or metadata.get("task_id") or "")
    company = metadata.get("company") or ""
    hops = []
    for hop in chain.get("hops", []) or []:
        hops.append(
            {
                "hop_number": int(hop.get("hop_number", len(hops) + 1)),
                "hop_type": hop.get("hop_type"),
                "question": hop.get("question", ""),
                "answer": hop.get("answer", ""),
                "doc_id": hop.get("doc_id"),
            }
        )

    problem = {
        "id": idx,
        "problem_id": f"privacy_agent_{row['chain_id']}",
        "task": chain.get("numbered_questions", ""),
        "dataset": dataset_name,
        "domain": DOMAIN_NAME,
        "chain_id": str(row["chain_id"]),
        "task_id": task_id,
        "company": company,
        "pattern": row.get("pattern") or chain.get("pattern"),
        "numbered_questions": chain.get("numbered_questions", ""),
        "global_question": chain.get("global_question", ""),
        "global_answer": chain.get("global_answer", ""),
        "hops": hops,
        "n_hops": len(hops),
        "expected_doc_ids": [hop["doc_id"] for hop in hops if hop.get("doc_id")],
        "metadata": {
            "task_id": task_id,
            "company": company,
            "curated_confidence": row.get("confidence"),
            "curated_verdict": row.get("verdict"),
            "source_file": row.get("source_file"),
        },
    }
    return problem


def _load_from_materialized_jsonl(path: str | Path) -> list[dict]:
    rows = _load_jsonl(path)
    problems: list[dict] = []
    for idx, row in enumerate(rows):
        row = dict(row)
        row.setdefault("domain", DOMAIN_NAME)
        row.setdefault("dataset", Path(path).stem)
        row.setdefault("id", idx)
        row.setdefault("problem_id", f"privacy_agent_{row.get('chain_id', idx)}")
        problems.append(row)
    return problems


def load_problems(
    dataset_names: list[str] | str | None = None,
    seed: int | None = None,
    annotations_path: str | Path | None = DEFAULT_ANNOTATIONS_PATH,
    curated_path: str | Path | None = DEFAULT_CURATED_CHAINS_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_examples: int | None = None,
    **_: Any,
) -> list[dict]:
    del seed
    annotations_path = annotations_path or DEFAULT_ANNOTATIONS_PATH
    curated_path = curated_path or DEFAULT_CURATED_CHAINS_PATH

    if dataset_names is None:
        return []
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    problems: list[dict] = []
    next_id = 0
    for dataset_name in dataset_names:
        dataset_path = Path(str(dataset_name)).expanduser()
        if dataset_path.exists() and dataset_path.is_file():
            loaded = _load_from_materialized_jsonl(dataset_path)
        else:
            normalized_name = str(dataset_name).strip()
            if normalized_name not in {DEFAULT_DATASET_NAME, "accepted20", "privacy_agent_seed20"}:
                raise ValueError(
                    f"Unsupported privacy_agent dataset '{dataset_name}'. "
                    f"Use '{DEFAULT_DATASET_NAME}' or pass a materialized JSONL file."
                )
            selected_rows = _iter_selected_curated_rows(
                annotations_path=annotations_path,
                curated_path=curated_path,
                sample_size=sample_size,
            )
            loaded = [
                _flatten_problem(row=row, idx=next_id + offset, dataset_name=normalized_name)
                for offset, row in enumerate(selected_rows)
            ]

        for problem in loaded:
            if "id" not in problem:
                problem["id"] = next_id
            next_id = max(next_id, int(problem["id"])) + 1
            problem.setdefault("domain", DOMAIN_NAME)
            problems.append(problem)

    if max_examples is not None:
        problems = problems[: int(max_examples)]
    return problems
