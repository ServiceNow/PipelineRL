"""
SWE dataset preprocessing utility.

Clones repos, extracts gold_file_contents at base_commit, applies token-count
filtering, and optionally computes per-repo file stats for BM25 retrieval.
Run this once per dataset before training; output is saved as a HuggingFace disk dataset.

Usage:
    python -m pipelinerl.domains.swe.swe_preprocessor --config-name=swe/preprocess
"""

import json
import logging
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import git
import hydra
from datasets import Dataset, load_dataset, load_from_disk
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class RepoManager:
    """Clones or updates GitHub repositories to a local directory."""

    def __init__(self, repos_base_dir: str):
        self.repos_base_dir = Path(repos_base_dir)
        os.makedirs(self.repos_base_dir, exist_ok=True)

    def clone_or_update_repo(self, repo_name: str) -> Path:
        repo_url = f"https://github.com/{repo_name}.git"
        local_path = self.repos_base_dir / repo_name.replace("/", "_")

        try:
            if local_path.exists() and (local_path / ".git").exists():
                repo = git.Repo(local_path)
                repo.remotes.origin.fetch()
            else:
                if local_path.exists():
                    logger.warning("Removing broken directory %s before fresh clone", local_path)
                    shutil.rmtree(local_path)
                git.Repo.clone_from(repo_url, local_path)
            return local_path
        except Exception as e:
            logger.error("Failed to process repo %s: %s", repo_name, e)
            raise

    def clone_or_update_repos(self, repo_names: List[str]) -> Dict[str, Path]:
        results: Dict[str, Path] = {}
        failed: List[str] = []
        for name in tqdm(repo_names, desc="Cloning/updating repos"):
            try:
                path = self.clone_or_update_repo(name)
                if (path / ".git").exists():
                    results[name] = path
                else:
                    failed.append(name)
            except Exception:
                failed.append(name)
        if failed:
            logger.warning("Failed to clone/update %d repos: %s", len(set(failed)), list(set(failed)))
        return results


class SwePreprocessor:
    """Processes a SWE-style HuggingFace dataset into a training-ready disk dataset."""

    SOURCE_EXTENSIONS = {
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".php",
        ".rb", ".go", ".rs", ".kt", ".scala", ".swift", ".m", ".mm", ".sh", ".bash",
        ".zsh", ".fish", ".pl", ".r", ".R", ".sql", ".html", ".css", ".scss", ".sass",
        ".less", ".vue", ".jsx", ".tsx", ".json", ".yaml", ".yml", ".xml", ".toml",
        ".ini", ".cfg", ".conf", ".properties", ".gradle", ".cmake", ".make",
        ".dockerfile", ".md", ".rst", ".txt", ".lock", ".requirements", ".pyx", ".ipynb",
        ".pxd", ".pyi", ".pxi.in",
    }
    SKIP_DIRS = {
        "test", "tests", "__pycache__", ".git", ".svn", ".hg", "node_modules",
        ".pytest_cache", ".tox", "venv", ".env", "dist", ".idea", ".vscode",
        "target", "out", "bin", "obj", ".gradle", "coverage", ".coverage",
        ".nyc_output", "htmlcov",
    }

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.repos_base_dir = Path(cfg.repo_path)
        self.dataset_path = Path(cfg.dataset_path)
        self.min_token_threshold = cfg.min_token_threshold
        self.max_token_threshold = cfg.max_token_threshold
        self.num_map_processes = cfg.num_map_processes
        self.tokenizer_model = cfg.tokenizer_model
        self.repo_manager = RepoManager(self.repos_base_dir)
        self.file_stats_cache: Dict = {}
        self.tokenizer = self._init_tokenizer()

    def _init_tokenizer(self):
        try:
            tok = AutoTokenizer.from_pretrained(self.tokenizer_model)
            logger.info("Tokenizer loaded: %s", self.tokenizer_model)
            return tok
        except Exception as e:
            logger.warning("Could not load tokenizer (%s): %s — token filtering disabled", self.tokenizer_model, e)
            return None

    # ── File helpers ─────────────────────────────────────────────────────────

    def _is_source_file(self, filepath: str) -> bool:
        path = Path(filepath)
        if any(p.lower() in self.SKIP_DIRS for p in path.parts):
            return False
        if path.suffix.lower() in self.SOURCE_EXTENSIONS:
            return True
        return not path.suffix and path.name.lower() in {
            "makefile", "dockerfile", "readme", "license", "changelog",
            "requirements", "pipfile", "gemfile", "rakefile",
        }

    def _get_file_content(self, repo_path: Path, commit: str, filepath: str) -> Optional[str]:
        if not (repo_path / ".git").exists():
            return None
        try:
            return git.Repo(repo_path).git.show(f"{commit}:{filepath}")
        except Exception:
            return None

    def _parse_patch(self, patch: str) -> List[str]:
        return re.findall(r"^--- a/(.+)$", patch or "", re.MULTILINE)

    # ── Token filtering ───────────────────────────────────────────────────────

    def _filter_by_token_count(self, example: Dict) -> bool:
        if not self.tokenizer:
            return True
        try:
            contents = json.loads(example.get("gold_file_contents", "{}"))
            if not contents:
                return False
            text = " ".join(contents.values()) + " " + (example.get("problem_statement") or "")
            n = len(self.tokenizer.encode(text, add_special_tokens=False))
            if self.min_token_threshold is not None and n < self.min_token_threshold:
                return False
            if self.max_token_threshold is not None and n > self.max_token_threshold:
                return False
            return True
        except Exception as e:
            logger.error("Token filter error: %s", e)
            return False

    # ── File stats (for BM25, optional) ──────────────────────────────────────

    def _tokenize_content(self, content: str) -> Counter:
        tokens = re.findall(r"[a-zA-Z0-9_]+", content.lower())
        return Counter(t for t in tokens if 2 <= len(t) <= 50 and not (t.isdigit() and len(t) > 4))

    def _get_all_file_stats(self, repo_path: Path, commit: str) -> Dict[str, Dict]:
        key = (str(repo_path), commit)
        if key in self.file_stats_cache:
            return self.file_stats_cache[key]

        stats: Dict[str, Dict] = {}
        if not (repo_path / ".git").exists():
            self.file_stats_cache[key] = stats
            return stats

        try:
            repo = git.Repo(repo_path)
            files = repo.git.execute(["git", "ls-tree", "-r", "--name-only", commit])
            for filepath in (files.strip().split("\n") if files.strip() else []):
                if not filepath or not self._is_source_file(filepath):
                    continue
                try:
                    content = repo.git.show(f"{commit}:{filepath}")
                    content.encode("utf-8")  # skip binary
                    term_counts = self._tokenize_content(content)
                    stats[filepath] = {"path": filepath, "length": len(content), "term_counts": dict(term_counts)}
                except Exception:
                    continue
        except Exception as e:
            logger.error("File stats error for %s@%s: %s", repo_path, commit, e)

        self.file_stats_cache[key] = stats
        return stats

    # ── Per-example processing passes ────────────────────────────────────────

    def _extract_gold_file_contents_only(self, example: Dict, repo_paths: Dict[str, Path]) -> Dict:
        repo = example.get("repo")
        commit = example.get("base_commit")
        patch = example.get("patch")
        contents: Dict[str, str] = {}

        repo_path = repo_paths.get(repo) if repo else None
        if repo_path and commit and patch:
            for fp in self._parse_patch(patch):
                c = self._get_file_content(repo_path, commit, fp)
                if c is not None:
                    contents[fp] = c

        example["gold_file_contents"] = json.dumps(contents)
        return example

    def _add_file_stats_to_example(self, example: Dict, repo_paths: Dict[str, Path]) -> Dict:
        example["all_file_stats"] = json.dumps({})
        example["_invalid_example"] = True

        repo = example.get("repo")
        commit = example.get("base_commit")
        patch = example.get("patch")
        repo_path = repo_paths.get(repo) if repo else None

        if not (repo_path and commit and patch):
            return example

        gold_files = self._parse_patch(patch)
        if any(not self._is_source_file(f) for f in gold_files):
            return example

        all_stats = self._get_all_file_stats(repo_path, commit)
        example["all_file_stats"] = json.dumps(all_stats)
        example["_invalid_example"] = any(f not in all_stats for f in gold_files)
        return example

    # ── Main entry point ─────────────────────────────────────────────────────

    def process(self) -> Set[str]:
        if self.dataset_path.exists() and not self.cfg.force_reprocess:
            logger.info("Found existing dataset at %s, loading from disk", self.dataset_path)
            try:
                ds = load_from_disk(str(self.dataset_path))
                return set(ds["repo"])
            except Exception as e:
                logger.warning("Could not load existing dataset (%s), reprocessing", e)

        logger.info("Loading %s (split=%s) from Hub", self.cfg.hf_dataset_name, self.cfg.hf_split_name)
        dataset = load_dataset(self.cfg.hf_dataset_name, split=self.cfg.hf_split_name)
        logger.info("Loaded %d examples", len(dataset))

        unique_repos = set(dataset["repo"])
        repo_paths = self.repo_manager.clone_or_update_repos(list(unique_repos))

        # Pass 1: extract gold file contents (fast)
        dataset = dataset.map(
            lambda ex: self._extract_gold_file_contents_only(ex, repo_paths),
            batched=False, num_proc=self.num_map_processes,
            load_from_cache_file=False, desc="Extracting gold file contents",
        )

        # Token-count filtering (before expensive file stats)
        if self.tokenizer:
            before = len(dataset)
            dataset = dataset.filter(self._filter_by_token_count, desc="Token-count filtering")
            logger.info("Token filtering: %d → %d examples", before, len(dataset))

        # Pass 2: compute file stats (expensive, only for surviving examples)
        dataset = dataset.map(
            lambda ex: self._add_file_stats_to_example(ex, repo_paths),
            batched=False, num_proc=self.num_map_processes,
            load_from_cache_file=False, desc="Computing file stats",
        )

        before = len(dataset)
        dataset = dataset.filter(lambda ex: not ex.get("_invalid_example", True), desc="Filtering invalid examples")
        dataset = dataset.remove_columns(["_invalid_example"])
        logger.info("Validity filtering: %d → %d examples", before, len(dataset))

        logger.info("Saving to %s", self.dataset_path)
        dataset.save_to_disk(str(self.dataset_path))
        logger.info("Done. %d examples saved.", len(dataset))
        return unique_repos


@hydra.main(config_path="../../../../conf", config_name="swe/preprocess", version_base="1.3.2")
def main(cfg: DictConfig):
    SwePreprocessor(cfg).process()


if __name__ == "__main__":
    main()
