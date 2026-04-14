"""Vector-store shim for privacy_agent rollouts.

The DRBench agent expects one mutable vector store. For training throughput we split that
into two parts:
- a read-only task-local index built once offline
- a small per-rollout overlay for findings, retrieved web passages, and the final report
"""


import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from .drbench.embeddings import get_embeddings
from .local_index import StaticLocalIndex


class CompositeVectorStore:
    def __init__(
        self,
        static_local_index: StaticLocalIndex,
        overlay_store: Any,
        storage_dir: str | Path,
        task_id: str,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        semantic_threshold: float = 0.7,
        max_length: int = 8192,
        session_cache: Any = None,
        helper_client: Any = None,
        use_remote_local_search: bool = False,
    ):
        self.static_local_index = static_local_index
        self.overlay_store = overlay_store
        self.storage_dir = str(Path(storage_dir))
        self.task_id = task_id
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.semantic_threshold = semantic_threshold
        self.max_length = max_length
        self.session_cache = session_cache
        self.helper_client = helper_client
        self.use_remote_local_search = use_remote_local_search
        self._static_metadata_overrides: dict[str, dict] = {}
        self.search_log: list[dict[str, Any]] = []

    def _record_search(self, query: str, search_type: str, results: list[dict]) -> None:
        self.search_log.append(
            {
                "query": query,
                "search_type": search_type,
                "result_count": len(results),
                "results": [
                    {
                        "doc_id": result.get("doc_id"),
                        "metadata": copy.deepcopy(result.get("metadata", {})),
                        "similarity_score": result.get("similarity_score"),
                        "relevance_score": result.get("relevance_score"),
                    }
                    for result in results[:25]
                ],
            }
        )

    def _merge_static_entry(self, doc_id: str) -> dict:
        doc_info = copy.deepcopy(self.static_local_index.get_document(doc_id))
        overrides = self._static_metadata_overrides.get(doc_id)
        if overrides:
            doc_info.setdefault("metadata", {}).update(overrides)
        return doc_info

    def _direct_doc_lookup(self, doc_id: str) -> list[dict]:
        doc = self.get_document(doc_id)
        if doc is None:
            return []
        return [
            {
                "doc_id": doc_id,
                "similarity_score": 1.0,
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "preview": doc.get("content_preview", doc.get("content", "")[:300]),
                "timestamp": doc.get("timestamp"),
            }
        ]

    def _combined_semantic_search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        # This intentionally mirrors DRBench's exact cosine-search behavior for
        # local documents. The only difference is that local docs come from the
        # prebuilt static index instead of being embedded during the rollout.
        results: list[dict] = []
        query_vector: np.ndarray | None = None
        query_norm = 0.0

        if self.use_remote_local_search and self.helper_client is not None:
            results.extend(
                self.helper_client.search_local(
                    task_id=self.task_id,
                    query=query[: self.max_length],
                    k=top_k * 2,
                    threshold=float(threshold),
                )
            )
        elif self.static_local_index.has_embeddings():
            query_vector = np.array(
                get_embeddings(
                    [query[: self.max_length]],
                    model=self.embedding_model,
                    provider=self.embedding_provider,
                    helper_client=self.helper_client,
                )[0]
            )
            query_norm = float(np.linalg.norm(query_vector))

        if not self.use_remote_local_search and self.static_local_index.has_embeddings() and query_vector is not None:
            static_scores = self.static_local_index.cosine_similarities(query_vector)
            static_top_indices = np.argsort(static_scores)[::-1][: top_k * 2]
            for idx in static_top_indices:
                score = float(static_scores[int(idx)])
                if score < threshold:
                    continue
                doc_id = self.static_local_index.doc_ids[int(idx)]
                doc_info = self._merge_static_entry(doc_id)
                results.append(
                    {
                        "doc_id": doc_id,
                        "similarity_score": score,
                        "content": doc_info["content"],
                        "metadata": doc_info.get("metadata", {}),
                        "preview": doc_info.get("content_preview", doc_info["content"][:300]),
                        "timestamp": doc_info.get("timestamp"),
                    }
                )

        if self.overlay_store.embeddings is not None and len(self.overlay_store.doc_ids) > 0:
            if query_vector is None:
                query_vector = np.array(
                    get_embeddings(
                        [query[: self.max_length]],
                        model=self.embedding_model,
                        provider=self.embedding_provider,
                        helper_client=self.helper_client,
                    )[0]
                )
                query_norm = float(np.linalg.norm(query_vector))
            if not query_norm:
                results.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
                return results[:top_k]
            overlay_norms = np.linalg.norm(self.overlay_store.embeddings, axis=1)
            overlay_scores = np.dot(self.overlay_store.embeddings, query_vector) / (overlay_norms * query_norm)
            overlay_top_indices = np.argsort(overlay_scores)[::-1][: top_k * 2]
            for idx in overlay_top_indices:
                score = float(overlay_scores[int(idx)])
                if score < threshold:
                    continue
                doc_id = self.overlay_store.doc_ids[int(idx)]
                doc_info = self.overlay_store.documents[doc_id]
                results.append(
                    {
                        "doc_id": doc_id,
                        "similarity_score": score,
                        "content": doc_info["content"],
                        "metadata": doc_info.get("metadata", {}),
                        "preview": doc_info.get("content_preview", doc_info["content"][:300]),
                        "timestamp": doc_info.get("timestamp"),
                    }
                )

        results.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
        return results[:top_k]

    def store_document(self, content: str, metadata: dict | None = None, doc_id: str | None = None, check_duplicates: bool = True) -> str:
        return self.overlay_store.store_document(
            content=content,
            metadata=metadata,
            doc_id=doc_id,
            check_duplicates=check_duplicates,
        )

    def batch_store_documents(self, documents: list[dict]) -> list[str]:
        return self.overlay_store.batch_store_documents(documents)

    def _merge_metadata(self, doc_id: str, new_metadata: dict) -> None:
        if doc_id in self.overlay_store.documents:
            self.overlay_store._merge_metadata(doc_id, new_metadata)
            return
        if self.static_local_index.get_document(doc_id) is not None:
            merged = dict(self._static_metadata_overrides.get(doc_id, {}))
            merged.update(new_metadata)
            self._static_metadata_overrides[doc_id] = merged

    def semantic_search(self, query: str, top_k: int = 5, threshold: float | None = None) -> list[dict]:
        if query.startswith("doc_id:"):
            results = self._direct_doc_lookup(query.split(":", 1)[1].strip())[:top_k]
            self._record_search(query=query, search_type="direct_doc_lookup", results=results)
            return results

        if threshold is None:
            threshold = self.semantic_threshold

        has_any_embeddings = self.static_local_index.has_embeddings() or (
            self.overlay_store.embeddings is not None and len(self.overlay_store.doc_ids) > 0
        )
        if not has_any_embeddings:
            return self.keyword_search(query=query, top_k=top_k)

        results = self._combined_semantic_search(query=query, top_k=top_k, threshold=float(threshold))
        self._record_search(query=query, search_type="semantic", results=results)
        return results

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict]:
        if query.startswith("doc_id:"):
            results = self._direct_doc_lookup(query.split(":", 1)[1].strip())[:top_k]
            self._record_search(query=query, search_type="direct_doc_lookup", results=results)
            return results

        query_lower = query.lower()
        results = []

        for doc_id in self.static_local_index.doc_ids:
            doc_info = self._merge_static_entry(doc_id)
            content = doc_info["content"].lower()
            metadata_str = json.dumps(doc_info.get("metadata", {})).lower()
            content_score = content.count(query_lower)
            metadata_score = metadata_str.count(query_lower) * 2
            total_score = content_score + metadata_score
            if total_score > 0:
                results.append(
                    {
                        "doc_id": doc_id,
                        "relevance_score": total_score,
                        "content": doc_info["content"],
                        "metadata": doc_info.get("metadata", {}),
                        "preview": doc_info.get("content_preview", doc_info["content"][:300]),
                        "timestamp": doc_info.get("timestamp"),
                    }
                )

        for doc_id, doc_info in self.overlay_store.documents.items():
            content = doc_info["content"].lower()
            metadata_str = json.dumps(doc_info.get("metadata", {})).lower()
            content_score = content.count(query_lower)
            metadata_score = metadata_str.count(query_lower) * 2
            total_score = content_score + metadata_score
            if total_score > 0:
                results.append(
                    {
                        "doc_id": doc_id,
                        "relevance_score": total_score,
                        "content": doc_info["content"],
                        "metadata": doc_info.get("metadata", {}),
                        "preview": doc_info.get("content_preview", doc_info["content"][:300]),
                        "timestamp": doc_info.get("timestamp"),
                    }
                )

        results.sort(key=lambda item: item.get("relevance_score", 0), reverse=True)
        results = results[:top_k]
        self._record_search(query=query, search_type="keyword", results=results)
        return results

    def search(self, query: str, top_k: int = 5, use_semantic: bool = True) -> list[dict]:
        if use_semantic:
            return self.semantic_search(query=query, top_k=top_k)
        return self.keyword_search(query=query, top_k=top_k)

    def get_document(self, doc_id: str) -> dict | None:
        if doc_id in self.overlay_store.documents:
            return self.overlay_store.get_document(doc_id)
        if self.static_local_index.get_document(doc_id) is not None:
            return self._merge_static_entry(doc_id)
        return None

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self.overlay_store.documents:
            return self.overlay_store.delete_document(doc_id)
        return False

    def get_stats(self) -> dict[str, Any]:
        overlay_stats = self.overlay_store.get_stats()
        return {
            "total_documents": len(self.static_local_index.documents) + len(self.overlay_store.documents),
            "static_documents": len(self.static_local_index.documents),
            "overlay_documents": len(self.overlay_store.documents),
            "has_embeddings": self.static_local_index.has_embeddings() or overlay_stats.get("has_embeddings", False),
            "embedding_dimension": overlay_stats.get("embedding_dimension")
            or self.static_local_index.get_stats().get("embedding_dimension", 0),
            "storage_dir": self.storage_dir,
            "overlay_storage_dir": self.overlay_store.storage_dir,
        }

    def get_retrieval_diagnostics(self, expected_doc_ids: list[str]) -> dict[str, Any]:
        expected_web_docids = {
            doc_id
            for doc_id in expected_doc_ids
            if isinstance(doc_id, str) and doc_id.startswith("web/")
        }
        expected_local_suffixes = {
            doc_id.split("/", 2)[2]
            for doc_id in expected_doc_ids
            if isinstance(doc_id, str) and doc_id.startswith("local/") and len(doc_id.split("/", 2)) == 3
        }

        retrieved_web_docids = {
            metadata.get("docid")
            for metadata in (
                doc_info.get("metadata", {}) for doc_info in self.overlay_store.documents.values()
            )
            if metadata.get("docid")
        }
        retrieved_local_suffixes: set[str] = set()
        for entry in self.search_log:
            for result in entry.get("results", []):
                metadata = result.get("metadata", {}) or {}
                file_path = str(metadata.get("file_path", ""))
                if not file_path:
                    continue
                for suffix in expected_local_suffixes:
                    if file_path.endswith(suffix):
                        retrieved_local_suffixes.add(suffix)

        return {
            "expected_doc_ids": sorted(expected_doc_ids),
            "expected_web_docids": sorted(expected_web_docids),
            "retrieved_web_docids": sorted(docid for docid in retrieved_web_docids if docid),
            "retrieved_expected_web_docids": sorted(expected_web_docids & retrieved_web_docids),
            "expected_local_suffixes": sorted(expected_local_suffixes),
            "retrieved_local_suffixes": sorted(retrieved_local_suffixes),
            "retrieved_expected_local_suffixes": sorted(expected_local_suffixes & retrieved_local_suffixes),
            "search_log": copy.deepcopy(self.search_log),
        }
