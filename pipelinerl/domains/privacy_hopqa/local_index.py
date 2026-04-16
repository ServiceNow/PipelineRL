"""Read-only task-local document index for privacy_hopqa.

This keeps the original DRBench local-document search math: query embedding followed by
exact cosine similarity against stored chunk embeddings. The speedup comes from loading
prebuilt embeddings once, not from changing retrieval behavior.
"""


import json
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import get_embeddings


class StaticLocalIndex:
    def __init__(
        self,
        storage_dir: str | Path,
        documents: dict[str, dict],
        doc_ids: list[str],
        embeddings: np.ndarray,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        semantic_threshold: float = 0.7,
        max_length: int = 8192,
    ):
        self.storage_dir = str(Path(storage_dir))
        self.documents = documents
        self.doc_ids = doc_ids
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.semantic_threshold = semantic_threshold
        self.max_length = max_length
        self._embedding_norms: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        storage_dir: str | Path,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        semantic_threshold: float = 0.7,
        mmap_mode: str | None = "r",
    ) -> "StaticLocalIndex":
        storage_dir = Path(storage_dir).expanduser()
        documents = json.loads((storage_dir / "documents.json").read_text(encoding="utf-8"))
        doc_ids = json.loads((storage_dir / "index.json").read_text(encoding="utf-8")).get("doc_ids", [])
        embeddings = np.load(storage_dir / "embeddings.npy", mmap_mode=mmap_mode)

        metadata_path = storage_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            embedding_model = embedding_model or metadata.get("embedding_model")
            embedding_provider = embedding_provider or metadata.get("embedding_provider")

        return cls(
            storage_dir=storage_dir,
            documents=documents,
            doc_ids=list(doc_ids),
            embeddings=embeddings,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            semantic_threshold=semantic_threshold,
        )

    @property
    def embedding_norms(self) -> np.ndarray:
        # Cache norms because the same task index is reused across many rollouts
        # in the same worker process.
        if self._embedding_norms is None:
            self._embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        return self._embedding_norms

    def has_embeddings(self) -> bool:
        return self.embeddings is not None and len(self.doc_ids) > 0

    def get_document(self, doc_id: str) -> dict | None:
        return self.documents.get(doc_id)

    def cosine_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_vector)
        if not query_norm:
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        return np.dot(self.embeddings, query_vector) / (self.embedding_norms * query_norm)

    def _format_result(self, doc_id: str, score: float) -> dict:
        doc_info = self.documents[doc_id]
        return {
            "doc_id": doc_id,
            "similarity_score": float(score),
            "content": doc_info["content"],
            "metadata": doc_info.get("metadata", {}),
            "preview": doc_info.get("content_preview", doc_info["content"][:300]),
            "timestamp": doc_info.get("timestamp"),
        }

    def semantic_search(self, query: str, top_k: int = 5, threshold: float | None = None) -> list[dict]:
        if threshold is None:
            threshold = self.semantic_threshold
        if not self.has_embeddings():
            return self.keyword_search(query, top_k)

        query_vector = np.array(
            get_embeddings(
                [query[: self.max_length]],
                model=self.embedding_model,
                provider=self.embedding_provider,
            )[0]
        )
        similarities = self.cosine_similarities(query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append(self._format_result(self.doc_ids[int(idx)], float(similarities[idx])))
        return results

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict]:
        query_lower = query.lower()
        results = []
        for doc_id, doc_info in self.documents.items():
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
        return results[:top_k]

    def search(self, query: str, top_k: int = 5, use_semantic: bool = True) -> list[dict]:
        if use_semantic and self.has_embeddings():
            return self.semantic_search(query=query, top_k=top_k)
        return self.keyword_search(query=query, top_k=top_k)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "has_embeddings": self.has_embeddings(),
            "embedding_dimension": int(self.embeddings.shape[1]) if self.has_embeddings() else 0,
            "storage_dir": self.storage_dir,
        }
