import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import RunConfig
from .embeddings import get_embeddings
from .session_cache import SessionCache

logger = logging.getLogger(__name__)


class VectorStore:
    """Production-ready vector store for document storage and semantic search"""

    def __init__(
        self,
        run_config: RunConfig,
        storage_dir: str = "./vector_store",
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        max_length: int = 8192,
        session_cache: Optional[SessionCache] = None,
        helper_client: Any | None = None,
    ):
        self.storage_dir = storage_dir
        self.run_config = run_config
        self.embedding_model = embedding_model or run_config.get_embedding_model()
        self.embedding_provider = embedding_provider or run_config.get_embedding_provider()
        self.max_length = max_length
        self.session_cache = session_cache
        self.helper_client = helper_client
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        os.makedirs(storage_dir, exist_ok=True)

        # Storage files
        self.documents_file = os.path.join(storage_dir, "documents.json")
        self.embeddings_file = os.path.join(storage_dir, "embeddings.npy")
        self.index_file = os.path.join(storage_dir, "index.json")

        # In-memory storage
        self.documents = {}
        self.embeddings = None
        self.doc_ids = []

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load existing documents and embeddings"""
        try:
            # Load documents metadata
            if os.path.exists(self.documents_file):
                with open(self.documents_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)

            # Load embeddings
            if os.path.exists(self.embeddings_file):
                self.embeddings = np.load(self.embeddings_file)

            # Load index (doc_id to embedding index mapping)
            if os.path.exists(self.index_file):
                with open(self.index_file, "r") as f:
                    index_data = json.load(f)
                    self.doc_ids = index_data.get("doc_ids", [])

        except Exception as e:
            logger.warning(f"Warning: Could not load existing vector store data: {e}")
            self._reset_store()

    def _save_data(self):
        """Save documents and embeddings to disk"""
        with self._lock:
            # Save documents metadata
            with open(self.documents_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)

            # Save embeddings
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)

            # Save index
            index_data = {"doc_ids": self.doc_ids}
            with open(self.index_file, "w") as f:
                json.dump(index_data, f, indent=2)

    def _reset_store(self):
        """Reset the vector store"""
        self.documents = {}
        self.embeddings = None
        self.doc_ids = []

    def _generate_doc_id(self, content: str, metadata: Dict = None) -> str:
        """Generate a unique document ID"""
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        timestamp = datetime.now().isoformat()
        return f"doc_{content_hash[:8]}_{int(datetime.now().timestamp())}"

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for deduplication"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def find_duplicate(self, content: str, metadata: Optional[Dict] = None) -> Optional[Tuple[str, Dict]]:
        """Find duplicate document by content hash

        Args:
            content: Content to check for duplicates
            metadata: Optional metadata to help identify duplicates

        Returns:
            Tuple of (doc_id, existing_metadata) if duplicate found, None otherwise
        """
        content_hash = self._compute_content_hash(content)

        # Search through documents for matching content hash
        for doc_id, doc_info in list(self.documents.items()):
            doc_metadata = doc_info.get("metadata", {})
            if doc_metadata.get("content_hash") == content_hash:
                return doc_id, doc_info

        # Also check by source identifier if provided
        if metadata:
            source_type = metadata.get("source")
            source_identifier = metadata.get("source_identifier") or metadata.get("original_path")

            if source_type and source_identifier:
                for doc_id, doc_info in list(self.documents.items()):
                    doc_metadata = doc_info.get("metadata", {})
                    if doc_metadata.get("source") == source_type and (
                        doc_metadata.get("source_identifier") == source_identifier
                        or doc_metadata.get("original_path") == source_identifier
                    ):
                        return doc_id, doc_info

        return None

    def store_document(
        self, content: str, metadata: Dict = None, doc_id: str = None, check_duplicates: bool = True
    ) -> str:
        """Store a document with semantic embeddings and deduplication

        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID
            check_duplicates: Whether to check for duplicates

        Returns:
            Document ID (new or existing)
        """
        with self._lock:
            if metadata is None:
                metadata = {}

            # Check session cache first if available
            if self.session_cache and check_duplicates:
                cached_doc_id = self.session_cache.check_content(content)
                if cached_doc_id:
                    # Update query contexts if new context provided
                    if "query_context" in metadata:
                        self.session_cache.add_document(
                            cached_doc_id,
                            content,
                            source_type=metadata.get("source"),
                            source_identifier=metadata.get("source_identifier") or metadata.get("original_path"),
                            query_context=metadata.get("query_context"),
                        )
                        # Update the stored document's metadata
                        self._merge_metadata(cached_doc_id, metadata)
                    return cached_doc_id

            # Check for duplicates in vector store
            if check_duplicates:
                duplicate = self.find_duplicate(content, metadata)
                if duplicate:
                    existing_doc_id, existing_doc = duplicate
                    # Merge metadata
                    self._merge_metadata(existing_doc_id, metadata)

                    # Add to session cache
                    if self.session_cache:
                        self.session_cache.add_document(
                            existing_doc_id,
                            content,
                            source_type=metadata.get("source"),
                            source_identifier=metadata.get("source_identifier") or metadata.get("original_path"),
                            file_path=metadata.get("file_path"),
                            query_context=metadata.get("query_context"),
                        )

                    return existing_doc_id

            # Generate new doc ID if not provided
            if doc_id is None:
                doc_id = self._generate_doc_id(content, metadata)

            # Add content hash to metadata
            metadata["content_hash"] = self._compute_content_hash(content)
            metadata["first_seen"] = datetime.now().isoformat()
            metadata["access_count"] = 1

            # Initialize merged_contexts if query_context provided
            if "query_context" in metadata and metadata["query_context"]:
                metadata["merged_contexts"] = [metadata["query_context"]]

            # Store document metadata
            self.documents[doc_id] = {
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "content_preview": content[:300] + "..." if len(content) > 300 else content,
            }

            # Add to session cache
            if self.session_cache:
                self.session_cache.add_document(
                    doc_id,
                    content,
                    source_type=metadata.get("source"),
                    source_identifier=metadata.get("source_identifier") or metadata.get("original_path"),
                    file_path=metadata.get("file_path"),
                    query_context=metadata.get("query_context"),
                )

            # Generate embedding for the content (fail loudly if embedding fails)
            embedding = get_embeddings(
                [content[: self.max_length]],
                model=self.embedding_model,
                provider=self.embedding_provider,
                helper_client=self.helper_client,
            )[0]

            # Add to embeddings matrix
            if self.embeddings is None:
                self.embeddings = np.array([embedding])
                self.doc_ids = [doc_id]
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
                self.doc_ids.append(doc_id)

            # Save to disk
            self._save_data()

            return doc_id

    def _merge_metadata(self, doc_id: str, new_metadata: Dict):
        """Merge new metadata into existing document

        Args:
            doc_id: Document ID
            new_metadata: New metadata to merge
        """
        if doc_id not in self.documents:
            return

        existing_metadata = self.documents[doc_id].get("metadata", {})

        alias_fields = {
            "source_identifier": "source_identifiers",
            "original_path": "original_paths",
            "file_path": "file_paths",
            "relative_path": "relative_paths",
            "filename": "filenames",
        }
        for source_key, alias_key in alias_fields.items():
            values = [
                existing_metadata.get(source_key),
                new_metadata.get(source_key),
            ]
            aliases = existing_metadata.setdefault(alias_key, [])
            if not isinstance(aliases, list):
                aliases = [aliases]
                existing_metadata[alias_key] = aliases
            for value in values:
                if value and value not in aliases:
                    aliases.append(value)

        # Update access count
        existing_metadata["access_count"] = existing_metadata.get("access_count", 1) + 1
        existing_metadata["last_accessed"] = datetime.now().isoformat()

        # Merge query contexts
        if "query_context" in new_metadata and new_metadata["query_context"]:
            if "merged_contexts" not in existing_metadata:
                existing_metadata["merged_contexts"] = []
            if new_metadata["query_context"] not in existing_metadata["merged_contexts"]:
                existing_metadata["merged_contexts"].append(new_metadata["query_context"])

        # Update other metadata fields if they provide new information
        for key, value in new_metadata.items():
            if key not in ["query_context", "timestamp", "first_seen", "access_count", "merged_contexts"]:
                if key not in existing_metadata or existing_metadata[key] != value:
                    # For certain fields, keep a history
                    if key in ["tool_used", "source"]:
                        history_key = f"{key}_history"
                        if history_key not in existing_metadata:
                            existing_metadata[history_key] = (
                                [existing_metadata.get(key)] if key in existing_metadata else []
                            )
                        if value not in existing_metadata[history_key]:
                            existing_metadata[history_key].append(value)
                    existing_metadata[key] = value

        # Save updated data
        self._save_data()

    def semantic_search(self, query: str, top_k: int = 5, threshold: Optional[float] = None) -> List[Dict]:
        """Perform semantic search using embeddings"""
        if threshold is None:
            threshold = self.run_config.semantic_threshold
        if self.embeddings is None or len(self.doc_ids) == 0:
            return self.keyword_search(query, top_k)

        # Get query embedding (fail loudly if embeddings fail)
        query_embedding = get_embeddings(
            [query[: self.max_length]],
            model=self.embedding_model,
            provider=self.embedding_provider,
            helper_client=self.helper_client,
        )[0]
        query_vector = np.array(query_embedding)

        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_vector) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector)
        )

        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                doc_id = self.doc_ids[idx]
                doc_info = self.documents[doc_id]

                results.append(
                    {
                        "doc_id": doc_id,
                        "similarity_score": float(similarities[idx]),
                        "content": doc_info["content"],
                        "metadata": doc_info["metadata"],
                        "preview": doc_info["content_preview"],
                        "timestamp": doc_info["timestamp"],
                    }
                )

        return results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Fallback keyword-based search"""
        results = []
        query_lower = query.lower()

        for doc_id, doc_info in self.documents.items():
            content = doc_info["content"].lower()
            metadata_str = json.dumps(doc_info["metadata"]).lower()

            # Simple scoring based on keyword frequency
            content_score = content.count(query_lower)
            metadata_score = metadata_str.count(query_lower) * 2  # Weight metadata higher
            total_score = content_score + metadata_score

            if total_score > 0:
                results.append(
                    {
                        "doc_id": doc_id,
                        "relevance_score": total_score,
                        "content": doc_info["content"],
                        "metadata": doc_info["metadata"],
                        "preview": doc_info["content_preview"],
                        "timestamp": doc_info["timestamp"],
                    }
                )

        # Sort by relevance score
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:top_k]

    def search(self, query: str, top_k: int = 5, use_semantic: bool = True) -> List[Dict]:
        """Main search interface - uses semantic search by default, falls back to keyword"""
        if use_semantic and self.embeddings is not None:
            return self.semantic_search(query, top_k)
        else:
            return self.keyword_search(query, top_k)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a specific document by ID"""
        return self.documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store"""
        if doc_id not in self.documents:
            return False

        # Remove from documents
        del self.documents[doc_id]

        # Remove from embeddings if it exists
        if doc_id in self.doc_ids:
            idx = self.doc_ids.index(doc_id)
            self.doc_ids.pop(idx)

            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, idx, axis=0)
                if self.embeddings.shape[0] == 0:
                    self.embeddings = None

        # Save updated data
        self._save_data()
        return True

    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "has_embeddings": self.embeddings is not None,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "storage_size_mb": self._get_storage_size(),
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
        }

    def _get_storage_size(self) -> float:
        """Calculate total storage size in MB"""
        total_size = 0
        for file_path in [self.documents_file, self.embeddings_file, self.index_file]:
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        return round(total_size / (1024 * 1024), 2)

    def batch_store_documents(self, documents: List[Dict]) -> List[str]:
        """Store multiple documents efficiently"""
        doc_ids = []
        contents = []

        # Prepare documents
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            doc_id = doc.get("doc_id") or self._generate_doc_id(content, metadata)

            self.documents[doc_id] = {
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "content_preview": content[:300] + "..." if len(content) > 300 else content,
            }

            doc_ids.append(doc_id)
            contents.append(content[: self.max_length])  # Truncate to max length

        # Generate embeddings in batch (fail loudly on error)
        embeddings = get_embeddings(
            contents,
            model=self.embedding_model,
            provider=self.embedding_provider,
            helper_client=self.helper_client,
        )

        if self.embeddings is None:
            self.embeddings = np.array(embeddings)
            self.doc_ids = doc_ids.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, np.array(embeddings)])
            self.doc_ids.extend(doc_ids)

        # Save to disk
        self._save_data()
        return doc_ids

    def deduplicate_store(self, preserve_latest: bool = False) -> Dict[str, int]:
        """Deduplicate existing documents in the vector store

        Args:
            preserve_latest: If True, keep the most recently accessed version

        Returns:
            Statistics about deduplication process
        """
        stats = {
            "total_documents": len(self.documents),
            "duplicates_found": 0,
            "documents_merged": 0,
            "documents_removed": 0,
        }

        # Group documents by content hash
        content_groups = {}
        for doc_id, doc_info in self.documents.items():
            metadata = doc_info.get("metadata", {})
            content_hash = metadata.get("content_hash")

            # Generate content hash if missing
            if not content_hash:
                content_hash = self._compute_content_hash(doc_info["content"])
                metadata["content_hash"] = content_hash

            if content_hash not in content_groups:
                content_groups[content_hash] = []
            content_groups[content_hash].append(doc_id)

        # Process duplicates
        docs_to_remove = []
        for content_hash, doc_ids in content_groups.items():
            if len(doc_ids) > 1:
                stats["duplicates_found"] += len(doc_ids) - 1

                # Sort by access time or creation time
                if preserve_latest:
                    doc_ids.sort(
                        key=lambda d: self.documents[d]["metadata"].get(
                            "last_accessed", self.documents[d]["metadata"].get("timestamp", "")
                        ),
                        reverse=True,
                    )
                else:
                    # Sort by first seen
                    doc_ids.sort(
                        key=lambda d: self.documents[d]["metadata"].get(
                            "first_seen", self.documents[d]["metadata"].get("timestamp", "")
                        )
                    )

                # Keep the first one, merge metadata from others
                primary_doc_id = doc_ids[0]

                for duplicate_doc_id in doc_ids[1:]:
                    # Merge metadata
                    duplicate_metadata = self.documents[duplicate_doc_id].get("metadata", {})
                    self._merge_metadata(primary_doc_id, duplicate_metadata)

                    # Mark for removal
                    docs_to_remove.append(duplicate_doc_id)
                    stats["documents_merged"] += 1

        # Remove duplicate documents and their embeddings
        for doc_id in docs_to_remove:
            self.delete_document(doc_id)
            stats["documents_removed"] += 1

        # Save the cleaned data
        self._save_data()

        return stats
