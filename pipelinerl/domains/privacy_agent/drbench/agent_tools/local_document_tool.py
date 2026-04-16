"""
Local document processing tools for DrBench Agent
Handles bulk ingestion of document folders and intelligent file search
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..config import RunConfig
from ..local_search_logging import log_local_document_search
from .base import ResearchContext, Tool
from .content_processor import ContentProcessor

logger = logging.getLogger(__name__)

LOCAL_FILE_SEARCH_PURPOSE = """Intelligent search within locally ingested documents using semantic similarity.
        IDEAL FOR: Finding specific information within your document collection, targeted retrieval from local files, contextual document search.
        USE WHEN: You need to find specific information from previously ingested local documents, want to focus search on certain file types, or need document excerpts with source references.
        PARAMETERS: query (search terms), file_type_filter (optional), folder_filter (optional), top_k (number of results)
        OUTPUTS: Relevant document excerpts with file paths, similarity scores, and synthesized findings from local document collection."""  # noqa: E501


@dataclass
class IngestionStats:
    """Statistics for document ingestion process"""

    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_size_mb: float = 0.0
    supported_formats: Set[str] = None
    processing_time_seconds: float = 0.0

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = set()


class LocalDocumentIngestionTool:
    """Simple tool for bulk ingestion of local documents into vector store"""

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        # Text formats
        ".txt",
        ".md",
        ".csv",
        ".tsv",
        ".log",
        # Document formats
        ".pdf",
        ".docx",
        ".doc",
        ".rtf",
        ".odt",
        # Spreadsheet formats
        ".xlsx",
        ".xls",
        ".ods",
        # Presentation formats
        ".pptx",
        ".ppt",
        ".odp",
        # Web formats
        ".html",
        ".htm",
        ".xml",
        ".json",
        ".jsonl",
    }

    def __init__(self, content_processor: ContentProcessor, max_workers: int = 4):
        self.content_processor = content_processor
        self.max_workers = max_workers

    def ingest_paths(
        self,
        folder_paths: Optional[List[str | Path]] = None,
        file_paths: Optional[List[str | Path]] = None,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> IngestionStats:
        """
        Ingest documents from both folders and individual files

        Args:
            folder_paths: List of folder paths to process
            file_paths: List of individual file paths to process
            file_extensions: Optional filter for file extensions
            recursive: Whether to process subdirectories (for folders)

        Returns:
            IngestionStats with processing results
        """
        start_time = datetime.now()
        stats = IngestionStats()

        # Validate extensions
        if file_extensions:
            extensions_set = set(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in file_extensions)
        else:
            extensions_set = self.SUPPORTED_EXTENSIONS

        logger.info(f"Starting document ingestion from {len(folder_paths or [])} folders and {len(file_paths or [])} files")
        logger.info(f"File extensions filter: {extensions_set}")

        # Collect all files to process
        files_to_process = []

        # Process folders
        if folder_paths:
            for folder_path in folder_paths:
                folder = Path(folder_path)
                if not folder.exists():
                    logger.warning(f"Folder does not exist: {folder}")
                    continue

                if not folder.is_dir():
                    logger.warning(f"Path is not a directory: {folder}")
                    continue

                # Collect files from folder
                pattern = "**/*" if recursive else "*"
                for file_path in folder.glob(pattern):
                    if file_path.is_file() and file_path.suffix.lower() in extensions_set:
                        files_to_process.append(file_path)
                        stats.total_size_mb += file_path.stat().st_size / (1024 * 1024)

        # Process individual files
        if file_paths:
            for file_path in file_paths:
                file = Path(file_path)
                if not file.exists():
                    logger.warning(f"File does not exist: {file}")
                    continue

                if not file.is_file():
                    logger.warning(f"Path is not a file: {file}")
                    continue

                if file.suffix.lower() in extensions_set:
                    files_to_process.append(file)
                    stats.total_size_mb += file.stat().st_size / (1024 * 1024)
                else:
                    logger.warning(f"File extension {file.suffix} not in allowed extensions: {file}")

        stats.total_files = len(files_to_process)
        logger.info(f"Found {stats.total_files} files to process ({stats.total_size_mb:.2f} MB)")

        # Process files in parallel
        if files_to_process:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path for file_path in files_to_process
                }

                # Process results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            stats.processed_files += 1
                            stats.supported_formats.add(file_path.suffix.lower())
                        else:
                            if result.get("skipped"):
                                stats.skipped_files += 1
                            else:
                                stats.failed_files += 1
                                logger.warning(f"Failed to process {file_path}: {result.get('error')}")
                    except Exception as e:
                        stats.failed_files += 1
                        logger.error(f"Error processing {file_path}: {e}")

        stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()

        logger.info(f"Document ingestion completed in {stats.processing_time_seconds:.2f}s")
        logger.info(
            f"Results: {stats.processed_files} processed, {stats.skipped_files} skipped, {stats.failed_files} failed"
        )

        return stats

    def ingest_folders(
        self, folder_paths: List[str], file_extensions: Optional[List[str]] = None, recursive: bool = True
    ) -> IngestionStats:
        """
        Ingest all documents from specified folders (legacy method, calls ingest_paths)

        Args:
            folder_paths: List of folder paths to process
            file_extensions: Optional filter for file extensions
            recursive: Whether to process subdirectories

        Returns:
            IngestionStats with processing results
        """
        return self.ingest_paths(folder_paths=folder_paths, file_extensions=file_extensions, recursive=recursive)

    def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file and return result"""

        try:
            # Special handling for JSONL files containing Mattermost or email data
            if file_path.suffix.lower() == ".jsonl":
                query_context = f"Local document from: {file_path.parent}"
                additional_metadata = {
                    "source_type": "local_document",
                    "file_path": str(file_path),
                    "ingestion_time": datetime.now().isoformat(),
                    "file_size_bytes": file_path.stat().st_size,
                    "folder_path": str(file_path.parent),
                    "relative_path": (
                        str(file_path.relative_to(file_path.parents[0]))
                        if len(file_path.parts) > 1
                        else str(file_path.name)
                    ),
                }

                # Check if this is a Mattermost JSONL file
                if self._is_mattermost_jsonl(file_path):
                    result = self._process_mattermost_jsonl(file_path, query_context, additional_metadata)
                    if result.get("success"):
                        return {
                            "success": True,
                            "file_path": str(file_path),
                            "posts_stored": result.get("posts_stored", 0),
                            "doc_ids": result.get("doc_ids", []),
                        }
                    else:
                        return {
                            "success": False,
                            "file_path": str(file_path),
                            "error": result.get("error", "Failed to process Mattermost JSONL"),
                        }

                # Check if this is an email JSONL file
                elif self._is_email_jsonl(file_path):
                    result = self._process_email_jsonl(file_path, query_context, additional_metadata)
                    if result.get("success"):
                        return {
                            "success": True,
                            "file_path": str(file_path),
                            "emails_stored": result.get("emails_stored", 0),
                            "doc_ids": result.get("doc_ids", []),
                        }
                    else:
                        return {
                            "success": False,
                            "file_path": str(file_path),
                            "error": result.get("error", "Failed to process email JSONL"),
                        }

                # If not a special JSONL file, fall through to normal processing

            # Check if file already exists in vector store to avoid duplicates
            if hasattr(self.content_processor, "session_cache") and self.content_processor.session_cache:
                cached_doc_id = self.content_processor.session_cache.check_source("local_file", str(file_path))
                if cached_doc_id:
                    return {
                        "success": True,
                        "skipped": True,
                        "reason": "Already processed",
                        "file_path": str(file_path),
                    }

            # Process file using existing ContentProcessor
            result = self.content_processor.process_file(
                file_path=str(file_path),
                query_context=f"Local document from: {file_path.parent}",
                additional_metadata={
                    "source_type": "local_document",
                    "file_path": str(file_path),
                    "ingestion_time": datetime.now().isoformat(),
                    "file_size_bytes": file_path.stat().st_size,
                    "folder_path": str(file_path.parent),
                    "relative_path": (
                        str(file_path.relative_to(file_path.parents[0]))
                        if len(file_path.parts) > 1
                        else str(file_path.name)
                    ),
                },
            )

            if result.get("success"):
                return {"success": True, "file_path": str(file_path), "doc_id": result.get("doc_id")}
            else:
                return {
                    "success": False,
                    "file_path": str(file_path),
                    "error": result.get("error", "Unknown processing error"),
                }

        except Exception as e:
            return {"success": False, "file_path": str(file_path), "error": str(e)}

    def _is_mattermost_jsonl(self, file_path: Path) -> bool:
        """Check if JSONL file contains Mattermost chat data"""
        try:
            import json

            with open(file_path, "r", encoding="utf-8") as f:
                # Check first few lines for Mattermost-specific structure
                for i, line in enumerate(f):
                    if i >= 10:  # Check first 10 lines
                        break
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            # Look for Mattermost-specific structure (actual posts, not just metadata)
                            if data.get("type") == "post" and "post" in data:
                                return True
                        except json.JSONDecodeError:
                            continue
            return False
        except Exception:
            return False

    def _is_email_jsonl(self, file_path: Path) -> bool:
        """Check if JSONL file contains email data"""
        try:
            import json

            with open(file_path, "r", encoding="utf-8") as f:
                # Check first few lines for email-specific structure
                for i, line in enumerate(f):
                    if i >= 10:  # Check first 10 lines
                        break
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            # Look for email-specific fields
                            if data.get("type") == "email" or (
                                "from" in data and "subject" in data and ("to" in data or "body" in data)
                            ):
                                return True
                        except json.JSONDecodeError:
                            continue
            return False
        except Exception:
            return False

    def _process_mattermost_jsonl(
        self, file_path: Path, query_context: str, additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process Mattermost JSONL file by storing each post individually"""
        try:
            import json

            # Check if file already exists in vector store to avoid duplicates
            if hasattr(self.content_processor, "session_cache") and self.content_processor.session_cache:
                cached_doc_id = self.content_processor.session_cache.check_source("local_file", str(file_path))
                if cached_doc_id:
                    return {
                        "success": True,
                        "skipped": True,
                        "reason": "Already processed",
                        "file_path": str(file_path),
                    }

            stored_count = 0
            failed_count = 0
            doc_ids = []

            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line.strip())

                        # Only process post entries
                        if data.get("type") != "post" or "post" not in data:
                            continue

                        post = data["post"]

                        # Create content matching enterprise tool format
                        content = f"""
Mattermost Message:
User: {post.get('user', 'unknown')}
Channel: {post.get('channel', 'unknown')}
Team: {post.get('team', 'unknown')}
Time: {datetime.fromtimestamp(post.get('create_at', 0) / 1000).isoformat() if post.get('create_at') else 'unknown'}
Message: {post.get('message', '')}
"""

                        # Create metadata matching enterprise tool format (with citation enhancements from commit cb5f912)
                        post_id = f"mattermost_post_{post.get('user', 'unknown')}_{post.get('create_at', line_num)}"
                        metadata = {
                            "source_type": "local_document",
                            "file_path": str(file_path),
                            "ingestion_time": datetime.now().isoformat(),
                            "tool_used": "local_document_ingestion",
                            "source": "mattermost",
                            "source_identifier": post_id,
                            "type": "mattermost_post",  # This is what _extract_source_info looks for
                            "query_context": query_context,
                            "line_number": line_num,
                            "user": post.get("user", "unknown"),
                            "user_name": post.get("user", "unknown"),  # Add user_name for citation formatting
                            "channel": post.get("channel", "unknown"),
                            "channel_name": post.get("channel", "unknown"),  # Add channel_name for citation formatting
                            "team": post.get("team", "unknown"),
                            "team_name": post.get("team", "unknown"),  # Add team_name for citation formatting
                            "create_at": post.get("create_at"),
                            "timestamp": (
                                datetime.fromtimestamp(post.get("create_at", 0) / 1000).isoformat()
                                if post.get("create_at")
                                else "unknown"
                            ),
                            "message": post.get("message", ""),
                            "message_preview": post.get("message", "")[:100],  # Add message preview for citation
                        }

                        # Add additional metadata if provided
                        if additional_metadata:
                            metadata.update(additional_metadata)

                        # Store in vector store directly
                        doc_id = self.content_processor.vector_store.store_document(content=content, metadata=metadata)

                        if doc_id:
                            stored_count += 1
                            doc_ids.append(doc_id)

                            # Register in session cache to prevent duplicates
                            if hasattr(self.content_processor, "session_cache") and self.content_processor.session_cache:
                                self.content_processor.session_cache.register_document(
                                    doc_id=doc_id, source_type="mattermost", source_identifier=post_id
                                )
                        else:
                            failed_count += 1

                    except json.JSONDecodeError as e:
                        failed_count += 1
                        logger.warning(f"Failed to parse JSON on line {line_num} in {file_path}: {e}")
                        continue
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to process line {line_num} in {file_path}: {e}")
                        continue

            logger.info(f"Processed Mattermost JSONL: {stored_count} posts stored, {failed_count} failed")

            return {
                "success": True,
                "file_path": str(file_path),
                "posts_stored": stored_count,
                "posts_failed": failed_count,
                "doc_ids": doc_ids,
            }

        except Exception as e:
            return {"success": False, "file_path": str(file_path), "error": str(e)}

    def _process_email_jsonl(
        self, file_path: Path, query_context: str, additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process email JSONL file by storing each email individually"""
        try:
            import json

            # Check if file already exists in vector store to avoid duplicates
            if hasattr(self.content_processor, "session_cache") and self.content_processor.session_cache:
                cached_doc_id = self.content_processor.session_cache.check_source("local_file", str(file_path))
                if cached_doc_id:
                    return {
                        "success": True,
                        "skipped": True,
                        "reason": "Already processed",
                        "file_path": str(file_path),
                    }

            stored_count = 0
            failed_count = 0
            doc_ids = []

            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line.strip())

                        # Only process email entries
                        if data.get("type") != "email":
                            continue

                        # Create content matching enterprise tool format
                        content = f"""
Email Message:
From: {data.get('from', 'unknown')}
To: {', '.join(data.get('to', [])) if isinstance(data.get('to'), list) else data.get('to', 'unknown')}
Subject: {data.get('subject', 'No Subject')}
Date: {data.get('date', 'unknown')}
Body: {data.get('body', '')}
"""

                        # Create metadata matching enterprise tool format (with citation enhancements from commit cb5f912)
                        email_id = data.get("id", f"email_{line_num}")
                        subject = data.get("subject", "No Subject")
                        sender = data.get("from", "unknown")

                        metadata = {
                            "source_type": "local_document",
                            "file_path": str(file_path),
                            "ingestion_time": datetime.now().isoformat(),
                            "tool_used": "local_document_ingestion",
                            "source": "email_imap",
                            "source_identifier": f"email_{email_id}",
                            "type": "email_message",
                            "query_context": query_context,
                            "line_number": line_num,
                            "id": email_id,
                            "email_id": email_id,  # Add email_id for citation formatting
                            "from": sender,
                            "sender": sender,  # Add sender for citation formatting
                            "from_name": data.get("from_name", ""),
                            "to": data.get("to", []),
                            "cc": data.get("cc", []),
                            "subject": subject,
                            "title": subject,  # Add title for citation formatting
                            "date": data.get("date", "unknown"),
                            "body": data.get("body", ""),
                            "folder": data.get("folder", "unknown"),
                            "read": data.get("read", False),
                            "attachments": data.get("attachments", []),
                            "timestamp": data.get("date", "unknown"),  # Add timestamp for citation formatting
                        }

                        # Add additional metadata if provided
                        if additional_metadata:
                            metadata.update(additional_metadata)

                        # Store in vector store directly
                        doc_id = self.content_processor.vector_store.store_document(content=content, metadata=metadata)

                        if doc_id:
                            stored_count += 1
                            doc_ids.append(doc_id)

                            # Register in session cache to prevent duplicates
                            if hasattr(self.content_processor, "session_cache") and self.content_processor.session_cache:
                                self.content_processor.session_cache.register_document(
                                    doc_id=doc_id, source_type="email", source_identifier=f"email_{email_id}"
                                )
                        else:
                            failed_count += 1

                    except json.JSONDecodeError as e:
                        failed_count += 1
                        logger.warning(f"Failed to parse JSON on line {line_num} in {file_path}: {e}")
                        continue
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to process line {line_num} in {file_path}: {e}")
                        continue

            logger.info(f"Processed email JSONL: {stored_count} emails stored, {failed_count} failed")

            return {
                "success": True,
                "file_path": str(file_path),
                "emails_stored": stored_count,
                "emails_failed": failed_count,
                "doc_ids": doc_ids,
            }

        except Exception as e:
            return {"success": False, "file_path": str(file_path), "error": str(e)}


class LocalFileSearchTool(Tool):
    """Tool for intelligent search within locally ingested documents"""

    def __init__(self, vector_store, model: str, llm_adapter: Any, run_config: RunConfig):
        self.vector_store = vector_store
        self.model = model
        self.llm_adapter = llm_adapter
        self.run_config = run_config

    @property
    def purpose(self) -> str:
        return LOCAL_FILE_SEARCH_PURPOSE

    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute search within local documents"""

        try:
            # Parse search parameters
            params = self._parse_search_query(query)
            params["raw_query"] = query
            search_query = params.get("query", query)
            file_type_filter = params.get("file_type_filter")
            folder_filter = params.get("folder_filter")
            top_k = params.get("top_k", 10)

            # Search vector store with no threshold — local docs have lower similarity
            # than web results, and we filter by source_type anyway
            search_results = self.vector_store.semantic_search(
                query=search_query, top_k=top_k * 2, threshold=0.0
            )

            # Filter results to local documents only
            local_results = []
            for result in search_results:
                metadata = result.get("metadata", {})

                # Only include local documents
                if metadata.get("source_type") != "local_document":
                    continue

                # Exclude synthesized documents to avoid recursive synthesis
                if metadata.get("type") in ["ai_synthesis_with_sources", "ai_synthesis", "research_finding"]:
                    continue

                # Apply file type filter
                if file_type_filter:
                    file_path = metadata.get("file_path", "")
                    if not any(file_path.lower().endswith(ext.lower()) for ext in file_type_filter):
                        continue

                # Apply folder filter
                if folder_filter:
                    folder_path = metadata.get("folder_path", "")
                    if not any(folder_filter_path in folder_path for folder_filter_path in folder_filter):
                        continue

                local_results.append(result)

            # Limit to requested number
            local_results = local_results[:top_k]

            if not local_results:
                result = self.create_error_output(
                    "local_document_search",
                    query,
                    "No relevant local documents found. Make sure documents have been ingested first.",
                )
                log_local_document_search(self.run_config, search_query, params, result)
                return result

            search_mode = str(getattr(self.run_config, "local_document_search_mode", "")).strip().lower()
            if not search_mode:
                search_mode = (
                    "synthesize"
                    if bool(getattr(self.run_config, "local_document_search_synthesis", False))
                    else "retrieve_only"
                )
            if search_mode not in {"synthesize", "retrieve_only"}:
                search_mode = "retrieve_only"
            synthesis_enabled = search_mode == "synthesize"
            synthesis = None

            # Store synthesis in vector store with source tracking when enabled.
            source_doc_ids = [result.get("doc_id") for result in local_results if result.get("doc_id")]
            if synthesis_enabled:
                synthesis = self._synthesize_local_results(local_results, search_query, context)
                if self.vector_store and synthesis and source_doc_ids:
                    synthesis_metadata = {
                        "tool_used": "local_document_search",
                        "type": "ai_synthesis_with_sources",
                        "source": "local_synthesis",
                        "query_context": search_query,
                        "synthesis_method": "local_document_search",
                        "source_document_ids": source_doc_ids,
                        "timestamp": datetime.now().isoformat(),
                    }

                    self.vector_store.store_document(content=synthesis, metadata=synthesis_metadata)

            # Extract file statistics
            file_paths = set()
            file_types = set()
            folders = set()

            for result in local_results:
                metadata = result.get("metadata", {})
                if metadata.get("file_path"):
                    file_paths.add(metadata["file_path"])
                    file_types.add(Path(metadata["file_path"]).suffix)
                if metadata.get("folder_path"):
                    folders.add(metadata["folder_path"])

            result = self.create_success_output(
                tool_name="local_document_search",
                query=search_query,
                synthesis=synthesis,
                files_searched=len(file_paths),
                file_types_found=list(file_types),
                folders_searched=list(folders),
                results_count=len(local_results),
                data_retrieved=True,
                stored_in_vector=bool(synthesis_enabled and synthesis),  # Prevent duplicate storage as research_finding
                results={
                    "synthesis": synthesis,
                    "mode": search_mode,
                    "local_documents": [
                        {
                            "file_path": r.get("metadata", {}).get("file_path"),
                            "content_excerpt": (
                                r.get("content", "")[:500] + "..."
                                if len(r.get("content", "")) > 500
                                else r.get("content", "")
                            ),
                            "relevance_score": r.get("score", 0.0),
                        }
                        for r in local_results[:5]  # Top 5 for display
                    ],
                },
            )
            log_local_document_search(
                self.run_config,
                search_query,
                params,
                result,
                extra={"mode": search_mode},
            )
            return result

        except Exception as e:
            logger.error(f"Local file search failed: {e}")
            result = self.create_error_output("local_document_search", query, str(e))
            log_local_document_search(self.run_config, query, {"raw_query": query}, result)
            return result

    def _parse_search_query(self, query: str) -> Dict[str, Any]:
        """Parse search parameters from query string"""

        params = {"query": query}

        # Extract filters if present
        # Format: "search_term file_type_filter=['.pdf', '.docx'] folder_filter=['folder1'] top_k=5"

        if "file_type_filter=" in query:
            try:
                start = query.find("file_type_filter=") + len("file_type_filter=")
                end = query.find("]", start) + 1
                filter_part = query[start:end]
                file_type_filter = eval(filter_part) if filter_part else None
                params["file_type_filter"] = file_type_filter
                # Remove filter from query
                params["query"] = query.replace(f"file_type_filter={filter_part}", "").strip()
            except Exception as e:
                logger.warning(f"Could not parse file_type_filter from query: {e}")

        if "folder_filter=" in query:
            try:
                start = query.find("folder_filter=") + len("folder_filter=")
                end = query.find("]", start) + 1
                filter_part = query[start:end]
                folder_filter = eval(filter_part) if filter_part else None
                params["folder_filter"] = folder_filter
                # Remove filter from query
                params["query"] = query.replace(f"folder_filter={filter_part}", "").strip()
            except Exception as e:
                logger.warning(f"Could not parse folder_filter from query: {e}")

        if "top_k=" in query:
            try:
                start = query.find("top_k=") + len("top_k=")
                # Find the next space or end of string
                end = query.find(" ", start)
                if end == -1:
                    end = len(query)
                top_k_str = query[start:end]
                params["top_k"] = int(top_k_str)
                # Remove from query
                params["query"] = query.replace(f"top_k={top_k_str}", "").strip()
            except Exception as e:
                logger.warning(f"Could not parse top_k from query: {e}")

        return params

    def _synthesize_local_results(self, results: List[Dict], query: str, context: ResearchContext) -> str:
        """Synthesize search results from local documents with proper citations"""
        if not results:
            return "No local documents found matching the query."

        # Prepare content with document IDs for citations
        doc_content_with_ids = []
        for result in results[:10]:  # Limit for token management
            content = result.get("content", "")
            doc_id = result.get("doc_id")
            metadata = result.get("metadata", {})
            relative_path = metadata.get("relative_path", metadata.get("filename", "Unknown"))

            if content and doc_id:
                doc_content_with_ids.append(
                    {"doc_id": doc_id, "file_name": relative_path, "content": content[:1000]}  # Limit content size
                )

        if not doc_content_with_ids:
            return "Retrieved local documents but no content available for synthesis."

        # Generate synthesis prompt with document attribution
        synthesis_prompt = f"""
Based on the research query: "{query}"
And the original research question: "{context.original_question}"

Documents available for analysis:
{json.dumps(doc_content_with_ids, indent=2)}

CITATION REQUIREMENTS:
- EXACT FORMAT: [DOC:doc_id] - with colon after DOC
- Use INDIVIDUAL citations: [DOC:doc_1][DOC:doc_2] NOT [DOC:doc_1; DOC:doc_2]
- Cite EVERY claim with source documents
- NEVER make claims without document support

Synthesize the following information into key insights:

Provide:
1. Key findings directly relevant to the query
2. Important patterns or trends identified
3. Contradictions or conflicts in the information
4. Gaps that still need research
5. Actionable insights or conclusions

Be comprehensive but concise. Focus on insights that directly address the query.
Every claim MUST have [DOC:doc_id] citations.
"""

        try:
            synthesis = self.llm_adapter.generate_text(
                synthesis_prompt,
                model=self.model,
                log_name="local_doc_synthesis",
            )
            # Fix common citation formatting issues
            return self._fix_malformed_citations(synthesis)
        except Exception as e:
            return f"Error generating synthesis: {e}"

    def _fix_malformed_citations(self, text: str) -> str:
        """Fix common citation formatting mistakes"""
        import re

        # Fix [DOC doc_id] -> [DOC:doc_id]
        text = re.sub(r"\[DOC\s+([^\]]+)\]", r"[DOC:\1]", text)

        # Fix [DOC_id] -> [DOC:id]
        text = re.sub(r"\[DOC_([^\]]+)\]", r"[DOC:\1]", text)

        return text
