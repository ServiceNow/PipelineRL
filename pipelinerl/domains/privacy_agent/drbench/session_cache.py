"""
Session-level cache for content deduplication during research sessions.
Provides in-memory caching to prevent duplicate processing and storage.

This cache prevents duplicate documents from being processed and stored multiple times
during a research session. It tracks:
- Content by hash (SHA-256) to identify identical content
- Source identifiers (file paths, post IDs) to avoid re-processing same sources  
- File paths to prevent re-downloading of files
- Query contexts to merge related research queries
- Access patterns for analytics

When a document is accessed multiple times, the cache:
1. Returns the existing document ID instead of creating a new one
2. Merges query contexts from multiple research queries
3. Updates access counts and timestamps
4. Preserves all research context for better citations
"""

import hashlib
from datetime import datetime
from typing import Dict, Optional, Set, Tuple


class SessionCache:
    """In-memory cache for tracking processed content during a research session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        
        # Caches for different types of content
        self.content_cache: Dict[str, str] = {}  # content_hash -> doc_id
        self.source_cache: Dict[str, str] = {}   # source_identifier -> doc_id
        self.file_cache: Dict[str, Tuple[str, str]] = {}  # file_path -> (content_hash, doc_id)
        
        # Track access patterns
        self.access_count: Dict[str, int] = {}
        self.query_contexts: Dict[str, Set[str]] = {}  # doc_id -> set of query contexts
        
    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def compute_source_hash(self, source_type: str, source_identifier: str) -> str:
        """Compute hash for source identification"""
        source_string = f"{source_type}:{source_identifier}"
        return hashlib.sha256(source_string.encode('utf-8')).hexdigest()
    
    def check_content(self, content: str) -> Optional[str]:
        """
        Check if content has been processed before
        
        Args:
            content: The content to check
            
        Returns:
            doc_id if content exists, None otherwise
        """
        content_hash = self.compute_content_hash(content)
        return self.content_cache.get(content_hash)
    
    def check_source(self, source_type: str, source_identifier: str) -> Optional[str]:
        """
        Check if a source has been processed before
        
        Args:
            source_type: Type of source (e.g., 'nextcloud', 'mattermost')
            source_identifier: Unique identifier (file path, post ID, etc.)
            
        Returns:
            doc_id if source exists, None otherwise
        """
        source_hash = self.compute_source_hash(source_type, source_identifier)
        return self.source_cache.get(source_hash)
    
    def check_file(self, file_path: str) -> Optional[Tuple[str, str]]:
        """
        Check if a file has been processed before
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (content_hash, doc_id) if file exists, None otherwise
        """
        return self.file_cache.get(file_path)
    
    def add_document(
        self, 
        doc_id: str, 
        content: str, 
        source_type: Optional[str] = None,
        source_identifier: Optional[str] = None,
        file_path: Optional[str] = None,
        query_context: Optional[str] = None
    ):
        """
        Add a document to the cache
        
        Args:
            doc_id: Document ID from vector store
            content: Document content
            source_type: Type of source
            source_identifier: Unique source identifier
            file_path: File path if applicable
            query_context: Research query context
        """
        content_hash = self.compute_content_hash(content)
        
        # Add to content cache
        self.content_cache[content_hash] = doc_id
        
        # Add to source cache if provided
        if source_type and source_identifier:
            source_hash = self.compute_source_hash(source_type, source_identifier)
            self.source_cache[source_hash] = doc_id
        
        # Add to file cache if provided
        if file_path:
            self.file_cache[file_path] = (content_hash, doc_id)
        
        # Track access
        self.access_count[doc_id] = self.access_count.get(doc_id, 0) + 1
        
        # Track query contexts
        if query_context:
            if doc_id not in self.query_contexts:
                self.query_contexts[doc_id] = set()
            self.query_contexts[doc_id].add(query_context)
    
    def get_merged_contexts(self, doc_id: str) -> list:
        """Get all query contexts for a document"""
        return list(self.query_contexts.get(doc_id, set()))
    
    def get_access_count(self, doc_id: str) -> int:
        """Get access count for a document"""
        return self.access_count.get(doc_id, 0)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "unique_documents": len(set(self.content_cache.values())),
            "content_hashes": len(self.content_cache),
            "source_entries": len(self.source_cache),
            "file_entries": len(self.file_cache),
            "total_accesses": sum(self.access_count.values()),
            "duplicate_preventions": sum(c - 1 for c in self.access_count.values() if c > 1)
        }
    
    def clear(self):
        """Clear all caches"""
        self.content_cache.clear()
        self.source_cache.clear()
        self.file_cache.clear()
        self.access_count.clear()
        self.query_contexts.clear()