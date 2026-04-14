"""
Unified Citation Registry for DRBench Agent

This module implements a unified citation registry with deferred citation resolution
to eliminate citation duplication and ensure sequential numbering.
"""

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """Information about a document for citation purposes."""

    doc_id: str
    source_info: dict
    underlying_docs: List[str] = field(default_factory=list)
    document_type: str = "regular"  # 'regular', 'ai_synthesis'
    first_appearance: Optional[int] = None
    citation_number: Optional[int] = None


class UnifiedCitationRegistry:
    """
    Single source of truth for all citation management.

    This registry implements deferred citation resolution, maintaining documents
    in [DOC:doc_id] format throughout processing and only converting to [^N]
    format at the final assembly stage.
    """

    def __init__(self) -> None:
        """Initialize the unified citation registry."""
        self.documents: Dict[str, DocumentInfo] = {}
        self.citation_assignments: Dict[str, int] = {}  # doc_id -> citation_number
        self.appearance_order: List[str] = []  # ordered list of first appearances
        self.processed_sections: List[str] = []  # track document order
        self._citation_counter = 0

    def register_document(
        self,
        doc_id: str,
        source_info: dict,
        underlying_docs: Optional[List[str]] = None,
        document_type: str = "regular",
    ) -> None:
        """
        Register a document for eventual citation.

        Args:
            doc_id: Unique document identifier
            source_info: Metadata about the document source
            underlying_docs: List of underlying document IDs (for AI synthesis)
            document_type: Type of document ('regular' or 'ai_synthesis')
        """
        if doc_id in self.documents:
            logger.debug(f"Document {doc_id} already registered, skipping")
            return

        self.documents[doc_id] = DocumentInfo(
            doc_id=doc_id, source_info=source_info, underlying_docs=underlying_docs or [], document_type=document_type
        )

        logger.debug(f"Registered document {doc_id} of type {document_type}")

    def finalize_citations(self, final_text: str) -> Tuple[str, Dict[str, int]]:
        """
        Scan final text and assign citation numbers based on appearance order.

        This method performs a single pass through the complete document,
        assigns sequential citation numbers based on first appearance,
        and handles AI synthesis by expanding to underlying documents.

        Args:
            final_text: The complete document text with [DOC:doc_id] references

        Returns:
            Tuple of (processed_text with [^N] citations, citation_assignments dict)
        """
        # Reset citation counter for fresh numbering
        self._citation_counter = 0
        self.citation_assignments.clear()
        self.appearance_order.clear()

        # Find all [DOC:doc_id] references in order
        doc_pattern = r"\[DOC:([^\]]+)\]"
        doc_matches = list(re.finditer(doc_pattern, final_text))

        # Track which docs have been seen
        seen_docs: Set[str] = set()

        # First pass: Assign citation numbers based on first appearance
        for match in doc_matches:
            doc_id = match.group(1)

            if doc_id not in self.documents:
                logger.warning(f"Document {doc_id} not registered, skipping")
                continue

            doc_info = self.documents[doc_id]

            # Skip AI synthesis documents - they don't get citation numbers
            # Their content should be used directly with preserved underlying citations
            if doc_info.document_type == "ai_synthesis":
                continue

            # Regular documents get citation numbers
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                self._citation_counter += 1
                self.citation_assignments[doc_id] = self._citation_counter
                self.appearance_order.append(doc_id)

        # Second pass: Replace [DOC:doc_id] with [^N] citations
        def replace_doc_reference(match: re.Match[str]) -> str:
            doc_id = match.group(1)

            if doc_id not in self.documents:
                logger.warning(f"Removing unregistered document reference: {doc_id}")
                return ""

            doc_info = self.documents[doc_id]

            # Handle AI synthesis documents by removing them (no citation numbers)
            if doc_info.document_type == "ai_synthesis":
                logger.debug(
                    f"Removing AI synthesis reference {doc_id} (content should have preserved underlying citations)"
                )
                return ""

            # Regular documents get citation numbers
            if doc_id in self.citation_assignments:
                citation_num = self.citation_assignments[doc_id]
                return f"[^{citation_num}]"
            else:
                logger.warning(f"No citation assigned for {doc_id}")
                return ""

        # Perform the replacement
        processed_text = re.sub(doc_pattern, replace_doc_reference, final_text)

        logger.info(f"Finalized {self._citation_counter} citations")

        return processed_text, self.citation_assignments

    def get_citation_number(self, doc_id: str) -> Optional[int]:
        """
        Get assigned citation number for document.

        Args:
            doc_id: Document identifier

        Returns:
            Citation number if assigned, None otherwise
        """
        return self.citation_assignments.get(doc_id)

    def generate_references_section(self) -> str:
        """
        Generate the final references section.

        Returns:
            Formatted references section with citations in numerical order
        """
        if not self.citation_assignments:
            return "\n## References\n\nNo references cited."

        # Create ordered dict by citation number
        ordered_refs = OrderedDict()
        for doc_id, citation_num in sorted(self.citation_assignments.items(), key=lambda x: x[1]):
            if doc_id in self.documents:
                doc_info = self.documents[doc_id]
                # Skip AI synthesis documents as they don't have direct citations
                if doc_info.document_type != "ai_synthesis":
                    ordered_refs[citation_num] = doc_info.source_info

        # Format references
        references = ["\n## References\n"]
        for citation_num, source_info in ordered_refs.items():
            ref_text = self._format_reference(citation_num, source_info)
            references.append(ref_text)

        return "\n\n".join(references)

    def _format_reference(self, citation_num: int, source_info: dict) -> str:
        """
        Format a single reference entry.

        Args:
            citation_num: Citation number
            source_info: Source metadata

        Returns:
            Formatted reference string
        """
        title = source_info.get("title", source_info.get("filename", "Untitled"))
        source = source_info.get("source", source_info.get("source_type", "Unknown source"))
        source_type = source_info.get("type", source_info.get("source_type", ""))
        url = source_info.get("url", "")

        # Truncate long titles
        if len(title) > 200:
            title = title[:197] + "..."

        # Handle different source types
        if source_type == "enterprise_email" or source_type == "email_message":
            # Format: [^1]: Email Subject - Email from sender@domain.com on YYYY-MM-DD
            sender = source_info.get("sender", source_info.get("from", "Unknown Sender"))
            date = source_info.get("date", "Unknown Date")

            # Clean up sender email (extract just email part if it has name)
            if "<" in sender and ">" in sender:
                # Extract email from "Name <email@domain.com>" format
                sender = sender.split("<")[1].split(">")[0]

            # Clean up date (just take the date part, not full timestamp)
            if date and len(date) > 10:
                # Try to extract just the date part from longer timestamps
                import re

                date_match = re.search(r"\d{1,2}\s+\w{3}\s+\d{4}", date)
                if date_match:
                    date = date_match.group(0)
                elif len(date) > 25:
                    date = date[:25] + "..."

            return f"[^{citation_num}]: **{title}** - Email from {sender} on {date}"

        elif source_type == "enterprise_chat":
            # Format: [^3]: Enterprise Chat - Message from user in team/channel
            user = source_info.get("user", "Unknown User")
            channel = source_info.get("channel", "Unknown Channel")
            team = source_info.get("team", "")

            # Build location string
            location = f"Channel: {channel}"
            if team and team != "unknown":
                location = f"Team: {team}, {location}"

            return f"[^{citation_num}]: **{title}** - Enterprise Chat (User: {user}, {location})"

        elif source_type == "internal":
            return f"[^{citation_num}]: **{title}** - Internal Document (`{source_info.get('path', 'Unknown Path')}`)"

        elif source_type == "external":
            if url and url != "Unknown URL":
                return f"[^{citation_num}]: **{title}** - Web Source ([{url}]({url}))"
            else:
                return f"[^{citation_num}]: **{title}** - Web Source"

        elif source_type == "enterprise_file":
            server = source_info.get("server", "Enterprise Server")
            path = source_info.get("path", "Unknown Path")

            # For Nextcloud, clean up path to show user-friendly format
            if "nextcloud" in server.lower() and "/files/" in path:
                # Remove everything up to and including "/files/{username}/"
                if "/files/" in path:
                    # Extract user-friendly path
                    user_friendly_path = path.split("/files/", 1)[1]
                    user_friendly_path = (
                        user_friendly_path.split("/", 1)[1] if "/" in user_friendly_path else user_friendly_path
                    )
                else:
                    user_friendly_path = path
                return f"[^{citation_num}]: **{title}** - {server} File (`{user_friendly_path}`)"
            else:
                return f"[^{citation_num}]: **{title}** - {server} File (`{path}`)"

        elif source_type == "enterprise_api":
            tool = source_info.get("source_tool", "Unknown Tool")
            return f"[^{citation_num}]: **{title}** - Enterprise API ({tool})"

        else:
            # Default format
            if url:
                return f"[^{citation_num}]: **{title}** - {source} ({url})"
            else:
                return f"[^{citation_num}]: **{title}** - {source}"

    def get_statistics(self) -> dict:
        """
        Get citation statistics for debugging.

        Returns:
            Dictionary with citation statistics
        """
        ai_synthesis_count = sum(1 for doc in self.documents.values() if doc.document_type == "ai_synthesis")
        regular_count = len(self.documents) - ai_synthesis_count

        return {
            "total_documents": len(self.documents),
            "regular_documents": regular_count,
            "ai_synthesis_documents": ai_synthesis_count,
            "total_citations": self._citation_counter,
            "unique_cited_documents": len(self.citation_assignments),
            "appearance_order": self.appearance_order[:10],  # First 10 for debugging
        }

    def migrate_from_old_registry(self, old_source_registry: dict, old_ai_insights: dict) -> None:
        """
        Migration utility to convert from old registry format.

        Args:
            old_source_registry: Legacy source registry
            old_ai_insights: Legacy AI synthesis insights
        """
        # Migrate regular sources
        for doc_id, source_data in old_source_registry.items():
            if source_data.get("citation_id") != "skip":
                self.register_document(
                    doc_id=doc_id, source_info=source_data.get("source_info", {}), document_type="regular"
                )

        # Migrate AI synthesis documents
        for doc_id, insight_data in old_ai_insights.items():
            underlying_docs = insight_data.get("source_document_ids", [])
            self.register_document(
                doc_id=doc_id,
                source_info={
                    "title": f"AI Synthesis: {doc_id}",
                    "source": "AI Analysis",
                    "synthesis_method": insight_data.get("synthesis_method", "unknown"),
                },
                underlying_docs=underlying_docs,
                document_type="ai_synthesis",
            )

        logger.info(f"Migrated {len(self.documents)} documents to unified registry")


class LegacyCitationSupport:
    """Support for existing citation formats during migration."""

    @staticmethod
    def detect_format(text: str) -> str:
        """
        Detect if text uses old or new citation format.

        Args:
            text: Text to analyze

        Returns:
            'legacy' for [^N] format, 'new' for [DOC:id] format, 'mixed' for both
        """
        has_legacy = bool(re.search(r"\[\^\d+\]", text))
        has_new = bool(re.search(r"\[DOC:[^\]]+\]", text))

        if has_legacy and has_new:
            return "mixed"
        elif has_legacy:
            return "legacy"
        elif has_new:
            return "new"
        else:
            return "none"

    @staticmethod
    def convert_legacy_format(text: str, doc_id_mapping: Dict[str, str]) -> str:
        """
        Convert legacy [^N] format to [DOC:doc_id] format.

        Args:
            text: Text with legacy citations
            doc_id_mapping: Mapping from citation numbers to doc_ids

        Returns:
            Text with new format citations
        """

        def replace_citation(match: re.Match[str]) -> str:
            citation_num = match.group(1)
            doc_id = doc_id_mapping.get(citation_num)
            if doc_id:
                return f"[DOC:{doc_id}]"
            else:
                logger.warning(f"No doc_id mapping for citation {citation_num}")
                return match.group(0)

        return re.sub(r"\[\^(\d+)\]", replace_citation, text)

    @staticmethod
    def mixed_format_resolution(text: str, registry: UnifiedCitationRegistry) -> str:
        """
        Handle text with both formats during transition.

        Args:
            text: Text with mixed citation formats
            registry: Citation registry for resolution

        Returns:
            Text with unified format
        """
        # First, ensure all citations are in [DOC:] format
        # This is a placeholder - actual implementation would need
        # reverse mapping from citation numbers to doc_ids
        logger.warning("Mixed format detected - manual resolution may be needed")
        return text
