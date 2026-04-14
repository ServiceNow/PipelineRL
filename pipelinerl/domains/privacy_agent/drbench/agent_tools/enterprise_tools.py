"""Refactored Enterprise API Tool using modular architecture"""

import logging
from typing import Any, Dict, List, Optional

from .base import ResearchContext, Tool
from .content_processor import ContentProcessor
from .enterprise.adapters import EmailAdapter, FileBrowserAdapter, MattermostAdapter, NextcloudAdapter
from .enterprise.discovery import DiscoveryCache, ServiceDiscovery
from .enterprise.utils import extract_search_terms

logger = logging.getLogger(__name__)


# Service adapter registry
SERVICE_ADAPTERS = {
    "nextcloud": NextcloudAdapter,
    "mattermost": MattermostAdapter,
    "filebrowser": FileBrowserAdapter,
    "email_imap": EmailAdapter,
}


class EnterpriseAPITool(Tool):
    """Enhanced enterprise API tool using modular service adapters"""

    @property
    def purpose(self) -> str:
        return """Access to proprietary internal enterprise data, documents, communications, and business systems for competitive advantage.
        IDEAL FOR: Internal performance metrics, employee communications, confidential documents, customer interactions, project files, financial data, strategic plans, and proprietary research that provides unique organizational insights.
        USE WHEN: Research requires internal context, organizational knowledge, confidential data, employee perspectives, historical performance, internal benchmarks, or proprietary information not available externally.
        PARAMETERS: query (internal search terms - e.g., 'customer satisfaction Q4 2024', 'project alpha budget allocation', 'sales team feedback on new product')
        OUTPUTS: Privileged internal content with organizational context, employee insights, confidential data, and internal metrics that provide strategic competitive intelligence unavailable through external sources."""

    def __init__(self, env, content_processor: ContentProcessor, model: str, session_cache=None):
        self.env = env
        self.content_processor = content_processor
        self.model = model
        self.session_cache = session_cache

        # Initialize discovery cache and service discovery
        self.discovery_cache = DiscoveryCache()
        self.service_discovery = ServiceDiscovery(self.discovery_cache)

        # Cache for service adapters
        self._adapters = {}


    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute enterprise API interactions using service adapters"""
        try:
            # Get available services
            available_apps = self._get_available_apps()
            if not available_apps:
                return self.create_error_output("enhanced_enterprise_api", query, "No enterprise services available")

            # Initialize service adapters
            adapters = self._get_or_create_adapters(available_apps)
            if not adapters:
                return self.create_error_output(
                    "enhanced_enterprise_api", query, "Failed to initialize service adapters"
                )

            # Extract search terms
            search_terms = extract_search_terms(query)

            # Execute actions across services
            results = []
            processed_files = []
            successful_actions = 0

            for service_name, adapter in adapters.items():
                try:
                    # Execute search on each service
                    logger.info(f"  👀 Searching on {service_name} with terms: {search_terms}")
                    service_results = adapter.search(search_terms, {"query": query})

                    if service_results:
                        successful_actions += 1

                        # Process files if applicable
                        if service_name in ["nextcloud", "filebrowser"]:
                            for file_info in service_results[:10]:  # Limit processing
                                file_result = self._process_file(adapter, file_info, query, context)
                                if file_result and file_result.get("success"):
                                    processed_files.append(file_result.get("file_path"))

                        mattermost_posts_stored = 0
                        if service_name == "mattermost":
                            mattermost_posts_stored = self._store_mattermost_posts(service_results, query)

                        emails_stored = 0
                        if service_name == "email_imap":
                            emails_stored = self._store_emails(service_results, query)

                        result_item = {
                            "service": service_name,
                            "success": True,
                            "items_found": len(service_results),
                            "results": service_results,
                        }

                        if service_name == "mattermost":
                            result_item["mattermost_posts_stored"] = mattermost_posts_stored
                        elif service_name == "email_imap":
                            result_item["emails_stored"] = emails_stored

                        results.append(result_item)

                except Exception as e:
                    logger.error(f"Error executing on {service_name}: {e}")
                    results.append({"service": service_name, "success": False, "error": str(e)})

            total_stored = len(processed_files)
            total_stored += sum(result.get("mattermost_posts_stored", 0) for result in results)
            total_stored += sum(result.get("emails_stored", 0) for result in results)

            return self.create_success_output(
                tool_name="enhanced_enterprise_api",
                query=query,
                results=results,
                data_retrieved=successful_actions > 0,
                services_queried=len(adapters),
                successful_services=successful_actions,
                processed_files=processed_files,
                files_processed=len(processed_files),
                search_terms=search_terms,
                stored_in_vector=True,
                content_stored_in_vector=total_stored,
            )

        except Exception as e:
            return self.create_error_output(
                "enhanced_enterprise_api", query, f"Enterprise API execution failed: {str(e)}"
            )

    def _get_available_apps(self) -> Dict[str, Any]:
        """Get available enterprise applications"""
        try:
            return self.env.get_available_apps()
        except AttributeError:
            return {}

    def _get_or_create_adapters(self, available_apps: Dict) -> Dict[str, Any]:
        """Get or create service adapters for available apps"""
        adapters = {}

        for name, config in available_apps.items():
            service_name = name.lower()

            # Skip if no adapter available
            if service_name not in SERVICE_ADAPTERS:
                logger.debug(f"No adapter available for service: {name}")
                continue

            # Use cached adapter or create new one
            if service_name not in self._adapters:
                try:
                    adapter_class = SERVICE_ADAPTERS[service_name]
                    adapter = adapter_class(config)

                    # Discover capabilities with caching
                    self.service_discovery.discover_service(adapter)

                    self._adapters[service_name] = adapter
                except Exception as e:
                    logger.error(f"Failed to create adapter for {name}: {e}")
                    continue

            adapters[service_name] = self._adapters[service_name]

        return adapters

    def _process_file(self, adapter, file_info: Dict, query: str, context: ResearchContext) -> Optional[Dict]:
        """Download and process a file using the content processor"""
        try:
            file_path = file_info.get("path", "")
            file_name = file_info.get("name", "")

            if not file_path:
                return None

            # Download file
            download_result = adapter.download_file(file_path)
            if not download_result.get("success"):
                return None

            # Save to temp file and process
            import os
            import tempfile

            _, ext = os.path.splitext(file_name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".txt", mode="wb") as temp_file:
                temp_file.write(download_result["content"])
                temp_file_path = temp_file.name

            try:
                # Process with content processor
                result = self.content_processor.process_file(
                    file_path=temp_file_path,
                    query_context=f"{adapter.service_name} file: {file_name} | Query: {query}",
                    additional_metadata={
                        "tool_used": "enhanced_enterprise_api",
                        "source": adapter.service_name,
                        "original_path": file_path,
                        "source_identifier": file_path,  # Add source identifier for deduplication
                        "file_name": file_name,
                        "service_name": adapter.service_name,
                    },
                )

                if result.get("success"):
                    context.files_created.append(result.get("file_path"))

                return result

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Failed to process file {file_info.get('name', 'unknown')}: {e}")
            return None

    def _store_mattermost_posts(self, posts: List[Dict], query: str):
        """Store Mattermost posts in vector store"""
        if not self.content_processor or not self.content_processor.vector_store:
            return

        stored_count = 0
        for post in posts[:10]:  # Limit storage
            try:
                content = f"""
Mattermost Message:
User: {post.get('user_name', post.get('user_id', 'unknown'))}
Channel: {post.get('channel_name', post.get('channel_id', 'unknown'))}
Time: {post.get('timestamp', 'unknown')}
Message: {post.get('message', '')}
"""

                # Enhance metadata with extracted sources
                post_id = post.get("id", f"{post.get('user_id', 'unknown')}_{post.get('timestamp', 'unknown')}")
                enhanced_metadata = {
                    "tool_used": "enhanced_enterprise_api",
                    "source": "mattermost",
                    "source_identifier": f"mattermost_post_{post_id}",  # Add unique identifier
                    "type": "message",
                    "query_context": query,
                    **post,
                }

                doc_id = self.content_processor.vector_store.store_document(
                    content=content,
                    metadata=enhanced_metadata,
                )

                if doc_id:
                    stored_count += 1

            except Exception as e:
                logger.error(f"Failed to store Mattermost post: {e}")

        logger.info(f"Stored {stored_count} Mattermost posts in vector store")
        return stored_count

    def _store_emails(self, emails: List[Dict], query: str) -> int:
        """Store email messages in vector store"""
        if not self.content_processor or not self.content_processor.vector_store:
            return 0

        stored_count = 0
        for email_data in emails[:10]:  # Limit storage
            try:
                content = f"""
Email Message:
From: {email_data.get('from', 'unknown')}
To: {email_data.get('to', 'unknown')}
Subject: {email_data.get('subject', 'No Subject')}
Date: {email_data.get('date', 'unknown')}
Preview: {email_data.get('preview', '')}
"""

                # Enhance metadata with email details
                email_id = email_data.get("id", "unknown")
                enhanced_metadata = {
                    "tool_used": "enhanced_enterprise_api",
                    "source": "email_imap",
                    "source_identifier": f"email_{email_id}",
                    "type": "email_message",
                    "query_context": query,
                    **email_data,
                }

                doc_id = self.content_processor.vector_store.store_document(
                    content=content,
                    metadata=enhanced_metadata,
                )

                if doc_id:
                    stored_count += 1

            except Exception as e:
                logger.error(f"Failed to store email: {e}")

        logger.info(f"Stored {stored_count} emails in vector store")
        return stored_count
