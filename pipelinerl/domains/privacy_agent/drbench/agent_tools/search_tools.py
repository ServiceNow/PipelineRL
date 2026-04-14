import json
import logging
from datetime import datetime
from typing import Any, Dict

import requests

from .base import ResearchContext, Tool
from ..config import RunConfig
from ..internet_search_logging import log_internet_search

logger = logging.getLogger(__name__)


class InternetSearchTool(Tool):
    """Tool for searching the internet and fetching URL content using Serper or similar service"""

    @property
    def purpose(self) -> str:
        return """External market research, competitive intelligence, and public data analysis. 
        IDEAL FOR: Market trends, competitor analysis, industry reports, public research papers, news articles, regulatory information, and technology comparisons.
        USE WHEN: Research requires public/external sources, competitor benchmarking, market validation, industry context, or recent developments.
        PARAMETERS: query (specific search terms work best - e.g., 'AI market size 2024', 'competitor pricing strategies', 'regulatory changes fintech')
        OUTPUTS: Search results with URLs, snippets, and relevant content that gets automatically processed and stored for synthesis."""

    def __init__(
        self,
        api_key: str,
        run_config: RunConfig,
        service: str = "serper",
        vector_store: Any = None,
        content_processor: Any = None,
    ):
        self.api_key = api_key
        self.run_config = run_config
        self.service = service
        self.base_url = "https://google.serper.dev/search" if service == "serper" else None
        self.vector_store = vector_store
        self.content_processor = content_processor

    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute internet search with URL content fetching and standardized output"""

        if self.run_config.no_web:
            result = self.create_error_output(
                "internet_search",
                query,
                "Web access disabled (--no-web).",
            )
            log_internet_search(
                self.run_config,
                tool="internet_search",
                query=query,
                params={"query": query},
                result=result,
            )
            return result

        if not self.api_key:
            result = self.create_error_output(
                "internet_search",
                query,
                "API key not provided for internet search",
            )
            log_internet_search(
                self.run_config,
                tool="internet_search",
                query=query,
                params={"query": query},
                result=result,
            )
            return result

        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = json.dumps({"q": query, "num": 10})  # Number of results

        logger.debug("InternetSearchTool query=%r", query)

        try:
            response = requests.post(self.base_url, headers=headers, data=payload, timeout=30)
            
            response.raise_for_status()
            results = response.json()

            # Extract key information
            search_results = []
            for result in results.get("organic", []):
                search_results.append(
                    {"title": result.get("title"), "link": result.get("link"), "snippet": result.get("snippet")}
                )

            # Include additional info if available
            additional_info = {}
            if "answerBox" in results:
                additional_info["answer_box"] = results["answerBox"]
            if "knowledgeGraph" in results:
                additional_info["knowledge_graph"] = results["knowledgeGraph"]

            # Enhanced: Fetch content from top URLs if content processor is available
            fetched_content = []
            content_stored_count = 0
            
            if self.content_processor and search_results:
                logger.info("Fetching content from top %s search results...", min(5, len(search_results)))
                
                for i, result in enumerate(search_results[:5]):  # Fetch top 5 URLs
                    url = result.get("link")
                    if url:
                        try:
                            # Use content processor to fetch and store URL content
                            content_result = self.content_processor.process_url(
                                url=url,
                                query_context=f"Search query: {query}"
                            )
                            
                            if content_result.get("success"):
                                fetched_content.append({
                                    "url": url,
                                    "title": result.get("title"),
                                    "content_length": content_result.get("content_length", 0),
                                    "stored_in_vector": content_result.get("stored_in_vector", False),
                                    "doc_id": content_result.get("doc_id"),
                                    "file_path": content_result.get("file_path")
                                })
                                
                                if content_result.get("stored_in_vector"):
                                    content_stored_count += 1
                                    
                                # Also store search result metadata linking to content
                                if self.vector_store:
                                    search_doc_id = self.vector_store.store_document(
                                        content=f"Search Result for '{query}'\n\nTitle: {result.get('title')}\nURL: {url}\nSnippet: {result.get('snippet')}\n\nFull content stored separately with doc_id: {content_result.get('doc_id')}",
                                        metadata={
                                            "type": "search_result",
                                            "query": query,
                                            "url": url,
                                            "title": result.get("title"),
                                            "snippet": result.get("snippet"),
                                            "search_rank": i + 1,
                                            "linked_content_doc_id": content_result.get("doc_id"),
                                            "timestamp": datetime.now().isoformat()
                                        }
                                    )
                            else:
                                fetched_content.append({
                                    "url": url,
                                    "title": result.get("title"),
                                    "error": content_result.get("error"),
                                    "stored_in_vector": False
                                })
                                
                        except Exception as e:
                            fetched_content.append({
                                "url": url,
                                "title": result.get("title"),
                                "error": str(e),
                                "stored_in_vector": False
                            })
                            
                logger.info(
                    "Fetched content from %s URLs, %s stored in vector store",
                    len(fetched_content),
                    content_stored_count,
                )

            result = self.create_success_output(
                tool_name="internet_search",
                query=query,
                results=search_results,
                data_retrieved=len(search_results) > 0,
                total_results=len(search_results),
                additional_info=additional_info,
                raw_response=results,  # Keep raw response for debugging
                # Enhanced fields
                fetched_content=fetched_content,
                urls_processed=len(fetched_content),
                content_stored_in_vector=content_stored_count,
                stored_in_vector=content_stored_count > 0
            )
            log_internet_search(
                self.run_config,
                tool="internet_search",
                query=query,
                params={"query": query},
                result=result,
            )
            return result

        except requests.exceptions.RequestException as e:
            logger.error("Internet search request failed: %s", e)
            result = self.create_error_output(
                "internet_search",
                query,
                f"Network error: {str(e)}",
            )
            log_internet_search(
                self.run_config,
                tool="internet_search",
                query=query,
                params={"query": query},
                result=result,
            )
            return result
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", e)
            result = self.create_error_output(
                "internet_search",
                query,
                f"Invalid JSON response: {str(e)}",
            )
            log_internet_search(
                self.run_config,
                tool="internet_search",
                query=query,
                params={"query": query},
                result=result,
            )
            return result
        except Exception as e:
            logger.error("Unexpected error in internet search: %s", e)
            result = self.create_error_output(
                "internet_search",
                query,
                f"Search failed: {str(e)}",
            )
            log_internet_search(
                self.run_config,
                tool="internet_search",
                query=query,
                params={"query": query},
                result=result,
            )
            return result
