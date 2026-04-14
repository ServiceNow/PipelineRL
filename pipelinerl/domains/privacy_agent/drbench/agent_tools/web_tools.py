"""Web fetching tools for the privacy_agent vendored DRBench slice."""

import logging
import re
from typing import Any, Dict, List

from .base import ResearchContext, Tool

logger = logging.getLogger(__name__)


class EnhancedURLFetchTool(Tool):
    """Enhanced URL fetching tool with automatic content processing and vector storage"""

    @property
    def purpose(self) -> str:
        return """Direct content extraction from specific URLs, documents, and web resources.
        IDEAL FOR: Deep analysis of specific reports, whitepapers, financial statements, regulatory documents, case studies, or competitor websites identified during research.
        USE WHEN: You have specific URLs to analyze, need full content from known sources, or want to extract detailed information from targeted web resources.
        PARAMETERS: urls (comma-separated list of specific URLs - e.g., 'https://company.com/annual-report.pdf,https://competitor.com/pricing')
        OUTPUTS: Full content extraction with intelligent parsing, processed text, and metadata that gets automatically stored for synthesis and analysis."""

    def __init__(self, content_processor, model: str, llm_adapter: Any):
        self.content_processor = content_processor
        self.model = model
        self.llm_adapter = llm_adapter


    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute URL fetching with comprehensive processing and standardized output"""

        try:
            # Extract URLs from query
            urls = self._extract_urls_from_query(query, context)

            # If no URLs found, try to generate relevant ones
            if not urls:
                urls = self._generate_relevant_urls(query, context)

            if not urls:
                return self.create_error_output(
                    "enhanced_url_fetch",
                    query,
                    "No URLs found to fetch and unable to generate relevant URLs",
                )

            results = []
            processed_files = []

            for url in urls[:5]:  # Limit to 5 URLs
                # Process the URL using content processor
                process_result = self.content_processor.process_url(
                    url,
                    query_context=f"Query: {query} | Original question: {context.original_question}",
                )

                results.append(process_result)

                # Track successfully processed files
                if process_result.get("success") and process_result.get("file_path"):
                    processed_files.append(process_result["file_path"])
                    context.files_created.append(process_result["file_path"])

                    # Also track extracted text files
                    if process_result.get("extracted_path"):
                        context.files_created.append(process_result["extracted_path"])

            # Generate summary of findings
            summary = self._generate_findings_summary(results, query)

            # Count successful operations
            successful_urls = [r for r in results if r.get("success")]
            urls_with_data = [r for r in results if r.get("stored_in_vector")]

            return self.create_success_output(
                tool_name="enhanced_url_fetch",
                query=query,
                results=results,
                data_retrieved=len(urls_with_data) > 0,
                urls_processed=len(results),
                successful_downloads=len(successful_urls),
                processed_files=processed_files,
                findings_summary=summary,
                content_stored_in_vector=len(urls_with_data),
            )

        except Exception as e:
            return self.create_error_output("enhanced_url_fetch", query, f"URL fetching failed: {str(e)}")

    def _extract_urls_from_query(self, query: str, context: ResearchContext) -> List[str]:
        """Extract URLs from query or context"""
        urls = []

        # Direct URL extraction from query
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls.extend(re.findall(url_pattern, query))

        # Check context for URLs in previous findings
        for finding in context.findings.values():
            if isinstance(finding, dict) and "url" in finding:
                urls.append(finding["url"])

        return list(set(urls))

    def _generate_relevant_urls(self, query: str, context: ResearchContext) -> List[str]:
        """Generate relevant URLs using LLM"""

        prompt = f"""
Given this research query, suggest 3-5 specific URLs that are likely to contain relevant information:

Query: "{query}"
Original research question: "{context.original_question}"

Consider these types of sources:
1. Academic papers (arxiv.org, scholar.google.com, research institutions)
2. Official documentation (organization websites, government sites)
3. News articles (major news outlets)
4. Industry reports (company websites, industry associations)
5. Technical resources (GitHub, documentation sites)

Return only valid URLs, one per line, no explanations, no markdown formatting, no bullet points, no quotes, 
no enumeration, no additional text.
Focus on authoritative sources that would have recent, accurate information.
"""

        try:
            response = self.llm_adapter.generate_text(
                prompt,
                model=self.model,
                log_name="web_url_extraction",
            )
            lines = response.strip().split("\n")
            urls = []

            for line in lines:
                line = line.strip()
                if line.startswith("http://") or line.startswith("https://"):
                    urls.append(line)
                elif "." in line and not line.startswith("#"):
                    # Try to fix URLs missing protocol
                    if not line.startswith("www."):
                        urls.append("https://" + line)
                    else:
                        urls.append("https://" + line)

            return urls[:5]

        except Exception as e:
            logger.error(f"Error generating URLs: {e}")
            return []

    def _generate_findings_summary(self, results: List[Dict], query: str) -> str:
        """Generate a summary of findings from processed URLs"""

        successful_results = [r for r in results if r.get("success")]

        if not successful_results:
            return "No content was successfully retrieved from URLs."

        # Combine extracted content for summary using lazy loading
        all_content = []
        for result in successful_results:
            content = self.load_extracted_content(result)
            if content and len(content) > 100:  # Only meaningful content
                all_content.append(content[:500])  # Limit per source

        if not all_content:
            return "Content was downloaded but text extraction was unsuccessful."

        # Generate summary using LLM
        summary_prompt = f"""
Based on the query "{query}", summarize the key findings from the following content:

{chr(10).join(all_content)}

Provide a concise summary highlighting:
1. Key facts relevant to the query
2. Important insights discovered
3. Any conflicting information found
4. Gaps that might need further research

Keep the summary focused and factual.
"""

        try:
            return self.llm_adapter.generate_text(
                summary_prompt,
                model=self.model,
                log_name="web_findings_summary",
            )
        except Exception as e:
            return f"Summary generation failed: {e}"
