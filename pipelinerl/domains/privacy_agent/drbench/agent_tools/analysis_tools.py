import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..vector_store import VectorStore

from .base import ResearchContext, Tool
from .model_config import get_analysis_config

logger = logging.getLogger(__name__)


class SmartAnalysisTool(Tool):
    """Enhanced analysis tool that uses vector store for context-aware synthesis without token limits"""

    @property
    def purpose(self) -> str:
        return """Advanced AI-powered synthesis and analysis of collected research data using vector search and intelligent reasoning.
        IDEAL FOR: Cross-referencing findings, identifying patterns, generating insights, analyzing complex relationships, creating strategic recommendations, and answering complex questions requiring synthesis of multiple sources.
        USE WHEN: You have collected substantial research data and need to analyze relationships, extract insights, compare findings, or generate evidence-based conclusions and recommendations.
        PARAMETERS: query (analytical questions - e.g., 'Compare competitive positioning strategies', 'What are the key market risks?', 'Synthesize customer feedback patterns')
        OUTPUTS: Intelligent analysis with evidence-based insights, cross-referenced findings, strategic recommendations, and detailed citations to source materials for verification."""

    def __init__(
        self,
        model: str,
        llm_adapter: Any,
        vector_store: Optional[VectorStore] = None,
        capacity_tier: Optional[str] = None,
        max_relevant_docs: int = None,
        max_chars: int = None,
        workspace_dir: Optional[str] = None,
    ):
        self.model = model
        self.llm_adapter = llm_adapter
        self.vector_store = vector_store
        self.workspace_dir = workspace_dir

        # Get configuration (model-agnostic with optional optimizations)
        config = get_analysis_config(
            capacity_tier=capacity_tier, max_relevant_docs=max_relevant_docs, max_chars=max_chars
        )

        self.max_relevant_docs = config["max_relevant_docs"]
        self.max_chars = config["max_chars"]
        self.max_per_source = config["max_per_source"]


    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute smart analysis using vector store to avoid token limits"""

        try:
            # Get context summary instead of full findings
            context_summary = context.get_context_summary()

            # Search vector store for relevant documents
            relevant_docs = []
            if self.vector_store:
                # Search based on the synthesis query
                search_results = self.vector_store.search(query, top_k=8)
                relevant_docs.extend(search_results)

                # Also search based on the original question
                question_results = self.vector_store.search(context.original_question, top_k=4)
                relevant_docs.extend(question_results)

                # Remove duplicates by doc_id AND filter out AI synthesis/findings sources
                seen_ids = set()
                unique_docs = []
                for doc in relevant_docs:
                    doc_id = doc["doc_id"]
                    metadata = doc.get("metadata", {})
                    doc_type = metadata.get("type", "")
                    tool_used = metadata.get("tool_used", "")

                    # Skip AI synthesis documents and smart analysis findings
                    if (
                        doc_type in ["ai_synthesis_with_sources", "ai_synthesis", "research_finding"]
                        or tool_used == "smart_analysis"
                        or "synthesis" in doc_type
                    ):
                        continue

                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs.append(doc)

                relevant_docs = sorted(unique_docs, key=lambda x: x.get("similarity_score", 0), reverse=True)
                relevant_docs = relevant_docs[: self.max_relevant_docs]  # Limit to top N docs

            # Create a chunked synthesis approach
            synthesis_results = self._chunked_synthesis(
                query=query,
                context_summary=context_summary,
                relevant_docs=relevant_docs,
                original_question=context.original_question,
            )

            # Store the synthesis result in vector store with source tracking
            stored_doc_id = None
            if self.vector_store and synthesis_results.get("synthesis"):
                # Extract source document IDs for citation tracking
                source_doc_ids = [doc["doc_id"] for doc in relevant_docs if doc.get("doc_id")]

                # Create metadata that tracks the source documents
                synthesis_metadata = {
                    "tool_used": "smart_analysis",
                    "type": "ai_synthesis_with_sources" if source_doc_ids else "ai_synthesis",
                    "source": "analysis",
                    "query_context": query,
                    "original_question": context.original_question,
                    "docs_analyzed": len(relevant_docs),
                    "synthesis_method": "chunked_vector_search",
                    "source_document_ids": source_doc_ids,  # Track source docs for citation
                    "timestamp": datetime.now().isoformat(),
                }

                # Store the synthesis in vector store
                stored_doc_id = self.vector_store.store_document(
                    content=synthesis_results["synthesis"], metadata=synthesis_metadata
                )

                # Save AI synthesis to readable file
                self._save_synthesis_to_file(
                    synthesis_results["synthesis"], query, context.original_question, source_doc_ids, stored_doc_id
                )

            return self.create_success_output(
                tool_name="smart_analysis",
                query=query,
                results=synthesis_results,
                data_retrieved=True,
                context_size=len(context_summary),
                docs_analyzed=len(relevant_docs),
                vector_store_used=self.vector_store is not None,
                stored_doc_id=stored_doc_id,
                stored_in_vector=True,  # Prevent duplicate storage as research_finding
            )

        except Exception as e:
            logger.error(f"Smart analysis failed: {str(e)}")
            return self.create_error_output("smart_analysis", query, f"Smart analysis failed: {str(e)}")

    def _chunked_synthesis(
        self, query: str, context_summary: Dict, relevant_docs: List[Dict], original_question: str
    ) -> Dict[str, Any]:
        """Perform synthesis in chunks to avoid token limits"""

        # First, synthesize the relevant documents
        doc_synthesis = self._synthesize_documents(relevant_docs, query)

        # Then create final synthesis combining context and doc synthesis
        final_prompt = f"""
Based on the research question: "{original_question}"
And the specific query: "{query}"

Context Summary:
{json.dumps(context_summary, indent=2)}

Document Analysis (contains citations in [DOC:doc_id] format):
{doc_synthesis}

CRITICAL CITATION REQUIREMENTS:
- The Document Analysis above contains citations in [DOC:doc_id] format
- You MUST preserve and use these [DOC:doc_id] citations in your final synthesis
- EXACT FORMAT: [DOC:doc_id] - with colon after DOC, not underscore or any other character
- Use INDIVIDUAL citations for each document: [DOC:doc_1][DOC:doc_2] NOT [DOC:doc_1; DOC:doc_2]
- When making claims based on the document analysis, include the appropriate [DOC:doc_id] citations
- Example: "Internal analysis shows transparency gaps [DOC:doc_12345]."
- Example: "Multiple sources confirm trend [DOC:doc_67890][DOC:doc_11111]."
- WRONG FORMAT: [DOC_doc_id] or [DOC doc_id] or [DOC:doc_1; DOC:doc_2] - these are INCORRECT
- DO NOT create new citations - only use the existing [DOC:doc_id] references from the Document Analysis

Provide a comprehensive synthesis with proper citations:
1. **Quantitative Insights** - All numerical data with calculations and aggregations [DOC:citations]
2. **Business Outcomes & Achievements** - Specific performance improvements and results [DOC:citations]  
3. **Supporting Qualitative Analysis** - Context and implications [DOC:citations]
4. **Identified Gaps** - Missing data or analysis needed [DOC:citations]
5. **Confidence Level** - High (explicit data), Medium (derived), Low (limited data)

MATHEMATICAL OPERATIONS REQUIRED:
- When you find percentages for different segments (e.g., 35% finance, 40% healthcare), calculate combined totals
- When you find absolute costs with baseline references, calculate percentage increases
- When you find multiple data points, aggregate appropriately

FACT VERIFICATION CRITICAL:
- ONLY claim what documents explicitly state
- Use exact quotes for key claims: "According to [DOC:doc_id], the document states: '[exact quote]'"
- If calculating aggregations, show your work: "Finance (35%) + Healthcare (40%) = 75% combined [DOC:doc_1][DOC:doc_2]"
- Never claim "internal assessments show X" unless the document provides the actual assessment results

BUSINESS FOCUS:
- Prioritize achievement data: improvements in metrics, cost savings, efficiency gains
- Look for competitive advantages, market positioning improvements
- Identify specific business outcomes and quantifiable benefits

Be concise but thorough. Focus on quantitative findings and business achievements while preserving all [DOC:doc_id] citations.
"""

        final_synthesis = self.llm_adapter.generate_text(
            final_prompt,
            model=self.model,
            log_name="analysis_final_synthesis",
        )

        # Fix malformed citations in final synthesis too
        final_synthesis = self._fix_malformed_citations(final_synthesis)

        return {
            "synthesis": final_synthesis,
            "context_summary": context_summary,
            "documents_analyzed": len(relevant_docs),
            "synthesis_method": "chunked_vector_search",
        }

    def _synthesize_documents(self, documents: List[Dict], query: str) -> str:
        """Synthesize information from vector store documents with proper citation references"""

        if not documents:
            return "No relevant documents found in vector store."

        # Prepare document content with doc_id references for citation
        doc_content_with_ids = []
        for doc in documents:
            doc_id = doc.get("doc_id", "unknown")
            content = doc.get("content", "")[: self.max_chars]  # Limit content length
            metadata = doc.get("metadata", {})
            source_type = metadata.get("source", "unknown")
            doc_title = metadata.get("title", metadata.get("filename", "Untitled"))

            doc_content_with_ids.append(
                {"doc_id": doc_id, "content": content, "source_type": source_type, "title": doc_title}
            )

        # Create synthesis prompt that will generate proper citations with quantitative focus
        synthesis_prompt = f"""
You are analyzing documents to answer: "{query}"

Documents available for analysis:
{json.dumps(doc_content_with_ids, indent=2)}

CRITICAL QUANTITATIVE ANALYSIS REQUIREMENTS:
- IDENTIFY and EXTRACT all numerical data: percentages, costs, counts, ratios, timeframes
- PERFORM MATHEMATICAL OPERATIONS when appropriate:
  * Aggregate percentages across categories (e.g., 35% finance + 40% healthcare = 75% combined)
  * Calculate percentage increases from baseline costs
  * Sum totals across different sources
  * Compute averages or weighted averages when meaningful
- PRESERVE exact numerical values - never round or approximate
- LOOK FOR: baseline costs, percentage increases, customer segments, achievement metrics, performance data

ACHIEVEMENT AND OUTCOME FOCUS:
- Specifically search for: win/loss ratios, renewal rates, churn improvements, sales performance
- Look for: cost savings, efficiency gains, customer satisfaction improvements
- Identify: before/after metrics, ROI data, performance benchmarks
- Extract: specific business outcomes and quantifiable achievements

CITATION REQUIREMENTS:
- EXACT FORMAT: [DOC:doc_id] - with colon after DOC
- Use INDIVIDUAL citations: [DOC:doc_1][DOC:doc_2] NOT [DOC:doc_1; DOC:doc_2]
- Cite EVERY numerical claim and calculation with source documents
- NEVER make claims without document support - if data isn't explicitly in documents, don't claim it exists

Required synthesis structure:
1. **Quantitative Findings** - All numerical data with calculations and citations
2. **Business Outcomes & Achievements** - Performance metrics, improvements, ROI with citations
3. **Supporting Evidence** - Qualitative insights that support quantitative findings with citations
4. **Data Gaps** - What numerical data is missing or unclear

FACT VERIFICATION:
- Only state what is EXPLICITLY written in the documents
- If a document mentions "internal assessments" but doesn't provide the actual findings, say "document references internal assessments but doesn't provide specific results" or avoid making claims about those assessments.
- Don't infer or extrapolate beyond what's directly stated
- Use phrases like "according to [document]" or "as stated in [document]" for clarity

Every numerical claim MUST have [DOC:doc_id] citations:
"""

        # Generate synthesis with citations
        synthesis_result = self.llm_adapter.generate_text(
            synthesis_prompt,
            model=self.model,
            log_name="analysis_synthesis",
        )

        # Fix malformed citations (common LLM mistakes)
        synthesis_result = self._fix_malformed_citations(synthesis_result)

        # Debug: Log if no DOC references were generated
        doc_ref_count = synthesis_result.count("[DOC:")
        logger.debug(f"SmartAnalysisTool generated {doc_ref_count} DOC references in synthesis")

        return synthesis_result

    def _save_synthesis_to_file(
        self, synthesis: str, query: str, original_question: str, source_doc_ids: List[str], vector_doc_id: str
    ) -> None:
        """Save AI synthesis to a readable markdown file in workspace"""
        if not self.workspace_dir:
            return

        try:
            from pathlib import Path

            # Create AI synthesis directory
            workspace_path = Path(self.workspace_dir)
            synthesis_dir = workspace_path / "ai_synthesis"
            synthesis_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Clean query for filename
            clean_query = query[:50].replace("/", "_").replace(" ", "_").replace("?", "").replace(":", "")
            filename = f"synthesis_{timestamp}_{clean_query}.md"
            filepath = synthesis_dir / filename

            # Count citations in synthesis
            citation_count = synthesis.count("[DOC:")

            # Create readable markdown content
            markdown_content = f"""# AI Synthesis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Query**: {query}
**Original Question**: {original_question}
**Vector DB Doc ID**: {vector_doc_id}
**Citations in Synthesis**: {citation_count}

## Synthesis

{synthesis}

## Metadata

- **Synthesis Method**: chunked_vector_search
- **Total Source Documents**: {len(source_doc_ids)}

## Source Document IDs

{chr(10).join(f"- {doc_id}" for doc_id in source_doc_ids[:20])}{'...' if len(source_doc_ids) > 20 else ''}

---

*This AI synthesis report was automatically generated by the SmartAnalysisTool as part of the research process.*
"""

            # Write to file
            filepath.write_text(markdown_content, encoding="utf-8")
            logger.info(f"Saved AI synthesis to readable file: {filepath}")

        except Exception as e:
            logger.warning(f"Failed to save AI synthesis to file: {e}")

    def _fix_malformed_citations(self, text: str) -> str:
        """Fix common malformed citation patterns in AI synthesis"""
        import re

        # Fix [DOC_doc_id] -> [DOC:doc_id] (underscore instead of colon)
        text = re.sub(r"\[DOC_([^\]]+)\]", r"[DOC:\1]", text)

        # Fix [DOC doc_id] -> [DOC:doc_id] (space instead of colon)
        text = re.sub(r"\[DOC\s+([^\]]+)\]", r"[DOC:\1]", text)

        # Fix [DOC-doc_id] -> [DOC:doc_id] (dash instead of colon)
        text = re.sub(r"\[DOC-([^\]]+)\]", r"[DOC:\1]", text)

        # Fix duplicate DOC: patterns like [DOC:DOC:doc_id] -> [DOC:doc_id]
        text = re.sub(r"\[DOC:DOC:([^\]]+)\]", r"[DOC:\1]", text)

        # Count fixed citations
        fixed_count = 0
        for pattern in [r"\[DOC_([^\]]+)\]", r"\[DOC\s+([^\]]+)\]", r"\[DOC-([^\]]+)\]"]:
            original_text = text
            if re.search(pattern, original_text):
                fixed_count += len(re.findall(pattern, original_text))

        if fixed_count > 0:
            logger.info(f"Fixed {fixed_count} malformed citations in AI synthesis")

        return text
