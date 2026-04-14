import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..vector_store import VectorStore

from .base import ResearchContext
from .citation_registry import UnifiedCitationRegistry
from .model_config import get_report_config

logger = logging.getLogger(__name__)


class ReportAssembler:
    """Enhanced report assembler with intelligent content synthesis and comprehensive metadata tracking"""

    def __init__(
        self,
        model: str,
        llm_adapter: Any,
        vector_store: Optional[VectorStore] = None,
        capacity_tier: Optional[str] = None,
        max_content_length: int = None,
        max_total_tokens: int = None,
        report_style: str = "research_report",
    ):
        self.model = model
        self.llm_adapter = llm_adapter
        self.vector_store = vector_store
        self.report_style = report_style
        self._parsed_answers = {}
        self._parsed_justifications = {}

        # Get configuration (model-agnostic with optional optimizations)
        config = get_report_config(
            capacity_tier=capacity_tier, max_content_length=max_content_length, max_total_tokens=max_total_tokens
        )

        self.max_content_length = config["max_content_length"]
        self.max_total_tokens = config["max_total_tokens"]

        self.evidence_metadata = {
            "document_ids": [],
            "evidence_map": {"web_sources": [], "internal_sources": [], "research_findings": []},
            "content_synthesis": {},
            "research_methodology": {},
        }
        # Unified citation registry for all citation management
        self.citation_registry = UnifiedCitationRegistry()

    def generate_comprehensive_report(self, context: ResearchContext, action_plan=None) -> str:
        """Generate comprehensive report with intelligent content synthesis and rich metadata tracking"""

        try:
            # Reset evidence metadata for this report
            self._reset_evidence_metadata()

            if self.report_style == "concise_qa":
                return self._generate_concise_qa_report(context, action_plan)

            # Stage 1: Intelligent Content Analysis
            thematic_content = self._analyze_and_cluster_content(context)

            # Stage 2: Multi-Theme Synthesis (this populates source registry)
            synthesized_sections = self._synthesize_thematic_content(thematic_content, context)

            # Stage 3: Build Evidence Metadata (after source registry is populated)
            evidence_summary = self._build_evidence_metadata(context)

            # Stage 4: Generate Clean Report
            report = self._assemble_final_report(synthesized_sections, evidence_summary, context)

            # Stage 5: Finalize Metadata
            self._finalize_metadata(context, action_plan, thematic_content)

            return report

        except Exception as e:
            return f"Error generating comprehensive report: {e}"

    def _reset_evidence_metadata(self):
        """Reset evidence metadata for new report generation"""
        self.evidence_metadata = {
            "document_ids": [],
            "evidence_map": {"web_sources": [], "internal_sources": [], "research_findings": []},
            "content_synthesis": {},
            "research_methodology": {},
        }
        # Reset unified citation registry
        self.citation_registry = UnifiedCitationRegistry()

    def _generate_text(self, prompt: str, log_name: str) -> str:
        return self.llm_adapter.generate_text(prompt, model=self.model, log_name=log_name)

    def _generate_concise_qa_report(self, context: ResearchContext, action_plan=None) -> str:
        """Generate a concise Q&A report with ANSWER_N/JUSTIFICATION_N format for easy parsing."""
        clean_question = self._extract_clean_question(context.original_question)

        # Parse sub-questions from the original question
        # Handles: "1. text", "2) text", "4 (final). text"
        sub_questions = []
        for line in clean_question.split("\n"):
            line = line.strip()
            m = re.match(r'^(\d+)\s*(?:\([^)]*\))?\s*[\.\)]\s*(.+)', line)
            if m:
                sub_questions.append({"num": m.group(1), "text": m.group(2).strip()})

        # If no numbered sub-questions found, treat the whole question as one
        if not sub_questions:
            sub_questions = [{"num": "1", "text": clean_question}]

        # Gather primary source evidence from vector store and preserve original metadata
        unique_content = self._collect_concise_qa_evidence(sub_questions)

        # Register documents in citation registry with full source metadata
        for content_item in unique_content[:30]:
            if content_item["doc_id"]:
                self._get_source_citation_id(
                    {
                        "doc_id": content_item["doc_id"],
                        "metadata": content_item["metadata"],
                    }
                )

        evidence_items = []
        for content_item in unique_content[:30]:
            source_info = self._extract_source_info(
                {
                    "doc_id": content_item["doc_id"],
                    "metadata": content_item["metadata"],
                }
            )
            evidence_entry = {
                "doc_id": content_item["doc_id"],
                "source_type": source_info.get("type", "unknown"),
                "source_title": source_info.get("title", "Unknown Source"),
                "content": content_item["text"],
                "score": round(content_item["score"], 3),
            }
            if source_info.get("source_document_ids"):
                evidence_entry["underlying_doc_ids"] = source_info["source_document_ids"]
            evidence_items.append(evidence_entry)

        # Truncate total content to fit context
        content_text = json.dumps(evidence_items, indent=2)
        if len(content_text) > self.max_content_length:
            content_text = content_text[:self.max_content_length]

        # Build the Q&A prompt
        q_list = "\n".join(f"Q{sq['num']}: {sq['text']}" for sq in sub_questions)

        qa_prompt = f"""Answer each numbered question with a SHORT, EXACT answer (1-5 words) and a brief justification.

Original Research Question:
{clean_question}

Sub-questions to answer:
{q_list}

Available Evidence:
{content_text}

For each question, output EXACTLY this format:
ANSWER_1: <short answer>
JUSTIFICATION_1: <1-2 sentences citing evidence> [DOC:doc_id]

ANSWER_2: <short answer>
JUSTIFICATION_2: <1-2 sentences citing evidence> [DOC:doc_id]

...

ANSWER_FINAL: <final answer>
JUSTIFICATION_FINAL: <1-2 sentences explaining how sub-answers combine>

Rules:
- Answers should be as concise as possible (a number, a name, a percentage)
- Justifications should cite specific evidence using [DOC:doc_id]
- Each JUSTIFICATION line must include at least one [DOC:doc_id] when supporting evidence exists
- If a question references a previous answer (e.g., "over the next (1) months"), resolve it
- If you cannot determine the answer, write: NOT_FOUND
- Use ONLY the provided evidence
- Output ONLY ANSWER_*/JUSTIFICATION_* lines
- DO NOT include a thinking process, markdown headers, bullets, JSON, or any extra commentary
- If you add extra text outside the required lines, the output is unusable
"""

        try:
            report_text = self._generate_text(qa_prompt, log_name="report_qa")
        except Exception as e:
            logger.error(f"Concise QA report generation failed: {e}")
            self._parsed_answers = {}
            self._parsed_justifications = {}
            return f"# Research Findings\n\nError generating report: {e}"

        # Parse ANSWER_N and JUSTIFICATION_N lines
        answers = {}
        justifications = {}
        for line in report_text.split("\n"):
            line = line.strip()
            m = re.match(r'^ANSWER_(\w+):\s*(.+)', line)
            if m:
                answers[m.group(1)] = m.group(2).strip()
            m = re.match(r'^JUSTIFICATION_(\w+):\s*(.+)', line)
            if m:
                justifications[m.group(1)] = m.group(2).strip()

        # Store parsed results as side-channel for runner to access
        self._parsed_answers = answers
        self._parsed_justifications = justifications

        # Build readable report text
        header = f"# Research Findings: {clean_question}\n\n"
        main_report = header + report_text

        # Resolve citations
        final_report, citation_assignments = self.citation_registry.finalize_citations(main_report)
        if citation_assignments:
            references_section = self.citation_registry.generate_references_section()
            final_report += "\n" + references_section

        # Finalize metadata
        if action_plan:
            self._finalize_metadata(context, action_plan, {})

        return final_report

    def _collect_concise_qa_evidence(self, sub_questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Collect concise-QA evidence while preferring primary source documents."""
        if not self.vector_store:
            return []

        primary_content: list[Dict[str, Any]] = []
        fallback_content: list[Dict[str, Any]] = []

        for sub_question in sub_questions:
            results = self.vector_store.search(sub_question["text"], top_k=10, use_semantic=True)
            for result in results:
                score = result.get("similarity_score", 0)
                if score <= 0.3:
                    continue

                metadata = result.get("metadata", {}) or {}
                evidence_item = {
                    "doc_id": result.get("doc_id", ""),
                    "text": result.get("content", "")[:2000],
                    "metadata": metadata,
                    "score": score,
                }

                if self._is_primary_concise_qa_source(metadata):
                    primary_content.append(evidence_item)
                else:
                    fallback_content.append(evidence_item)

        chosen_content = primary_content if primary_content else fallback_content
        deduped_content: list[Dict[str, Any]] = []
        seen_doc_ids: set[str] = set()

        for content_item in sorted(chosen_content, key=lambda item: -item["score"]):
            doc_id = content_item["doc_id"]
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            deduped_content.append(content_item)

        return deduped_content

    def _is_primary_concise_qa_source(self, metadata: Dict[str, Any]) -> bool:
        """Prefer directly retrieved source documents over synthetic findings for concise QA."""
        doc_type = metadata.get("type", "")
        tool_used = metadata.get("tool_used", "")

        if doc_type in {"research_finding", "research_report", "archived_findings"}:
            return False
        if doc_type in {"ai_synthesis", "ai_synthesis_with_sources"}:
            return False
        if doc_type == "search_result":
            return False
        if tool_used == "smart_analysis":
            return False
        return True

    def _analyze_and_cluster_content(self, context: ResearchContext) -> Dict[str, List[Dict]]:
        """Analyze vector store content and cluster by themes"""

        if not self.vector_store:
            return {"general": []}

        try:
            # Multi-query approach for comprehensive content discovery
            search_queries = self._generate_search_queries(context.original_question, context.plan)
            all_content = {}

            # Add specific internal-focused queries to ensure internal content is found
            internal_queries = self._generate_internal_search_queries(context.original_question, context.plan)
            all_search_queries = search_queries + internal_queries

            for query in all_search_queries:
                results = self.vector_store.search(query, top_k=15, use_semantic=True)
                theme = self._extract_theme_from_query(query)

                if theme not in all_content:
                    all_content[theme] = []

                for result in results:
                    # Use more lenient threshold for internal/enterprise sources
                    metadata = result.get("metadata", {})
                    is_internal = self._is_internal_source(metadata)
                    min_threshold = 0.5 if is_internal else 0.7  # Lower threshold for internal sources

                    if result.get("similarity_score", 0) > min_threshold:
                        # Skip AI synthesis without source tracking
                        if (
                            metadata.get("tool_used") == "smart_analysis"
                            and metadata.get("type") == "ai_synthesis"
                            and not metadata.get("source_document_ids")
                        ):
                            logger.debug(
                                f"Skipping AI synthesis without sources in content analysis: {result.get('doc_id')}"
                            )
                            continue

                        all_content[theme].append(
                            {
                                "content": result.get("content", ""),
                                "metadata": metadata,
                                "score": result.get("similarity_score", 0),
                                "doc_id": result.get("doc_id", ""),
                                "theme": theme,
                                "is_internal": is_internal,
                            }
                        )

            # Deduplicate and rank content within themes
            for theme in all_content:
                original_count = len(all_content[theme])
                all_content[theme] = self._deduplicate_and_rank(all_content[theme])
                final_count = len(all_content[theme])

                # Count internal vs external sources
                internal_count = sum(1 for item in all_content[theme] if item.get("is_internal", False))
                external_count = final_count - internal_count

                logger.debug(
                    f"Theme '{theme}': {original_count}→{final_count} items ({internal_count} internal, {external_count} external)"
                )

            total_items = sum(len(items) for items in all_content.values())
            total_internal = sum(
                sum(1 for item in items if item.get("is_internal", False)) for items in all_content.values()
            )
            logger.info(
                f"🔍 Total content analysis: {total_items} items ({total_internal} internal, {total_items - total_internal} external)"
            )

            return all_content

        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            return {"general": []}

    def _generate_search_queries(self, original_question: str, research_plan: Any) -> List[str]:
        """Generate targeted search queries for comprehensive content discovery"""

        # Safely handle missing or empty research_plan/research_investigation_areas
        actionable_tasks = []
        if (
            research_plan
            and isinstance(research_plan, dict)
            and "research_investigation_areas" in research_plan
            and isinstance(research_plan["research_investigation_areas"], list)
        ):
            actionable_tasks = research_plan["research_investigation_areas"]

        research_plan_length = len(actionable_tasks) if actionable_tasks else 5  # Default to 5 if no tasks
        research_questions = (
            "\n - ".join(
                [f'{task.get("research_focus", "")}: {task.get("business_rationale", "")}' for task in actionable_tasks]
            )
            if actionable_tasks
            else original_question
        )

        query_prompt = f"""
Generate comprehensive search queries to find ALL relevant documents for: "{original_question}"

Research Areas:
- {research_questions}

REQUIRED QUERY CATEGORIES (generate 2-3 queries per category):

1. **QUANTITATIVE DATA QUERIES:**
   - Cost analysis, budget impact, percentage changes
   - Customer surveys, satisfaction metrics, performance data
   - Financial analysis, ROI calculations, expense tracking

2. **ACHIEVEMENT & OUTCOME QUERIES:**
   - Win/loss ratio, sales performance, competitive analysis
   - Customer retention, churn analysis, renewal rates
   - Success stories, case studies, performance improvements

3. **COMPLIANCE & REGULATORY QUERIES:**
   - Regulatory requirements, compliance assessments, audit results
   - Legal analysis, policy documents, regulatory frameworks
   - Implementation guidelines, compliance gaps

4. **TECHNICAL IMPLEMENTATION QUERIES:**
   - Architecture assessments, technical requirements
   - Resource planning, staffing needs, implementation costs
   - System capabilities, technical gaps, upgrade requirements

5. **MARKET & CUSTOMER QUERIES:**
   - Customer concerns, market research, industry analysis
   - Competitor analysis, market positioning, customer feedback
   - Industry trends, market opportunities, customer segments

Use specific document type keywords: "analysis", "assessment", "report", "study", "survey", "review", "evaluation", "comparison", "performance", "metrics", "data", "statistics", "costs", "budget", "financial", "achievement", "outcome", "success", "improvement"

Return only the queries, one per line:
"""

        try:
            response = self._generate_text(query_prompt, log_name="report_query_generation")
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            return queries
        except:
            # Fallback queries
            fallback_tasks = actionable_tasks if actionable_tasks else []
            return [
                original_question,
                f"background {original_question}",
                f"current trends {original_question}",
                f"analysis {original_question}",
                f"implementation {original_question}",
            ] + [task.get("research_focus", original_question) for task in fallback_tasks]

    def _extract_theme_from_query(self, query: str) -> str:
        """Extract theme identifier from search query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["background", "history", "overview"]):
            return "background"
        elif any(word in query_lower for word in ["trends", "current", "recent", "latest"]):
            return "current_trends"
        elif any(word in query_lower for word in ["analysis", "research", "study", "data"]):
            return "analysis"
        elif any(word in query_lower for word in ["implementation", "practical", "how to", "guide"]):
            return "implementation"
        elif any(word in query_lower for word in ["future", "prediction", "forecast"]):
            return "future_outlook"
        else:
            return "general"

    def _deduplicate_and_rank(
        self, content_list: List[Dict], internal_ratio: float = 0.65, max_results: Optional[int] = 10
    ) -> List[Dict]:
        """Remove duplicates and rank content by quality, prioritizing internal sources"""

        # Simple deduplication by content similarity
        unique_content = []
        seen_content = set()

        for item in content_list:
            content_hash = hash(item["content"][:600])  # Hash first 600 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_content.append(item)

        # Rank by priority: internal sources first, then by score and content length
        def ranking_key(x):
            is_internal = x.get("is_internal", False)
            score = x["score"]
            content_length = len(x["content"])
            # Internal sources get priority boost
            priority_score = (2.0 if is_internal else 1.0) * score
            return (priority_score, content_length)

        ranked_content = sorted(unique_content, key=ranking_key, reverse=True)

        # Ensure we get a good mix: prioritize internal but include some external
        internal_items = [x for x in ranked_content if x.get("is_internal", False)]
        external_items = [x for x in ranked_content if not x.get("is_internal", False)]

        # Take top internal items and some external for balance
        max_results = max_results or min(len(ranked_content), 8)  # Limit to 8 items max per theme
        result = (
            internal_items[: int(max_results * internal_ratio)]
            + external_items[: int(max_results * (1 - internal_ratio))]
        )

        return result

    def _build_evidence_metadata(self, context: ResearchContext) -> Dict[str, Any]:
        """Build comprehensive evidence metadata from research context and source registry"""

        # Count sources by type from the actual source registry (more accurate)
        internal_count = 0
        enterprise_count = 0
        web_count = 0
        research_count = 0

        for doc_id, doc_info in self.citation_registry.documents.items():
            if doc_info.document_type == "ai_synthesis":
                continue  # Skip AI synthesis docs for counting
            source_info = doc_info.source_info
            source_type = source_info.get("type", "")

            if source_type == "internal":
                internal_count += 1
            elif source_type in ["enterprise_chat", "enterprise_file", "enterprise_api", "enterprise_email"]:
                enterprise_count += 1
            elif source_type == "external":
                web_count += 1
            elif source_type in ["research", "synthesis"]:
                research_count += 1

        # Legacy processing for backwards compatibility - track additional sources from findings
        for key, finding in context.findings.items():
            if isinstance(finding, dict) and finding.get("success"):
                # Track files processed that might not be in registry yet
                for file_path in finding.get("processed_files", []):
                    if file_path not in self.evidence_metadata["document_ids"]:
                        self.evidence_metadata["document_ids"].append(file_path)

                # Track URLs from fetched content that might not be in registry yet
                for content_item in finding.get("fetched_content", []):
                    url = content_item.get("url")
                    if url and url not in self.evidence_metadata["document_ids"]:
                        self.evidence_metadata["document_ids"].append(url)

        return {
            "total_sources": internal_count + enterprise_count + web_count + research_count,
            "web_sources": web_count,
            "internal_sources": internal_count,
            "enterprise_sources": enterprise_count,
            "research_findings": research_count,
            "themes_identified": [],
        }

    def _synthesize_thematic_content(
        self, thematic_content: Dict[str, List[Dict]], context: ResearchContext
    ) -> Dict[str, str]:
        """Synthesize content for each theme using focused LLM calls with inline citations"""

        # Dynamic content length based on number of items to process (use instance variable as base)
        base_content_length = self.max_content_length

        # Calculate average items per theme
        total_items = sum(len(items) for items in thematic_content.values())
        themes_count = len(thematic_content)
        avg_items_per_theme = total_items / themes_count if themes_count > 0 else 1

        # Scale content length based on complexity
        if avg_items_per_theme > 10:
            max_content_length = base_content_length // 2  # Reduce for high complexity
        elif avg_items_per_theme > 5:
            max_content_length = int(base_content_length * 0.75)  # Moderate reduction
        else:
            max_content_length = base_content_length  # Full length for simple cases

        logger.debug(f"Using max_content_length={max_content_length} for {avg_items_per_theme:.1f} avg items per theme")

        # First, build a global source registry from ALL content items across all themes
        # This ensures consistent citation numbering across the entire report
        all_items = []
        for theme_items in thematic_content.values():
            all_items.extend(theme_items)

        # Remove duplicates by doc_id
        seen_doc_ids = set()
        unique_items = []
        for item in all_items:
            doc_id = item.get("doc_id")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_items.append(item)

        # PRE-PROCESS: Build global citation registry by processing all items first
        # This ensures citation IDs are unique across all themes
        for item in unique_items:
            self._get_source_citation_id(item)

        # Build global citation registry
        global_sources = self._build_theme_source_registry(unique_items)

        synthesized_sections = {}

        for theme, content_items in thematic_content.items():
            if not content_items:
                continue

            # Sort content_items: prioritize internal sources over web sources
            prioritized_items = self._prioritize_sources(content_items)

            # Prepare content for synthesis with citation mapping
            content_for_synthesis = []
            seen_citation_ids = set()  # Track citation IDs to avoid duplicates

            for item in prioritized_items:
                source_id = self._get_source_citation_id(item)
                # Determine source priority based on enhanced source classification
                metadata = item["metadata"]
                source = metadata.get("source", "")
                source_type = metadata.get("source_type", metadata.get("type", ""))

                # Check if this is SmartAnalysisTool output with source tracking
                tool_used = metadata.get("tool_used", "")
                source_document_ids = metadata.get("source_document_ids", [])

                if tool_used == "smart_analysis" and source_document_ids:
                    # Inherit priority from highest priority source document. TOO SLOW
                    # priority = self._get_highest_priority_from_source_docs(source_document_ids)
                    priority = "internal"  # Default to internal for Smart Analysis
                elif (
                    source == "file"
                    or source in ["mattermost", "nextcloud", "enterprise", "api", "email_imap"]
                    or source_type
                    in [
                        "local_document",
                        "mattermost_post",
                        "nextcloud_file",
                        "email_message",
                        "enterprise_email",
                    ]
                ):
                    # Both internal files and enterprise sources get same high priority
                    priority = "internal"
                elif source == "url" or source_type == "search_result":
                    priority = "external"
                else:
                    priority = "research"

                # Special handling for AI synthesis content
                if source_id == "skip":
                    # Check if this is AI synthesis with underlying citations
                    doc_id = item.get("doc_id")
                    if doc_id and doc_id in self.citation_registry.documents:
                        doc_info = self.citation_registry.documents[doc_id]
                        underlying_citations = doc_info.underlying_docs

                        if underlying_citations:
                            # Include the AI synthesis content which already contains DOC references
                            # The AI synthesis content should be used as-is since it has proper citations
                            content = item["content"]
                            if len(content) > max_content_length:
                                content = self._smart_truncate_content(content, max_content_length)

                            # Check if content has DOC references - if so, preserve them
                            has_doc_refs = "[DOC:" in content

                            content_summary = {
                                "doc_id": item["doc_id"],
                                "text": content,
                                "source_type": "ai_synthesis",
                                "relevance": item["score"],
                                "citation_id": "HAS_DOC_REFS" if has_doc_refs else "NO_CITATION",
                                "source_priority": priority,  # Use dynamically determined priority
                                "source_description": f"AI synthesis from {len(underlying_citations)} sources",
                                "underlying_doc_ids": underlying_citations,  # Store underlying doc citations
                            }
                            content_for_synthesis.append(content_summary)

                            # Note: Don't add underlying citation IDs to seen_citation_ids
                            # This allows the underlying sources to still be included as regular sources
                            # Only the AI synthesis doc_id itself should be marked as seen
                    # Skip all items with source_id "skip" (AI synthesis without sources or already processed)
                    continue

                # Skip duplicate citation IDs to prevent redundant citations in synthesis
                doc_id = item["doc_id"]
                if doc_id in seen_citation_ids:
                    continue
                seen_citation_ids.add(doc_id)

                # Prepare content summary with intelligent truncation
                content = item["content"]
                if len(content) > max_content_length:
                    content = self._smart_truncate_content(content, max_content_length)
                content_summary = {
                    "doc_id": item["doc_id"],
                    "text": content,
                    "source_type": source,
                    "relevance": item["score"],
                    "citation_id": item["doc_id"],  # Use actual doc_id for citation reference
                    "source_priority": priority,
                    "source_description": self._get_source_description_for_synthesis(metadata),
                }
                content_for_synthesis.append(content_summary)

            # Check if we need to reduce content due to token limits
            estimated_tokens = len(json.dumps(content_for_synthesis)) + len(json.dumps(global_sources)) + 2000
            if estimated_tokens > self.max_total_tokens:  # Use configurable token limit
                logger.debug(f"Content too large ({estimated_tokens} tokens), applying adaptive truncation")
                content_for_synthesis = self._adaptive_content_batching(content_for_synthesis, max_content_length // 2)

            # Generate thematic synthesis with citation instructions
            synthesis_prompt = f"""
As an expert research analyst, synthesize the following content into a coherent, insightful, and well-supported analysis for the theme: "{theme}" directly related to the overarching research question: "{context.original_question}"

Content for Synthesis (ordered by source importance):
{json.dumps(content_for_synthesis, indent=2)}

Available Citations for Inline Reference:
{json.dumps(global_sources, indent=2)}

Source Priority Guidelines:
1.  **"internal"**: Highest priority (e.g., internal company documents, proprietary files, confidential reports, enterprise chat messages, local documents, CRM data, internal APIs, project management tools). Insights from these sources should form the primary foundation of the analysis.
2.  **"external"**: Medium priority (e.g., public web sources, academic papers, industry reports, news articles). Use these to provide broader context, external validation, or contrasting perspectives.
3.  **"research"**: Supporting priority (e.g., AI-generated research findings, analytical synthesis from multiple sources, preliminary data from exploratory analysis). These provide valuable cross-source insights and meta-analysis that connect patterns across multiple documents.

**Synthesis Requirements:**

* **QUANTITATIVE PRIORITY:** Lead with numerical data, calculations, and aggregations
    * Extract ALL percentages, costs, metrics, and performance data
    * Perform mathematical operations: aggregate percentages, calculate increases, sum totals
    * Example: "Finance customers (35%) combined with healthcare (40%) represent 75% of regulated industry concerns [DOC:doc_1][DOC:doc_2]"
    * Example: "The $2,000,000 investment represents a 20% increase over baseline costs [DOC:doc_3]"

* **ACHIEVEMENT & OUTCOME EMPHASIS:**
    * Prioritize: win/loss improvements, retention gains, cost savings, efficiency increases
    * Look for: before/after comparisons, ROI data, performance benchmarks
    * Extract: specific business results and quantifiable achievements
    * Use phrases like "achieved", "improved", "increased", "reduced", "enhanced"

* **FACT VERIFICATION (CRITICAL):**
    * ONLY state what documents explicitly contain - no inference or extrapolation
    * Use exact quotes for key numerical claims: "As stated in the document: '[exact quote]' [DOC:doc_id]"
    * If documents reference "internal analyses" without providing results, state: "document references internal analysis but doesn't provide specific findings"
    * Never claim data exists if not explicitly shown in documents

* **Narrative Flow with Quantitative Foundation:**
    * Lead each paragraph with specific numerical findings or achievements
    * Support with qualitative context and implications
    * Maintain logical flow while preserving exact data points
    * Connect quantitative findings to business impact

* **Mathematical Transparency:**
    * Show calculations when aggregating data: "Combined percentage: 35% + 40% = 75% [DOC:doc_1][DOC:doc_2]"
    * Explain derivations: "20% cost increase calculated from $2M investment against $10M baseline [DOC:doc_3]"

* **Citation Usage (Critical):**
    * **Format:** Instead of using citation numbers, reference sources by their document ID: "Internal review shows 15% increase [DOC:doc_079c2e0f_1752503636]"
    * **Use INDIVIDUAL citations:** [DOC:doc_1][DOC:doc_2] NOT [DOC:doc_1; DOC:doc_2]
    * **PRESERVE EXISTING CITATIONS:** When reusing AI synthesis content with [DOC:doc_id] citations, preserve exactly as they appear
    * **ACTIVELY RETRIEVE CITATIONS:** For AI synthesis without explicit citations, add citations to underlying documents from "underlying_doc_ids"
    * **NEVER HALLUCINATE CITATIONS:** Only use provided doc_id values - never invent citation IDs
    * **Cite every numerical claim and calculation with source documents**

* **Conciseness & Comprehensiveness:** 2-4 substantive paragraphs with quantitative focus
* **Integration:** Seamlessly weave numerical findings with supporting context
* **Output Format:** Markdown with inline [DOC:doc_id] citations only - no reference lists

Generate 2-4 paragraphs of synthesized analysis with proper inline citations:
"""

            try:
                synthesis = self._generate_text(synthesis_prompt, log_name="report_synthesis")

                # Clean up any spurious citation sections that the LLM might have added
                synthesis = self._clean_spurious_citations(synthesis)

                # Keep synthesis in [DOC:doc_id] format - citation resolution will happen at final assembly
                # This prevents premature conversion that causes citation duplication

                # Don't clean citations here - the LLM might reference documents from other themes
                # We'll handle orphaned citations later when generating references
                synthesized_sections[theme] = synthesis

                # Track metadata for this synthesis
                self._track_synthesis_metadata(theme, prioritized_items)

            except Exception as e:
                logger.warning(f"Theme synthesis failed for {theme}: {e}")
                synthesized_sections[theme] = (
                    f"Content analysis available for {len(prioritized_items)} sources in {theme}"
                )

        return synthesized_sections

    def _clean_spurious_citations(self, synthesis_text: str) -> str:
        """Remove any spurious citation sections that the LLM might have added to individual sections"""

        # Remove any standalone citation lines that look like section-level references
        # Pattern matches lines like "[^1]: **Title** - Description"
        citation_line_pattern = r"^\s*\[.*?\]:.*$"

        lines = synthesis_text.split("\n")
        cleaned_lines = []
        skip_next_lines = False

        for line in lines:
            # Check if this looks like a citation line
            if re.match(citation_line_pattern, line.strip()):
                # This is a spurious citation line, skip it
                skip_next_lines = True
                continue
            elif skip_next_lines and line.strip() == "":
                # Skip empty lines after citation blocks
                continue
            else:
                skip_next_lines = False
                cleaned_lines.append(line)

        # Remove any trailing empty lines
        while cleaned_lines and cleaned_lines[-1].strip() == "":
            cleaned_lines.pop()

        return "\n".join(cleaned_lines)

    def _smart_truncate_content(self, content: str, max_length: int) -> str:
        """Smart truncation that preserves important content and citations"""
        if len(content) <= max_length:
            return content

        # If content is much larger than max_length, do a simple intelligent cut first
        if len(content) > max_length * 3:
            # Try to find a good cut point in the latter 2/3 of max_length
            cut_point = max_length
            # Look for sentence boundaries
            for i in range(max_length - 100, min(len(content), max_length + 100)):
                if content[i] == "." and i + 1 < len(content) and content[i + 1] in " \n":
                    cut_point = i + 1
                    break
            content = content[:cut_point]

        # Strategy 1: Try to keep the most important parts
        # Look for structured content (bullets, numbered lists, conclusions)

        # Split into paragraphs - try different separators
        paragraphs = []
        if "\n\n" in content:
            paragraphs = content.split("\n\n")
        elif "\n" in content:
            paragraphs = content.split("\n")
        else:
            # Fallback: split by sentences
            import re

            paragraphs = re.split(r"(?<=[.!?])\s+", content)

        # Priority order for content selection:
        # 1. Conclusions, summaries, key insights
        # 2. Numbered or bulleted lists
        # 3. First paragraph (often contains main points)
        # 4. Paragraphs with specific data/numbers

        important_paragraphs = []
        regular_paragraphs = []

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para or len(para) < 10:  # Skip very short paragraphs
                continue

            # Check if paragraph contains important indicators
            is_important = (
                any(
                    keyword in para.lower()
                    for keyword in [
                        "conclusion",
                        "summary",
                        "key insight",
                        "important",
                        "critical",
                        "recommendation",
                        "action",
                        "next steps",
                        "findings",
                        "results",
                        "key finding",
                        "main point",
                        "essential",
                        "crucial",
                    ]
                )
                or para.startswith(("1.", "2.", "3.", "•", "-", "*", "#"))  # Lists and headers
                or any(char.isdigit() and "%" in para for char in para)  # Has percentages
                or len([c for c in para if c.isdigit()]) > 3  # Has multiple numbers
                or i == 0  # First paragraph often important
            )

            if is_important:
                important_paragraphs.append(para)
            else:
                regular_paragraphs.append(para)

        # Build result prioritizing important content
        result = ""
        separator = "\n\n" if "\n\n" in content else "\n"

        # Add important paragraphs first
        for para in important_paragraphs:
            needed_space = len(para) + len(separator)
            if len(result) + needed_space <= max_length:
                if result:
                    result += separator
                result += para
            else:
                # Try to fit a truncated version
                remaining = max_length - len(result) - len(separator) - 3  # -3 for "..."
                if remaining > 50:  # Only if meaningful space
                    if result:
                        result += separator
                    # Find the last complete sentence
                    truncated = para[:remaining]
                    last_sentence = truncated.rfind(".")
                    if last_sentence > remaining * 0.7:
                        result += truncated[: last_sentence + 1] + "..."
                    else:
                        result += truncated + "..."
                break

        # Add regular paragraphs if space allows
        for para in regular_paragraphs:
            needed_space = len(para) + len(separator)
            if len(result) + needed_space <= max_length:
                if result:
                    result += separator
                result += para
            else:
                # Try to fit a partial paragraph
                remaining = max_length - len(result) - len(separator) - 3
                if remaining > 50:
                    if result:
                        result += separator
                    truncated = para[:remaining]
                    last_sentence = truncated.rfind(".")
                    if last_sentence > remaining * 0.7:
                        result += truncated[: last_sentence + 1] + "..."
                    else:
                        result += truncated + "..."
                break

        # Fallback: if result is still empty, just take the beginning
        if not result.strip():
            result = content[: max_length - 3] + "..."

        return result.strip()

    def _adaptive_content_batching(self, content_items: List[Dict], target_length: int) -> List[Dict]:
        """Adaptively batch content items to fit within target length while preserving quality"""
        if not content_items:
            return content_items

        # Sort by priority and relevance
        def priority_score(item):
            priority_map = {"internal": 3, "external": 2, "research": 1}
            source_priority = priority_map.get(item.get("source_priority", "research"), 1)
            relevance = item.get("relevance", 0)
            return source_priority * 10 + relevance

        sorted_items = sorted(content_items, key=priority_score, reverse=True)

        # Build result keeping most important items
        result = []
        current_length = 0

        for item in sorted_items:
            item_length = len(item.get("text", ""))

            # Always include the first item (highest priority)
            if not result:
                if item_length > target_length:
                    # Truncate first item intelligently
                    item["text"] = self._smart_truncate_content(item["text"], target_length)
                result.append(item)
                current_length = len(item["text"])
                continue

            # Check if we can fit this item
            if current_length + item_length <= target_length:
                result.append(item)
                current_length += item_length
            else:
                # Try to fit a truncated version
                remaining = target_length - current_length
                if remaining > 200:  # Only if meaningful space remains
                    item["text"] = self._smart_truncate_content(item["text"], remaining)
                    result.append(item)
                break

        logger.debug(f"Adaptive batching: {len(content_items)} -> {len(result)} items ({current_length} chars)")
        return result

    def _get_highest_priority_from_source_docs(self, source_document_ids: List[str]) -> str:
        """Determine the highest priority from a list of source document IDs"""
        if not source_document_ids:
            return "research"

        priorities = []
        for doc_id in source_document_ids:
            if self.vector_store:
                # Get document from vector store to check its metadata
                docs = self.vector_store.search(f"doc_id:{doc_id}", top_k=1)
                if docs:
                    doc = docs[0]
                    metadata = doc.get("metadata", {})
                    source = metadata.get("source", "")
                    source_type = metadata.get("source_type", metadata.get("type", ""))

                    # Apply same priority logic as main function
                    if (
                        source == "file"
                        or source in ["mattermost", "nextcloud", "enterprise", "api", "email_imap"]
                        or source_type
                        in [
                            "local_document",
                            "mattermost_post",
                            "nextcloud_file",
                            "email_message",
                            "enterprise_email",
                        ]
                    ):
                        # Both internal files and enterprise sources get same high priority
                        priorities.append("internal")
                    elif source == "url" or source_type == "search_result":
                        priorities.append("external")
                    else:
                        priorities.append("research")

        # Return highest priority found (internal > external > research)
        priority_order = ["internal", "external", "research"]
        for priority in priority_order:
            if priority in priorities:
                return priority

        return "research"  # Default fallback

    def _prioritize_sources(self, content_items: List[Dict]) -> List[Dict]:
        """Prioritize sources: internal docs (including enterprise) > external web sources > research"""
        internal_sources = []  # Local files, internal documents, enterprise sources
        external_sources = []  # Web URLs, search results
        synthesized_sources = []  # AI-generated content
        research_sources = []  # research findings

        for item in content_items:
            metadata = item["metadata"]
            source = metadata.get("source", "")
            source_type = metadata.get("source_type", metadata.get("type", ""))
            tool_used = metadata.get("tool_used", "")

            # Classify sources by priority level
            if (
                source == "file"
                or source in ["mattermost", "nextcloud", "enterprise", "api", "email_imap"]
                or source_type
                in [
                    "local_document",
                    "mattermost_post",
                    "nextcloud_file",
                    "email_message",
                    "enterprise_email",
                ]
            ):
                # Both internal files and enterprise sources get same high priority
                internal_sources.append(item)
            elif source == "url" or source_type == "search_result":
                external_sources.append(item)
            elif source_type == "ai_synthesis_with_sources":
                synthesized_sources.append(item)  # These will have their sources properly cited
            else:
                research_sources.append(item)

        internal_sources.sort(key=lambda x: x["score"], reverse=True)
        external_sources.sort(key=lambda x: x["score"], reverse=True)
        synthesized_sources.sort(key=lambda x: x["score"], reverse=True)
        research_sources.sort(key=lambda x: x["score"], reverse=True)

        # Priority order: internal (local docs + enterprise) > synthesized > external > research
        return internal_sources + synthesized_sources + external_sources + research_sources

    def _get_source_description_for_synthesis(self, metadata: Dict) -> str:
        """Generate a brief source description for synthesis prompts"""
        source = metadata.get("source", "")
        source_type = metadata.get("type", "")

        if source == "file":
            return f"internal document ({metadata.get('filename', 'unknown')})"
        elif source == "mattermost" or source_type == "mattermost_post":
            return f"enterprise chat message from {metadata.get('user_id', 'unknown user')}"
        elif source == "email_imap" or source_type in ["email_message", "enterprise_email"]:
            return f"enterprise email from {metadata.get('sender', metadata.get('from', 'unknown sender'))}"
        elif source == "nextcloud" or source_type == "nextcloud_file":
            return f"enterprise file from Nextcloud ({metadata.get('filename', metadata.get('name', 'unknown'))})"
        elif source == "url" or source_type == "search_result":
            return f"web source ({metadata.get('title', metadata.get('url', 'unknown'))})"
        else:
            return f"research finding ({source or source_type or 'unknown'})"

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain name from URL for use as fallback title"""
        try:
            from urllib.parse import urlparse

            if not url or url == "Unknown URL":
                return ""

            parsed = urlparse(url)
            domain = parsed.netloc

            # Remove 'www.' prefix if present
            if domain.startswith("www."):
                domain = domain[4:]

            # Capitalize first letter for better presentation
            return domain.capitalize() if domain else ""
        except Exception:
            return ""

    def _generate_internal_search_queries(self, original_question: str, research_plan: Any) -> List[str]:
        """Generate targeted search queries for comprehensive content discovery"""

        actionable_tasks = []
        if (
            research_plan
            and isinstance(research_plan, dict)
            and "research_investigation_areas" in research_plan
            and isinstance(research_plan["research_investigation_areas"], list)
        ):
            actionable_tasks = research_plan["research_investigation_areas"]

        research_plan_length = len(actionable_tasks) if actionable_tasks else 5  # Default to 5 if no tasks
        research_questions = (
            "\n - ".join(
                [f'{task.get("research_focus", "")}: {task.get("business_rationale", "")}' for task in actionable_tasks]
            )
            if actionable_tasks
            else original_question
        )

        query_prompt = f"""
As an expert internal research assistant, generate {research_plan_length}-{2 * research_plan_length} highly specific and targeted search queries to comprehensively investigate the following within internal enterprise services (e.g., chat logs, emails, file shares, CRM records, internal wikis, project management tools):

Original Research Question: "{original_question}"

Key Research Plan Questions:
- {research_questions}

Each query should be designed to uncover relevant internal communications, documents, and data, covering the following aspects:

1.  **Direct References & Project/Topic Names:** Queries that directly use exact names of projects, initiatives, departments, or specific terms mentioned in the original question or research plan.
2.  **Key Personnel & Teams:** Queries involving the names of individuals, teams, or departments likely to be involved in discussions or holding relevant information.
3.  **Decision-Making & Approvals:** Queries to find discussions, emails, or documents related to specific decisions, approvals, justifications, or policy changes.
4.  **Problem Solving & Issue Resolution:** Queries focused on identifying discussions around challenges, solutions, bug reports, customer issues, or internal incidents.
5.  **Documentation & Data Artifacts:** Queries for specific document types (e.g., "meeting minutes", "proposal", "report", "spec", "SOP"), file names, or data points within CRM/databases.
6.  **Timeline & Date-Specific Information:** Queries incorporating dates, date ranges, or time-sensitive keywords (e.g., "Q1 2024", "last month", "since June 2023") if relevant to the research.
7.  **Rationale & Background Discussions:** Queries aiming to uncover the "why" behind decisions, the context of projects, or early-stage brainstorming.
8.  **Internal Stakeholder Communications:** Queries focusing on discussions between specific internal groups (e.g., "sales marketing sync", "engineering product review").

Prioritize using exact phrases where possible (e.g., by enclosing in quotes for tools that support it). Consider variations in how terms might be used internally (e.g., abbreviations, internal jargon). Avoid overly generic terms that would yield too many irrelevant results. Assume basic keyword search functionality is available across services.

Return only the queries, one per line. Do not include any additional text, enumeration, or formatting.
"""

        try:
            response = self._generate_text(query_prompt, log_name="report_internal_queries")
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            return queries
        except:
            # Fallback queries
            fallback_tasks = actionable_tasks if actionable_tasks else []
            return [
                original_question,
                f"background {original_question}",
                f"current trends {original_question}",
                f"analysis {original_question}",
                f"implementation {original_question}",
            ] + [task.get("research_focus", original_question) for task in fallback_tasks]

    def _is_internal_source(self, metadata: Dict) -> bool:
        """Determine if a source is internal/enterprise based on metadata"""
        source = metadata.get("source", "")
        source_type = metadata.get("source_type", metadata.get("type", ""))

        # Check for internal/enterprise indicators
        return (
            source in ["file", "mattermost", "nextcloud", "enterprise", "email_imap"]
            or source_type
            in ["local_document", "mattermost_post", "nextcloud_file", "internal_document", "email_message", "enterprise_email"]
            or "mattermost" in str(metadata.get("filename", "")).lower()
            or "internal" in str(metadata.get("query_context", "")).lower()
        )

    def _build_theme_source_registry(self, content_items: List[Dict]) -> Dict[str, Dict[str, str]]:
        """Build source registry for theme-specific citations using unified registry"""
        theme_sources = {}

        for item in content_items:
            source_id = self._get_source_citation_id(item)
            doc_id = item["doc_id"]

            # Skip entries that shouldn't have citations (like AI synthesis with sources)
            if source_id == "skip":
                continue

            # Get the source info from the unified registry using doc_id as key
            if doc_id in self.citation_registry.documents:
                source_info = self.citation_registry.documents[doc_id].source_info
                # Use doc_id as the key, not the source_id ("registered")
                theme_sources[doc_id] = source_info
            else:
                # This shouldn't happen, but just in case
                logger.warning(f"Document {doc_id} not found in citation registry")

        return theme_sources

    def _get_source_citation_id(self, item: Dict) -> str:
        """Register document in unified citation registry and return registration status.

        Returns "skip" for AI synthesis documents, "registered" for regular documents.
        """
        doc_id = item["doc_id"]
        source_info = self._extract_source_info(item)

        # Handle documents that should skip citations
        if source_info.get("type") == "skip_citation":
            return "skip"
        
        # Handle AI synthesis documents
        if source_info.get("type") == "ai_synthesis_with_sources":
            source_doc_ids = source_info.get("source_document_ids", [])
            logger.debug(f"AI synthesis has {len(source_doc_ids)} source documents to register")

            # Register the underlying source documents
            for source_doc_id in source_doc_ids:
                if self.vector_store:
                    try:
                        doc_data = self.vector_store.get_document(source_doc_id)
                        if doc_data:
                            source_metadata = doc_data.get("metadata", {})
                            source_info_extracted = self._extract_source_info(
                                {"doc_id": source_doc_id, "metadata": source_metadata}
                            )
                            self.citation_registry.register_document(
                                doc_id=source_doc_id, source_info=source_info_extracted, document_type="regular"
                            )
                    except Exception as e:
                        logger.debug(f"Could not register source document {source_doc_id}: {e}")

            # Register AI synthesis with underlying docs
            self.citation_registry.register_document(
                doc_id=doc_id, source_info=source_info, underlying_docs=source_doc_ids, document_type="ai_synthesis"
            )
            return "skip"

        elif source_info.get("type") == "ai_synthesis":
            # Register AI synthesis without sources
            self.citation_registry.register_document(
                doc_id=doc_id, source_info=source_info, document_type="ai_synthesis"
            )
            return "skip"

        # Register regular document
        self.citation_registry.register_document(doc_id=doc_id, source_info=source_info, document_type="regular")
        return "registered"

    # _register_source_documents method removed - replaced by direct registration in _get_source_citation_id

    def _extract_source_info(self, item: Dict) -> Dict[str, str]:
        """Extract source information for citations with enhanced support for all source types"""
        metadata = item["metadata"]

        # Handle Mattermost/chat sources first (from EnterpriseAPITool or local JSONL)
        if metadata.get("source") == "mattermost" or metadata.get("type") == "mattermost_post":
            # Use resolved names if available, otherwise fall back to IDs
            # Note: enterprise_tools.py stores as 'user_name' not 'username'
            username = metadata.get("user_name", metadata.get("username", metadata.get("user_id", "Unknown User")))
            channel_name = metadata.get("channel_name", metadata.get("channel_id", "Unknown Channel"))
            team_name = metadata.get("team_name", "")
            return {
                "type": "enterprise_chat",
                "title": f"Mattermost Message",
                "channel": channel_name,
                "team": team_name,
                "user": username,
                "message_preview": metadata.get("message_preview", ""),
                "description": f"Mattermost message from {username}",
                "timestamp": metadata.get("timestamp", ""),
            }
        # Handle Email/IMAP sources first (from EmailAdapter or local JSONL)
        elif metadata.get("source") == "email_imap" or metadata.get("type") == "email_message":
            # Extract email details similar to Mattermost handling
            sender = metadata.get("sender", metadata.get("from", "Unknown Sender"))
            subject = metadata.get("subject", metadata.get("title", ""))
            date = metadata.get("date", metadata.get("timestamp", "Unknown Date"))
            email_id = metadata.get("email_id", metadata.get("id", "unknown"))
            # Use subject as title, fallback to content preview
            if not subject or subject.strip() == "":
                subject = item.get("content", "")[:50].split("\n")[0]
                if len(subject) > 47:
                    subject = subject[:47] + "..."
            elif len(subject) > 80:
                subject = subject[:77] + "..."
            return {
                "type": "enterprise_email",
                "title": subject or "Email Message",
                "sender": sender,
                "from": sender,  # Keep both for compatibility
                "date": date,
                "source": "email_imap",
                "description": f"Email from {sender}",
                "timestamp": metadata.get("timestamp", ""),
                "email_id": email_id,
            }
        # Handle local document sources (from LocalDocumentIngestionTool)
        elif metadata.get("source_type") == "local_document":
            file_path = metadata.get("file_path", "Unknown Path")
            relative_path = metadata.get("relative_path", metadata.get("filename", "Unknown"))
            return {
                "type": "internal",
                "title": relative_path,
                "path": file_path,
                "description": f"Local document: {relative_path}",
                "timestamp": metadata.get("ingestion_time", metadata.get("timestamp", "")),
                "folder_path": metadata.get("folder_path", ""),
                "file_size": metadata.get("file_size_bytes", 0),
            }

        # Handle internal file sources (from ContentProcessor)
        elif metadata.get("source") == "file":
            # Check if this is actually a web source that was downloaded
            if metadata.get("url"):
                return {
                    "type": "external",
                    "title": metadata.get("title", metadata.get("filename", "Web Document")),
                    "url": metadata.get("url"),
                    "description": f"Downloaded web document: {metadata.get('title', metadata.get('filename', 'Unknown'))}",
                    "timestamp": metadata.get("timestamp", ""),
                }
            else:
                return {
                    "type": "internal",
                    "title": metadata.get("filename", "Internal Document"),
                    "path": metadata.get(
                        "original_path", metadata.get("file_path", metadata.get("path", "Unknown Path"))
                    ),
                    "description": f"Internal document: {metadata.get('filename', 'Unknown')}",
                    "timestamp": metadata.get("timestamp", ""),
                }

        # Handle web URL sources (from ContentProcessor URL processing)
        elif metadata.get("source") == "url":
            # Try to get a meaningful title from various metadata fields
            title = (
                metadata.get("title")
                or metadata.get("filename")
                or self._extract_domain_from_url(metadata.get("url", ""))
                or "Web Source"
            )
            return {
                "type": "external",
                "title": title,
                "url": metadata.get("url", "Unknown URL"),
                "description": f"Web source: {title}",
                "timestamp": metadata.get("timestamp", ""),
            }

        # Handle search result sources (from InternetSearchTool)
        elif metadata.get("type") == "search_result":
            return {
                "type": "external",
                "title": metadata.get("title", "Search Result"),
                "url": metadata.get("url", "Unknown URL"),
                "description": f"Search result: {metadata.get('title', 'Unknown')}",
                "search_rank": metadata.get("search_rank", "N/A"),
                "timestamp": metadata.get("timestamp", ""),
            }

        # Handle Nextcloud/file server sources (from EnterpriseAPITool)
        elif metadata.get("source") == "nextcloud" or metadata.get("type") == "nextcloud_file":
            # Use the original Nextcloud path if available, otherwise fall back to other paths
            nextcloud_path = metadata.get(
                "nextcloud_path",
                metadata.get(
                    "path",
                    metadata.get("file_path", metadata.get("original_path", "Unknown Path")),
                ),
            )

            # Extract proper filename from Nextcloud path or use provided filename
            # Priority: nextcloud_filename > extract from nextcloud_path > filename > fallback
            title = metadata.get("nextcloud_filename")
            if not title and nextcloud_path:
                # Extract filename from path like "/remote.php/dav/files/admin/shared/Certified_Professionals.pdf"
                title = nextcloud_path.split("/")[-1] if "/" in nextcloud_path else nextcloud_path
            if not title:
                title = metadata.get("filename", metadata.get("name", "Enterprise File"))

            # Avoid using temp file names as titles
            if title and title.startswith("tmp") and len(title) < 15:
                # This looks like a temp file, try to get better name from path
                if nextcloud_path and "/" in nextcloud_path:
                    title = nextcloud_path.split("/")[-1]
                else:
                    title = "Nextcloud File"

            return {
                "type": "enterprise_file",
                "title": title,
                "path": nextcloud_path,
                "server": "Nextcloud",
                "description": f"Nextcloud file: {title}",
                "timestamp": metadata.get("timestamp", metadata.get("last_modified", "")),
            }

        # Handle FileBrowser sources (from EnterpriseAPITool)
        elif metadata.get("source") == "filebrowser" or metadata.get("service_name") == "filebrowser":
            # Get the original FileBrowser path and filename
            filebrowser_path = metadata.get("original_path", metadata.get("file_path", "Unknown Path"))

            # Extract proper filename from FileBrowser path or use provided filename
            # Priority: file_name > extract from original_path > filename > fallback
            title = metadata.get("file_name")
            if not title and filebrowser_path:
                # Extract filename from path like "/AI_Act_Compliance_Monitoring_Process.pptx"
                title = filebrowser_path.split("/")[-1] if "/" in filebrowser_path else filebrowser_path
            if not title:
                title = metadata.get("filename", "FileBrowser File")

            # Avoid using temp file names as titles
            if title and title.startswith("tmp") and len(title) < 15:
                # This looks like a temp file, try to get better name from path
                if filebrowser_path and "/" in filebrowser_path:
                    title = filebrowser_path.split("/")[-1]
                else:
                    title = "FileBrowser File"

            return {
                "type": "enterprise_file",
                "title": title,
                "path": filebrowser_path,
                "server": "FileBrowser",
                "description": f"FileBrowser file: {title}",
                "timestamp": metadata.get("timestamp", ""),
            }

        elif metadata.get("tool_used") == "smart_analysis" or metadata.get("type") in [
            "ai_synthesis_with_sources",
            "ai_synthesis",
        ]:
            source_doc_ids = metadata.get("source_document_ids", [])
            docs_analyzed = (
                metadata.get("docs_analyzed") or len(source_doc_ids) or metadata.get("documents_analyzed", 0)
            )

            # Use explicit type if provided, otherwise infer from source_doc_ids
            synthesis_type = metadata.get("type")
            if not synthesis_type:
                synthesis_type = "ai_synthesis_with_sources" if source_doc_ids else "ai_synthesis"

            if synthesis_type == "ai_synthesis_with_sources" or source_doc_ids:
                return {
                    "type": "ai_synthesis_with_sources",
                    "title": metadata.get("title", "AI Research Synthesis"),
                    "source_tool": metadata.get("tool_used", "smart_analysis"),
                    "description": f"AI-generated analysis synthesized from {docs_analyzed} documents",
                    "docs_analyzed": docs_analyzed,
                    "source_document_ids": source_doc_ids,
                    "synthesis_method": metadata.get("synthesis_method", "vector_search"),
                    "timestamp": metadata.get("timestamp", ""),
                }
            else:
                return {
                    "type": synthesis_type,
                    "title": metadata.get("title", "AI Research Analysis"),
                    "source_tool": metadata.get("tool_used", "smart_analysis"),
                    "description": f"AI-generated research analysis",
                    "docs_analyzed": docs_analyzed,
                    "synthesis_method": metadata.get("synthesis_method", "vector_search"),
                    "timestamp": metadata.get("timestamp", ""),
                }

        # Handle other enterprise API sources
        elif metadata.get("source") in ["enterprise", "api"] or (metadata.get("tool_used") and metadata.get("tool_used") != "local_document_search"):
            tool_used = metadata.get("tool_used", "Unknown Tool")
            api_type = metadata.get("api_type", "")
            query_context = metadata.get("query_context", "")
            service_name = metadata.get("service_name", "")

            if tool_used == "enhanced_enterprise_api":
                if metadata.get("source") == "mattermost":
                    service_name = "Mattermost"
                elif metadata.get("source") == "nextcloud":
                    service_name = "Nextcloud"
                elif metadata.get("source") == "filebrowser":
                    service_name = "FileBrowser"
                elif "file" in str(metadata.get("original_path", "")).lower():
                    service_name = "File Server"
                elif metadata.get("filename"):
                    service_name = "Enterprise File System"
                elif "search" in str(query_context).lower():
                    service_name = "Enterprise Search"
            if service_name:
                title = f"{service_name} Enterprise Data"
            elif api_type:
                title = f"Enterprise {api_type.title()} Data"
            elif query_context:
                # Truncate query context to reasonable length
                context_snippet = query_context[:50] + "..." if len(query_context) > 50 else query_context
                title = f"Enterprise Search: {context_snippet}"
            else:
                title = "Enterprise API Data"

            return {
                "type": "enterprise_api",
                "title": title,
                "source_tool": tool_used,
                "description": f"Enterprise data retrieved via {tool_used}",
                "timestamp": metadata.get("timestamp", ""),
            }

        # Handle research findings (vector store synthesis results)
        elif metadata.get("type") == "research_finding" or not metadata.get("source"):
            # Check if this might be a research synthesis with source information
            if metadata.get("synthesized_from_sources"):
                return {
                    "type": "synthesis",
                    "title": "Research Synthesis",
                    "description": f"Analysis synthesized from {len(metadata.get('synthesized_from_sources', []))} sources",
                    "source_count": len(metadata.get("synthesized_from_sources", [])),
                    "timestamp": metadata.get("timestamp", ""),
                }
            else:
                # Skip citations for generic research findings to avoid "Unknown source" entries
                return {
                    "type": "skip_citation",
                    "title": "Research Finding",
                    "description": "AI-generated research finding (citation skipped)",
                    "timestamp": metadata.get("timestamp", ""),
                }

        # Fallback for unrecognized sources
        else:
            return {
                "type": "unknown",
                "title": f"Unknown Source ({metadata.get('type', 'unspecified')})",
                "description": f"Source type: {metadata.get('source', metadata.get('type', 'unknown'))}",
                "timestamp": metadata.get("timestamp", ""),
            }

    def _get_retrieval_description(self, source_info: Dict[str, str]) -> str:
        """Get a concise description of how the source was retrieved"""
        source_type = source_info.get("type", "unknown")

        if source_type == "enterprise_file":
            return f"{source_info.get('title', 'File')} ({source_info.get('server', 'Enterprise')})"
        elif source_type == "enterprise_chat":
            user = source_info.get("user", "Unknown")
            channel = source_info.get("channel", "Unknown")
            return f"Message from {user} in #{channel}"
        elif source_type == "external":
            return source_info.get("title", "Web Source")
        else:
            return source_info.get("description", "Unknown Source")

    def _track_synthesis_metadata(self, theme: str, content_items: List[Dict]):
        """Track metadata for synthesized content"""

        for item in content_items:
            metadata = item["metadata"]
            doc_id = item["doc_id"]

            # Add to document_ids if not already present
            if doc_id and doc_id not in self.evidence_metadata["document_ids"]:
                self.evidence_metadata["document_ids"].append(doc_id)

            # Categorize in evidence_map
            if metadata.get("source") == "url":
                url = metadata.get("url")
                if url and url not in self.evidence_metadata["document_ids"]:
                    self.evidence_metadata["document_ids"].append(url)

                self.evidence_metadata["evidence_map"]["web_sources"].append(
                    {
                        "url": url,
                        "title": metadata.get("title", "Unknown"),
                        "doc_id": doc_id,
                        "relevance_score": item["score"],
                        "content_length": len(item["content"]),
                        "theme_coverage": [theme],
                        "date_accessed": metadata.get("timestamp", ""),
                        "content_type": metadata.get("content_type", "unknown"),
                    }
                )

            elif metadata.get("source") == "file":
                file_path = metadata.get("original_path") or metadata.get("file_path")
                if file_path and file_path not in self.evidence_metadata["document_ids"]:
                    self.evidence_metadata["document_ids"].append(file_path)

                self.evidence_metadata["evidence_map"]["internal_sources"].append(
                    {
                        "file_path": file_path,
                        "filename": metadata.get("filename", "Unknown"),
                        "doc_id": doc_id,
                        "relevance_score": item["score"],
                        "content_length": len(item["content"]),
                        "theme_coverage": [theme],
                        "content_type": metadata.get("content_type", "unknown"),
                    }
                )

            else:
                self.evidence_metadata["evidence_map"]["research_findings"].append(
                    {
                        "source": metadata.get("tool_used", "unknown"),
                        "doc_id": doc_id,
                        "relevance_score": item["score"],
                        "theme_coverage": [theme],
                        "finding_type": metadata.get("type", "research_finding"),
                    }
                )

    def _extract_clean_question(self, original_question: str) -> str:
        """Extract the actual question from an enhanced query or return the original question"""
        if "QUESTION:" in original_question:
            # Split by QUESTION: and take everything after it
            parts = original_question.split("QUESTION:", 1)
            if len(parts) > 1:
                # Clean up the extracted question
                clean_question = parts[1].strip()
                # Remove any trailing context that might be after the question
                # (like persona or additional context that sometimes appears after)
                return clean_question.split("\n")[0].strip()
        return original_question

    def _assemble_final_report(
        self,
        synthesized_sections: Dict[str, str],
        evidence_summary: Dict[str, Any],
        context: ResearchContext,
    ) -> str:
        """Assemble the final clean, readable report"""

        # Generate executive summary
        exec_summary = self._generate_executive_summary(synthesized_sections, context, evidence_summary)

        # Build report structure
        report_parts = []

        # Extract clean question for display
        clean_question = self._extract_clean_question(context.original_question)

        # Header
        report_parts.append(f"# Research Report: {clean_question}\n")

        # Executive Summary
        report_parts.append("## Executive Summary\n")
        report_parts.append(f"{exec_summary}\n")

        # Main Analysis - Dynamic sections based on available themes
        report_parts.append("## Analysis\n")

        # Order themes logically
        theme_order = [
            "background",
            "current_trends",
            "analysis",
            "implementation",
            "future_outlook",
            "general",
        ]
        theme_titles = {
            "background": "Background & Context",
            "current_trends": "Current Trends & Developments",
            "analysis": "Key Research Findings",
            "implementation": "Practical Applications",
            "future_outlook": "Future Outlook",
            "general": "Additional Insights",
        }

        for theme in theme_order:
            if theme in synthesized_sections:
                title = theme_titles.get(theme, theme.replace("_", " ").title())
                report_parts.append(f"### {title}\n")
                report_parts.append(f"{synthesized_sections[theme]}\n")

        # Build the main report content first
        main_report = "\n".join(report_parts)

        # Use unified citation registry for final citation resolution
        final_report, citation_assignments = self.citation_registry.finalize_citations(main_report)

        # Generate references section using unified registry
        references_section = self.citation_registry.generate_references_section()

        # Add references if we have citations
        if citation_assignments:
            final_report += "\n" + references_section
        else:
            final_report += "\n\n---\n"
            final_report += "*Analysis completed without external source citations.*"

        return final_report

    def _generate_executive_summary(
        self,
        synthesized_sections: Dict[str, str],
        context: ResearchContext,
        evidence_summary: Dict[str, Any],
        max_len: int = 1000,
    ) -> str:
        """Generate executive summary from synthesized content"""

        internal_sources_count = len(self.evidence_metadata["evidence_map"]["internal_sources"])
        web_sources_count = len(self.evidence_metadata["evidence_map"]["web_sources"])

        # Extract clean question for executive summary
        clean_question = self._extract_clean_question(context.original_question)

        summary_prompt = f"""
        Generate a concise executive summary for this research report:
        
        Research Question: {clean_question}
        
        Key Findings from Analysis (with internal sources prioritized):
        {json.dumps({k: v[:max_len] + "..." if len(v) > max_len else v for k, v in synthesized_sections.items()}, indent=2)}
        
        Research Scope: {evidence_summary['total_sources']} sources analyzed 
        ({internal_sources_count} internal documents, {web_sources_count} external sources)
        
        Requirements:
        - 2-3 paragraphs maximum
        - Lead with the most important findings, especially from internal sources
        - Emphasize insights from enterprise/internal documents when available
        - Include confidence level based on evidence strength
        - Natural, professional tone
        - Focus on insights and implications, not process
        - Mention source prioritization (internal over external) if relevant
        
        Generate executive summary:
        """

        try:
            return self._generate_text(summary_prompt, log_name="report_exec_summary")
        except Exception as e:
            return f"Key findings synthesized from {evidence_summary['total_sources']} sources across {len(synthesized_sections)} research themes. Analysis reveals comprehensive insights addressing the research question with evidence-based recommendations."

    def _finalize_metadata(self, context: ResearchContext, action_plan, thematic_content: Dict[str, List[Dict]]):
        """Finalize comprehensive metadata for the report"""

        # Content synthesis metadata
        self.evidence_metadata["content_synthesis"] = {
            "total_sources_used": len(self.evidence_metadata["document_ids"]),
            "primary_sources": len(self.evidence_metadata["evidence_map"]["web_sources"]),
            "secondary_sources": len(self.evidence_metadata["evidence_map"]["internal_sources"]),
            "research_findings": len(self.evidence_metadata["evidence_map"]["research_findings"]),
            "content_themes": list(thematic_content.keys()),
            "synthesis_confidence": ("high" if len(self.evidence_metadata["document_ids"]) > 5 else "medium"),
            "thematic_coverage": {theme: len(items) for theme, items in thematic_content.items()},
        }

        # Research methodology metadata
        vector_searches = 0  # VectorStoreRetrievalTool removed

        self.evidence_metadata["research_methodology"] = {
            "vector_searches_performed": vector_searches,
            "content_clustering_method": "thematic",
            "synthesis_approach": "multi-source_cross_validation_with_citations",
            "quality_filters_applied": ["relevance>0.7", "content_length>100"],
            "source_prioritization": "internal_documents_first",
            "citation_system": "markdown_footnotes",
            "total_operations": len(context.findings),
            "successful_operations": sum(
                1 for finding in context.findings.values() if isinstance(finding, dict) and finding.get("success", True)
            ),
            "citations_generated": len(self.citation_registry.documents),
            "internal_vs_external_ratio": self._calculate_source_ratio(),
        }

    def _calculate_source_ratio(self) -> Dict[str, float]:
        """Calculate ratio of internal vs external sources"""
        internal_count = len(self.evidence_metadata["evidence_map"]["internal_sources"])
        external_count = len(self.evidence_metadata["evidence_map"]["web_sources"])
        total = internal_count + external_count

        if total == 0:
            return {"internal_ratio": 0.0, "external_ratio": 0.0}

        return {"internal_ratio": internal_count / total, "external_ratio": external_count / total}

    def get_evidence_metadata(self) -> Dict[str, Any]:
        """Get comprehensive evidence metadata for the generated report"""
        # Include citation registry in metadata
        metadata = self.evidence_metadata.copy()

        # Get citation statistics from unified registry
        citation_stats = self.citation_registry.get_statistics()
        metadata["citation_registry"] = {
            "total_citations": citation_stats["total_citations"],
            "unique_cited_documents": citation_stats["unique_cited_documents"],
            "total_documents": citation_stats["total_documents"],
            "source_mapping": {
                doc_id: {
                    "citation_number": self.citation_registry.get_citation_number(doc_id),
                    "source_type": doc_info.source_info.get("type", "unknown"),
                    "title": doc_info.source_info.get("title", "Untitled"),
                    "document_type": doc_info.document_type,
                }
                for doc_id, doc_info in self.citation_registry.documents.items()
            },
        }
        return metadata
