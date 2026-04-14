import datetime
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import RunConfig
from ..utils import extract_json


@dataclass
class ResearchContext:
    """Accumulates research findings and context throughout the process with bounded memory"""

    original_question: str
    task_id: str | None = None
    plan: Optional[Dict] = None
    findings: Dict[str, Any] = field(default_factory=dict)
    files_created: List[str] = field(default_factory=list)
    apis_discovered: List[Dict] = field(default_factory=list)

    # New fields for bounded context management
    findings_summary: Dict[str, str] = field(default_factory=dict)  # Summarized findings by category
    findings_archive: List[str] = field(default_factory=list)  # Vector store doc IDs for archived findings
    max_findings_size: int = 15  # Keep only recent findings in memory
    vector_store: Optional[Any] = None  # Reference to vector store for archiving

    def add_finding(self, key: str, value: Any, category: str = "general"):
        """Add a finding with automatic archiving of old findings"""
        # Archive oldest findings if we're at capacity
        if len(self.findings) >= self.max_findings_size and self.vector_store:
            # Get the oldest keys (first 5)
            keys_to_archive = list(self.findings.keys())[:5]

            # Create archive document
            archive_content = {
                "archived_findings": {k: self.findings[k] for k in keys_to_archive},
                "timestamp": datetime.datetime.now().isoformat(),
                "research_question": self.original_question,
            }

            # Store in vector store
            doc_id = self.vector_store.store_document(
                content=json.dumps(archive_content, indent=2),
                metadata={"type": "archived_findings", "keys": keys_to_archive, "category": "research_context_archive"},
            )

            if doc_id:
                self.findings_archive.append(doc_id)
                # Remove archived findings from memory
                for k in keys_to_archive:
                    del self.findings[k]

        # Add new finding
        self.findings[key] = value

        # Update category summary if it's substantial
        if isinstance(value, dict) and value.get("summary"):
            self.findings_summary[category] = value.get("summary", str(value)[:200])

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a condensed summary suitable for LLM context"""
        return {
            "question": self.original_question,
            "findings_count": len(self.findings),
            "archived_count": len(self.findings_archive),
            "categories": list(self.findings_summary.keys()),
            "recent_findings": {k: self._summarize_finding(v) for k, v in list(self.findings.items())[-5:]},
            "category_summaries": self.findings_summary,
        }

    def _summarize_finding(self, finding: Any) -> str:
        """Create a brief summary of a finding"""
        if isinstance(finding, dict):
            if "summary" in finding:
                return finding["summary"]
            elif "results" in finding:
                return str(finding["results"])[:200] + "..."
        return str(finding)[:200] + "..."


class Tool(ABC):
    """Base interface for all research tools with standardized output"""

    @property
    @abstractmethod
    def purpose(self) -> str:
        """Return a brief description of what this tool is used for"""
        pass

    @abstractmethod
    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute the tool and return standardized output"""
        pass

    def load_extracted_content(self, result: Dict[str, Any]) -> str:
        """
        Lazy load extracted content from file path
        
        Args:
            result: Tool result containing extracted_path
            
        Returns:
            The extracted content as string, or empty string if not available
        """
        extracted_path = result.get("extracted_path")
        if not extracted_path:
            return ""
            
        try:
            with open(extracted_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    def _standardize_output(
        self,
        raw_output: Dict[str, Any],
        tool_name: str,
        query: str,
        force_success: Optional[bool] = None,
        force_data_retrieved: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Standardize tool output to ensure consistent format

        Args:
            raw_output: The original output from the tool
            tool_name: Name of the tool (e.g., "internet_search")
            query: Original query string
            force_success: Override success detection
            force_data_retrieved: Override data_retrieved detection

        Returns:
            Standardized output dictionary
        """
        # Start with the raw output
        standardized = raw_output.copy()

        # Ensure core fields exist
        standardized["tool"] = tool_name
        standardized["query"] = query

        # Determine success status
        if force_success is not None:
            success = force_success
        elif "success" in raw_output:
            success = raw_output["success"]
        elif "error" in raw_output and raw_output["error"]:
            success = False
        else:
            # Auto-detect success based on content
            success = self._detect_success(raw_output)

        standardized["success"] = success

        # Determine if data was retrieved
        if force_data_retrieved is not None:
            data_retrieved = force_data_retrieved
        elif "data_retrieved" in raw_output:
            data_retrieved = raw_output["data_retrieved"]
        else:
            data_retrieved = self._detect_data_retrieved(raw_output)

        standardized["data_retrieved"] = data_retrieved

        # Ensure error field exists for failures
        if not success and "error" not in standardized:
            standardized["error"] = "Operation failed"

        # Add summary field for easier debugging
        standardized["summary"] = self._generate_summary(standardized)

        return standardized

    def _detect_success(self, output: Dict[str, Any]) -> bool:
        """Auto-detect if operation was successful"""

        # If there's an explicit error, it's not successful
        if output.get("error"):
            return False

        # Check for positive indicators of success
        success_indicators = [
            # Data was found
            output.get("results") and len(output["results"]) > 0,
            output.get("apis_found") and len(output["apis_found"]) > 0,
            output.get("analyzed_files") and len(output["analyzed_files"]) > 0,
            output.get("documents_found", 0) > 0,
            output.get("synthesis") and len(str(output["synthesis"])) > 10,
            output.get("content") and len(str(output["content"])) > 10,
            output.get("processed_files") and len(output["processed_files"]) > 0,
            output.get("urls_processed", 0) > 0,
            output.get("servers_discovered", 0) > 0,
            # File operations succeeded
            output.get("file_path") and os.path.exists(str(output["file_path"])),
            output.get("stored_in_vector", False),
            # HTTP responses
            output.get("status_code") in [200, 201, 202] if "status_code" in output else None,
        ]

        # Return True if any positive indicator exists
        return any(indicator for indicator in success_indicators if indicator is not None)

    def _detect_data_retrieved(self, output: Dict[str, Any]) -> bool:
        """Auto-detect if meaningful data was retrieved"""

        # Check for various forms of retrieved data
        data_indicators = [
            # Explicit data fields
            output.get("results") and len(output["results"]) > 0,
            output.get("apis_found") and len(output["apis_found"]) > 0,
            output.get("analyzed_files") and len(output["analyzed_files"]) > 0,
            output.get("documents_found", 0) > 0,
            output.get("items_found", 0) > 0,
            output.get("processed_files") and len(output["processed_files"]) > 0,
            # Content fields
            output.get("synthesis") and len(str(output["synthesis"])) > 50,
            output.get("content") and len(str(output["content"])) > 50,
            output.get("text_content") and len(str(output["text_content"])) > 50,
            # Check extracted content via file existence and content_length
            output.get("extracted_path") and os.path.exists(str(output["extracted_path"])) and output.get("content_length", 0) > 50,
            # File/storage operations
            output.get("stored_in_vector", False),
            output.get("content_length", 0) > 100,
            output.get("documents_stored", 0) > 0,
            # Network operations
            output.get("data") and len(str(output["data"])) > 10,
            output.get("response_size", 0) > 100,
        ]

        return any(indicator for indicator in data_indicators if indicator is not None)

    def _generate_summary(self, output: Dict[str, Any]) -> str:
        """Generate a brief summary of the tool execution"""

        tool_name = output.get("tool", "unknown")
        success = output.get("success", False)
        data_retrieved = output.get("data_retrieved", False)

        if not success:
            error = output.get("error", "Unknown error")
            return f"{tool_name}: Failed - {error}"

        if data_retrieved:
            # Try to quantify the data retrieved
            data_count = (
                len(output.get("results", []))
                or len(output.get("processed_files", []))
                or output.get("documents_found", 0)
                or output.get("items_found", 0)
                or output.get("urls_processed", 0)
                or output.get("documents_stored", 0)
                or 1  # At least some data
            )
            return f"{tool_name}: Success - Retrieved {data_count} items"
        else:
            return f"{tool_name}: Success - Operation completed"

    def create_success_output(
        self, tool_name: str, query: str, results: Any = None, data_retrieved: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """Helper method to create standardized success output"""

        output = {"tool": tool_name, "query": query, "success": True, "data_retrieved": data_retrieved, **kwargs}

        if results is not None:
            output["results"] = results

        return self._standardize_output(output, tool_name, query)

    def create_error_output(self, tool_name: str, query: str, error: str, **kwargs) -> Dict[str, Any]:
        """Helper method to create standardized error output"""

        output = {
            "tool": tool_name,
            "query": query,
            "success": False,
            "data_retrieved": False,
            "error": error,
            **kwargs,
        }

        return self._standardize_output(output, tool_name, query)


class ToolRegistry:
    """Manages available research tools"""

    def __init__(self):
        self.tools: List[Tool] = []

    def register_tool(self, tool: Tool):
        self.tools.append(tool)

    def select_tools(self, query: str = None, context: ResearchContext = None) -> List[Tool]:
        """Return all available tools - let action planner handle intelligent selection"""
        return self.tools


class FileManager:
    """Handles local file operations"""

    def __init__(self, workspace_dir: str = "./research_workspace"):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)

    def save_file(self, filename: str, content: str) -> str:
        filepath = os.path.join(self.workspace_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def extract_text(self, filepath: str) -> str:
        """Extract text from file (starting with plain text support)"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"


class QueryPlanner:
    """Plans research sections and generates sub-queries"""

    def __init__(self, model: str, llm_adapter: Any, run_config: RunConfig):
        self.model = model
        self.llm_adapter = llm_adapter
        self.run_config = run_config

    def create_research_plan(
        self,
        question: str,
        tool_registry: ToolRegistry = None,
        env: Optional[Any] = None,
    ) -> Dict:
        """Generate structured research plan with sections and sub-queries"""

        available_tools = [f"{tool.__class__.__name__}" for tool in tool_registry.tools]

        enterprise_services = []
        if env is not None:
            enterprise_services = env.get_available_apps()
            enterprise_services = [
                f"{service['name']}: {service['description']}" for service in enterprise_services.values()
            ]
        tools_section = f"""
Available Research Tools:
{chr(10).join(f"- {tool}" for tool in available_tools)}

Available Enterprise Services:
{chr(10).join(f"- {service}" for service in enterprise_services)}
"""

        if self.run_config.report_style == "concise_qa":
            planning_prompt = f"""You are answering a set of numbered research questions. Some answers may be in the provided local documents, others may require web search.

Questions:
"{question}"

{tools_section}

Parse the numbered questions and plan how to find each answer.

Return JSON:
{{
  "research_investigation_areas": [
    {{
      "area_id": 1,
      "research_focus": "the question text",
      "information_needs": ["what specific fact or answer is needed"],
      "knowledge_sources": ["internal", "external", or "both"],
      "key_concepts": ["key search terms"]
    }}
  ]
}}

Create one investigation area per numbered question. For each:
- If the question asks about a specific company's internal data → knowledge_sources: "internal"
- If the question asks about general/public knowledge → knowledge_sources: "external"
- If unclear → knowledge_sources: "both"
"""
        else:
            planning_prompt = f"""
Design a comprehensive enterprise research strategy for: "{question}"

{tools_section}

As a senior enterprise researcher with deep business intelligence expertise, create a thorough investigation plan that combines rigorous research methodology with strategic business analysis. Your goal is to provide insights that drive informed decision-making in complex enterprise environments.

Generate a JSON object with strategic research investigation areas:

{{
  "research_investigation_areas": [
    {{
      "area_id": 1,
      "research_focus": "Core strategic domain, market segment, or business hypothesis to investigate",
      "information_needs": ["What specific intelligence is required for strategic decisions"],
      "knowledge_sources": ["internal" | "external" | "both"],
      "research_approach": "competitive_analysis" | "market_research" | "strategic_assessment" | "trend_analysis" | "risk_analysis" | "performance_benchmarking",
      "key_concepts": ["concept1", "concept2"],
      "business_rationale": "Why this investigation area is critical for enterprise strategy and decision-making",
      "expected_insights": "What strategic understanding or competitive intelligence this area should provide",
      "stakeholder_impact": "Which business units or decision-makers will benefit from these insights",
      "importance_level": "critical" | "important" | "supplementary"
    }}
  ],
  "research_methodology": {{
    "overall_approach": "Description of the integrated research and business intelligence strategy",
    "competitive_positioning": "How this research will inform competitive advantage",
    "knowledge_synthesis": "How different investigation areas will be integrated for strategic recommendations",
    "internal_leverage": "How to maximize insights from proprietary enterprise data and systems",
    "external_validation": "How external market data and industry intelligence will validate internal findings"
  }}
}}

Enterprise Research Design Principles:
- Adopt an enterprise researcher mindset: combine analytical rigor with business acumen
- Leverage proprietary internal data as competitive advantage while ensuring external market context
- Design investigations that directly inform strategic decisions and business outcomes
- Structure research to reveal market opportunities, competitive threats, and operational insights
- Prioritize research areas that maximize ROI and strategic value to the organization
- Consider multiple stakeholder perspectives across different business functions
- Plan for actionable intelligence that can drive concrete business decisions
- Balance comprehensive analysis with focused insights relevant to enterprise objectives
- Integrate competitive intelligence, market analysis, and internal performance data
- Design for both immediate tactical insights and longer-term strategic understanding
"""

        response = self.llm_adapter.generate_text(
            planning_prompt,
            model=self.model,
            log_name="query_planner",
        )
        try:
            return extract_json(response)
        except:
            # Fallback to basic structure if JSON parsing fails
            return {
                "research_investigation_areas": [{
                    "area_id": 1,
                    "research_focus": question,
                    "information_needs": ["Comprehensive analysis of the research question"],
                    "knowledge_sources": ["both"],
                    "research_approach": "strategic_assessment",
                    "key_concepts": [],
                    "business_rationale": "Primary research objective",
                    "expected_insights": "Core findings and strategic recommendations",
                    "stakeholder_impact": "All relevant business stakeholders",
                    "importance_level": "critical"
                }],
                "research_methodology": {
                    "overall_approach": "Comprehensive strategic analysis",
                    "competitive_positioning": "TBD based on findings",
                    "knowledge_synthesis": "Integrated analysis approach",
                    "internal_leverage": "Utilize all available internal data sources",
                    "external_validation": "Cross-reference with market intelligence"
                }
            }
