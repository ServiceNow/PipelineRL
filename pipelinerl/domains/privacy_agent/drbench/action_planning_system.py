import datetime
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import RunConfig

from .agent_tools.base import ResearchContext
from .utils import extract_json

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionType(Enum):
    WEB_SEARCH = "web_search"
    ENTERPRISE_API = "enterprise_api"
    MCP_QUERY = "mcp_query"
    URL_FETCH = "url_fetch"
    FILE_DOWNLOAD = "file_download"
    DATA_ANALYSIS = "data_analysis"
    CONTEXT_SYNTHESIS = "context_synthesis"
    LOCAL_DOCUMENT_SEARCH = "local_document_search"
    LOCAL_FILE_ANALYSIS = "local_file_analysis"


# Map common LLM/tool aliases to valid ActionType values.
_ACTION_TYPE_ALIASES = {
    "internet_search": "web_search",
    "web": "web_search",
    "web_search": "web_search",
    "browsecomp_search": "web_search",
    "analyze": "data_analysis",
    "enhanced_url_fetch": "url_fetch",
    "url_fetch": "url_fetch",
    "file_download": "file_download",
    "local_file_search": "local_document_search",
    "local_document_search": "local_document_search",
    "local_file_analysis": "local_file_analysis",
    "smart_analysis": "data_analysis",
    "analysis": "data_analysis",
    "data_analysis": "data_analysis",
    "context_synthesis": "context_synthesis",
    "enterprise_api": "enterprise_api",
    "mcp_query": "mcp_query",
}


def _coerce_action_type(raw_type: Any) -> ActionType:
    """Normalize LLM/tool action types to valid ActionType values."""
    if raw_type is None:
        normalized = "web_search"
    else:
        normalized = str(raw_type).strip().lower()
    normalized = _ACTION_TYPE_ALIASES.get(normalized, normalized)
    return ActionType(normalized)


@dataclass
class Action:
    """Represents a single executable action"""

    id: str
    type: ActionType
    description: str
    parameters: Dict[str, Any]
    status: ActionStatus = ActionStatus.PENDING
    priority: float = 0.5  # 0.0 to 1.0
    dependencies: List[str] = field(default_factory=list)  # Action IDs this depends on
    expected_output: str = ""
    actual_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    completion_time: Optional[str] = None  # ISO timestamp when action completed
    iteration_completed: Optional[int] = None  # Which iteration this action was completed in
    created_from_research_step: Optional[str] = None
    preferred_tool: Optional[str] = None  # Explicit tool preference
    score: float = 0.0  # Information gain score

    def to_dict(self) -> Dict[str, Any]:
        """Convert Action to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "parameters": self.parameters,
            "status": self.status.value,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "error": self.error,
            "execution_time": self.execution_time,
            "completion_time": self.completion_time,
            "iteration_completed": self.iteration_completed,
            "created_from_research_step": self.created_from_research_step,
            "preferred_tool": self.preferred_tool,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create Action from dictionary"""
        return cls(
            id=data["id"],
            type=_coerce_action_type(data.get("type")),
            description=data["description"],
            parameters=data["parameters"],
            status=ActionStatus(data["status"]),
            priority=data["priority"],
            dependencies=data["dependencies"],
            expected_output=data["expected_output"],
            actual_output=data.get("actual_output"),
            error=data.get("error"),
            execution_time=data.get("execution_time"),
            completion_time=data.get("completion_time"),
            iteration_completed=data.get("iteration_completed"),
            created_from_research_step=data.get("created_from_research_step"),
            preferred_tool=data.get("preferred_tool"),
            score=data.get("score", 0.0),
        )


@dataclass
class ActionPlan:
    """Represents a complete action plan"""

    id: str
    research_query: str
    actions: List[Action] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 10
    completed_actions: int = 0
    failed_actions: int = 0

    def add_action(self, action: Action):
        """Add an action to the plan"""
        self.actions.append(action)

    def get_next_actions(self, max_concurrent: int = 3) -> List[Action]:
        """Get the next actions that can be executed with enhanced scoring"""
        ready_actions = []

        for action in self.actions:
            if action.status == ActionStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_met = all(
                    (
                        self.get_action_by_id(dep_id) is None
                        or self.get_action_by_id(dep_id).status == ActionStatus.COMPLETED
                    )
                    for dep_id in action.dependencies
                )

                if dependencies_met:
                    ready_actions.append(action)

        # Apply information gain scoring
        for action in ready_actions:
            action.score = self._calculate_action_score(action)

        # Sort by score (highest first) and limit to max_concurrent
        ready_actions.sort(key=lambda a: a.score, reverse=True)
        return ready_actions[:max_concurrent]

    def get_action_by_id(self, action_id: str) -> Optional[Action]:
        """Get an action by its ID"""
        for action in self.actions:
            if action.id == action_id:
                return action
        return None

    def mark_completed(
        self,
        action_id: str,
        output: Dict[str, Any],
        execution_time: Optional[float] = None,
        iteration: Optional[int] = None,
    ):
        """Mark an action as completed with timing information"""
        action = self.get_action_by_id(action_id)
        if action:
            action.status = ActionStatus.COMPLETED
            action.actual_output = output
            action.completion_time = datetime.datetime.now().isoformat()
            if execution_time is not None:
                action.execution_time = execution_time
            if iteration is not None:
                action.iteration_completed = iteration
            self.completed_actions += 1

    def mark_failed(
        self, action_id: str, error: str, execution_time: Optional[float] = None, iteration: Optional[int] = None
    ):
        """Mark an action as failed with timing information"""
        action = self.get_action_by_id(action_id)
        if action:
            action.status = ActionStatus.FAILED
            action.error = error
            action.completion_time = datetime.datetime.now().isoformat()
            if execution_time is not None:
                action.execution_time = execution_time
            if iteration is not None:
                action.iteration_completed = iteration
            self.failed_actions += 1

    def is_complete(self) -> bool:
        """Check if the action plan is complete"""
        pending_actions = [a for a in self.actions if a.status == ActionStatus.PENDING]
        in_progress_actions = [a for a in self.actions if a.status == ActionStatus.IN_PROGRESS]
        return len(pending_actions) == 0 and len(in_progress_actions) == 0

    def _calculate_action_score(self, action: Action) -> float:
        """Calculate information gain score for an action"""
        # Base score from priority
        score = action.priority

        # Novelty score - prefer actions that explore new areas
        novelty_score = self._calculate_novelty_score(action)
        score *= 1.0 + novelty_score * 0.5  # Up to 50% boost for novelty

        # Source diversity score - balance internal and external sources
        diversity_score = self._calculate_source_diversity_score(action)
        score *= 1.0 + diversity_score * 0.3  # Up to 30% boost for diversity

        # Recency penalty - slightly prefer newer actions
        action_age = self.actions.index(action) if action in self.actions else 0
        recency_factor = 1.0 - (action_age / max(len(self.actions), 1)) * 0.1  # Max 10% penalty
        score *= recency_factor

        return score

    def _calculate_novelty_score(self, action: Action) -> float:
        """Calculate how novel/unique this action is compared to completed actions"""
        if not self.actions:
            return 1.0

        completed_actions = [a for a in self.actions if a.status == ActionStatus.COMPLETED]
        if not completed_actions:
            return 1.0

        # Check similarity with completed actions
        similarities = []
        for completed in completed_actions:
            if completed.type == action.type:
                # Simple query similarity for searches
                if action.type in [ActionType.WEB_SEARCH, ActionType.ENTERPRISE_API]:
                    query1 = action.parameters.get("query", "").lower()
                    query2 = completed.parameters.get("query", "").lower()
                    if query1 and query2:
                        words1 = set(query1.split())
                        words2 = set(query2.split())
                        if words1 and words2:
                            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                            similarities.append(overlap)

        # Return novelty score (1 - max similarity)
        return 1.0 - max(similarities) if similarities else 1.0

    def _calculate_source_diversity_score(self, action: Action) -> float:
        """Calculate source diversity to balance internal and external sources"""
        completed_actions = [a for a in self.actions if a.status == ActionStatus.COMPLETED]

        if not completed_actions:
            # Prefer enterprise sources initially
            return 1.0 if action.type in [ActionType.ENTERPRISE_API, ActionType.MCP_QUERY] else 0.5

        # Count source types in completed actions
        internal_count = sum(
            1 for a in completed_actions if a.type in [ActionType.ENTERPRISE_API, ActionType.MCP_QUERY]
        )
        external_count = sum(1 for a in completed_actions if a.type in [ActionType.WEB_SEARCH, ActionType.URL_FETCH])

        total = internal_count + external_count
        if total == 0:
            return 0.5

        # Calculate imbalance
        internal_ratio = internal_count / total

        # Prefer the underrepresented source type
        if action.type in [ActionType.ENTERPRISE_API, ActionType.MCP_QUERY]:
            # Internal action - score higher if we have fewer internal sources
            return 1.0 - internal_ratio
        else:
            # External action - score higher if we have more internal sources
            return internal_ratio

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "total_actions": len(self.actions),
            "completed": self.completed_actions,
            "failed": self.failed_actions,
            "pending": len([a for a in self.actions if a.status == ActionStatus.PENDING]),
            "in_progress": len([a for a in self.actions if a.status == ActionStatus.IN_PROGRESS]),
            "current_iteration": self.current_iteration,
            "success_rate": self.completed_actions / len(self.actions) if self.actions else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert ActionPlan to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "research_query": self.research_query,
            "actions": [action.to_dict() for action in self.actions],
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "completed_actions": self.completed_actions,
            "failed_actions": self.failed_actions,
            "stats": self.get_stats(),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionPlan":
        """Create ActionPlan from dictionary"""
        plan = cls(
            id=data["id"],
            research_query=data["research_query"],
            current_iteration=data["current_iteration"],
            max_iterations=data["max_iterations"],
            completed_actions=data["completed_actions"],
            failed_actions=data["failed_actions"],
        )
        plan.actions = [Action.from_dict(action_data) for action_data in data["actions"]]
        return plan

    def save_to_file(self, file_path: str) -> None:
        """Save ActionPlan to JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str) -> "ActionPlan":
        """Load ActionPlan from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _get_action_plan_guidelines(available_tool_names, report_style: str):
    """Generate action plan guidelines based on available tools"""

    if report_style == "concise_qa":
        return _get_qa_action_plan_guidelines(available_tool_names)

    # Build priority sources dynamically
    priority_sources = []
    start_instructions = []
    examples = []

    # Check for enterprise tools
    has_enterprise = any("EnterpriseAPITool" in tool_name for tool_name in available_tool_names)
    if has_enterprise:
        priority_sources.append(
            """
       - Use ENTERPRISE_API actions to search internal systems (Mattermost, Nextcloud, etc.)
       - Look for internal documents, enterprise chat messages, and company files
       - Priority should be 0.8-1.0 for these actions"""
        )
        start_instructions.append("internal/enterprise actions (ENTERPRISE_API, MCP_QUERY)")
        examples.append(
            """{
        "type": "enterprise_api",
        "description": "Search internal systems for relevant documents and discussions",
        "parameters": {"query": "specific internal search terms", "service": "auto"},
        "priority": 0.9,
        "expected_output": "Internal documents, chat messages, and enterprise data",
        "preferred_tool": "EnterpriseAPITool"
    }"""
        )

    # Check for local document tools
    has_local_docs = any("LocalFileSearchTool" in tool_name for tool_name in available_tool_names)
    if has_local_docs:
        priority_sources.append(
            """
       - Use LOCAL_DOCUMENT_SEARCH actions to search provided local documents
       - Look for relevant information in uploaded files and documents
       - Priority should be 0.8-1.0 for these actions"""
        )
        start_instructions.append("local document actions (LOCAL_DOCUMENT_SEARCH, LOCAL_FILE_ANALYSIS)")
        examples.append(
            """{
        "type": "local_document_search",
        "description": "Search local documents for relevant information",
        "parameters": {"query": "specific search terms for local docs", "top_k": 10},
        "priority": 0.9,
        "expected_output": "Relevant information from local document collection",
        "preferred_tool": "LocalFileSearchTool"
    }"""
        )

    # Always include web search
    examples.append(
        """{
        "type": "web_search",
        "description": "Search external sources for industry context and validation",
        "parameters": {"query": "industry trends validation terms", "num_results": 10},
        "priority": 0.6,
        "expected_output": "External articles and industry data for context",
        "preferred_tool": "InternetSearchTool"
    }"""
    )

    # Build the complete guidelines
    first_priority = (
        "**FIRST PRIORITY**: Internal/Enterprise Sources"
        if (has_enterprise or has_local_docs)
        else "**FIRST PRIORITY**: External Sources"
    )
    priority_content = (
        "".join(priority_sources)
        if priority_sources
        else """
       - Use WEB_SEARCH actions to find relevant information
       - Priority should be 0.7-1.0 for these actions"""
    )

    start_instruction = (
        f"START with {', '.join(start_instructions)}" if start_instructions else "START with web search actions"
    )

    return f"""SOURCE PRIORITIZATION STRATEGY (CRITICAL):
    1. {first_priority}{priority_content}

    2. **SECOND PRIORITY**: External Validation
       - Use WEB_SEARCH actions to find public information that supports/validates internal findings
       - Search for industry trends, best practices, external validation
       - Priority should be 0.5-0.7 for these actions

    3. **COMPLEMENT STRATEGY**:
       - Internal sources provide proprietary insights and current state
       - External sources provide industry context and validation
       - Both are needed for comprehensive analysis

    For each action, specify:
    1. Action type ({', '.join([action_type.value for action_type in ActionType])})
    2. Specific parameters needed
    3. Priority (0.0 to 1.0) - HIGHER for internal/enterprise sources
    4. Expected output description
    5. Preferred tool (choose from available tools above)

    Requirements:
    - {start_instruction}
    - FOLLOW with external validation actions (WEB_SEARCH, URL_FETCH)
    - Actions should be executable by the agent
    - Action types should be one of: {', '.join([action_type.value for action_type in ActionType])}
    - Actions should be independent and executable in parallel when possible
    - Don't make up websites or urls
    - If you need to search use web_search
    - If you need to learn about available APIs use enterprise_api, mcp_query, or search for information on the internet
    - Make sure that each action has a CLEAR, DESCRIPTIVE description and expected output
    - Descriptions should be specific (e.g., "Search internal Mattermost for customer feedback on X feature" not "Search internal systems")
    - ALWAYS prioritize enterprise tools for internal data and use internet tools as supporting context

    Return a JSON array of actions (NO dependencies field needed):
    [
    {','.join(examples)}
    ]

    Just return valid JSON, no other text.
    """  # noqa: E501


def _get_qa_action_plan_guidelines(available_tool_names):
    """QA-mode action plan guidelines: lightweight multihop research guidance."""

    has_local_docs = any("LocalFileSearchTool" in tool_name for tool_name in available_tool_names)
    has_web = any("InternetSearchTool" in t or "BrowseCompSearchTool" in t for t in available_tool_names)
    has_analysis = any("SmartAnalysisTool" in t for t in available_tool_names)

    allowed_action_types = []
    if has_web:
        allowed_action_types.append(ActionType.WEB_SEARCH)
    if has_local_docs:
        allowed_action_types.append(ActionType.LOCAL_DOCUMENT_SEARCH)
    if has_analysis:
        allowed_action_types.append(ActionType.DATA_ANALYSIS)
    if not allowed_action_types:
        allowed_action_types.append(ActionType.WEB_SEARCH)

    # Build tool-aware examples
    examples = []
    if has_local_docs:
        examples.append("""{
        "type": "local_document_search",
        "description": "Search local documents for [specific query]",
        "parameters": {"query": "specific search terms", "top_k": 10},
        "priority": 0.8,
        "expected_output": "Relevant evidence from local documents",
        "preferred_tool": "LocalFileSearchTool"
    }""")
    if has_web:
        web_tool = "BrowseCompSearchTool" if any("BrowseCompSearchTool" in t for t in available_tool_names) else "InternetSearchTool"
        examples.append(f"""{{
        "type": "web_search",
        "description": "Search web for [specific query]",
        "parameters": {{"query": "specific search terms", "num_results": 10}},
        "priority": 0.8,
        "expected_output": "Relevant evidence from web sources",
        "preferred_tool": "{web_tool}"
    }}""")
    if has_analysis:
        examples.append("""{
        "type": "data_analysis",
        "description": "Analyze the collected evidence to resolve a specific uncertainty",
        "parameters": {"query": "specific analytical question to answer from collected evidence"},
        "priority": 0.6,
        "expected_output": "A synthesis of the most relevant evidence and remaining gaps",
        "preferred_tool": "SmartAnalysisTool"
    }""")

    action_types = ", ".join(action_type.value for action_type in allowed_action_types)

    return f"""Generate a small set of useful research actions for this investigation area.

    These numbered questions are part of one larger multihop task. Choose actions that seem most useful for improving the final answer.

    Guidance:
    - One action may help answer more than one numbered question
    - Prefer specific, high-signal search queries over broad exploratory ones
    - Prefer retrieval actions first; use data_analysis only when synthesizing collected evidence would clarify the answer

    For each action, specify:
    1. Action type ({action_types})
    2. Specific parameters needed
    3. Priority (0.0 to 1.0)
    4. Expected output description
    5. Preferred tool (choose from available tools)

    Requirements:
    - Generate only the actions that seem worthwhile for this investigation area
    - Search queries should be specific and targeted
    - Action types should be one of: {action_types}
    - Do NOT generate URL_FETCH, FILE_DOWNLOAD, ENTERPRISE_API, MCP_QUERY, or CONTEXT_SYNTHESIS actions
    - Do not invent unsupported actions or tools
    - Action types should be one of: {action_types}

    Return a JSON array of actions:
    [
    {','.join(examples)}
    ]

    Just return valid JSON, no other text.
    """


class ActionPlanner:
    """Plans and manages action execution separate from research planning"""

    def __init__(self, model: str, llm_adapter: Any, run_config: RunConfig):
        self.model = model
        self.llm_adapter = llm_adapter
        self.run_config = run_config
        self.active_plans: Dict[str, ActionPlan] = {}

    def create_action_plan(
        self, research_plan: Optional[Dict[str, Any]], context: ResearchContext, tool_registry
    ) -> ActionPlan:
        """Create an action plan from a research plan"""

        plan_id = f"action_plan_{hash(context.original_question)}_{len(self.active_plans)}"
        action_plan = ActionPlan(id=plan_id, research_query=context.original_question)

        # Extract actionable tasks from research plan (if available)
        actionable_tasks = research_plan.get("research_investigation_areas", []) if research_plan else []

        # Generate actions for each task (or fallback actions if no research plan)
        all_actions = []
        if actionable_tasks:
            for i, task in enumerate(actionable_tasks):
                actions = self._generate_actions_for_task(task, tool_registry, context)
                for action in actions:
                    # Use consistent task ID extraction logic
                    action.created_from_research_step = self._extract_task_id(task)
                    all_actions.append(action)
        else:
            # Fallback: generate basic actions directly from the query
            fallback_actions = self._generate_fallback_actions(context, tool_registry)
            all_actions.extend(fallback_actions)

        # Deduplicate actions before adding to plan
        unique_actions = self._deduplicate_actions(all_actions, [])
        for action in unique_actions:
            action_plan.add_action(action)

        self.active_plans[plan_id] = action_plan
        return action_plan

    def evolve_action_plan(
        self, plan_id: str, new_findings: Dict[str, Any], context: ResearchContext, tool_registry
    ) -> List[Action]:
        """Evolve the action plan based on new findings"""

        action_plan = self.active_plans.get(plan_id)
        if not action_plan:
            return []

        # Use LLM to analyze findings and suggest new actions
        new_actions = self._generate_adaptive_actions(action_plan, new_findings, context, tool_registry)

        # Deduplicate actions before adding them
        deduplicated_actions = self._deduplicate_actions(new_actions, action_plan.actions)

        for action in deduplicated_actions:
            action_plan.add_action(action)

        return deduplicated_actions

    def _generate_tool_guidelines(self, tool_registry) -> str:
        """Generate dynamic tool selection guidelines from the tool registry"""
        guidelines = []
        for tool in tool_registry.tools:
            tool_name = tool.__class__.__name__
            try:
                purpose = tool.purpose
                guidelines.append(f'    - Use "{tool_name}" for {purpose}')
            except (AttributeError, NotImplementedError):
                # Handle tools that don't have purpose implemented yet
                guidelines.append(f'    - "{tool_name}" is available')

        return "\n".join(guidelines) if guidelines else "    - Use appropriate tools based on task requirements"

    def _log_prompt(self, prompt: str, filename: str) -> None:
        """Log prompts to run_dir/prompts when enabled."""
        if not self.run_config.log_prompts:
            return
        if not self.run_config.run_dir:
            raise ValueError("--run-dir is required for prompt logging.")
        prompt_dir = Path(self.run_config.run_dir) / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / filename).write_text(prompt, encoding="utf-8")

    def _extract_task_id(self, task: Dict[str, Any]) -> str:
        """Extract task identifier from research investigation area"""
        # Current research plan format uses area_id
        area_id = task.get("area_id")
        if area_id:
            return str(area_id)

        # Fallback: create meaningful ID from research_focus
        focus = task.get("research_focus", "")
        if focus:
            # Clean the focus text to create a valid identifier
            focus = re.sub(r"[^a-zA-Z0-9\s]", "", focus)  # Remove special chars
            focus = focus.replace(" ", "_").lower()[:20]  # Replace spaces, lowercase, limit length
            return f"task_{focus}"

        return "unknown"

    def _normalize_action_configs(self, raw_configs: Any, fallback_query: str) -> List[Dict[str, Any]]:
        """Coerce common LLM JSON shapes into a list of action config dicts."""
        if isinstance(raw_configs, dict):
            for key in ("actions", "new_actions", "suggested_actions"):
                value = raw_configs.get(key)
                if isinstance(value, list):
                    raw_configs = value
                    break
            else:
                raw_configs = [raw_configs]

        if not isinstance(raw_configs, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in raw_configs:
            if isinstance(item, dict):
                normalized.append(item)
                continue

            if isinstance(item, str):
                query = item.strip()
                if not query:
                    continue
                normalized.append(
                    {
                        "type": "web_search",
                        "description": query,
                        "parameters": {"query": query or fallback_query},
                        "priority": 0.5,
                        "expected_output": "Relevant information for the research question",
                    }
                )

        return normalized

    def _allowed_action_types(self, tool_registry) -> set[ActionType]:
        """Return the action types that are actually supported in the current mode."""
        if self.run_config.report_style != "concise_qa":
            return set(ActionType)

        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]
        allowed: set[ActionType] = set()

        if any("InternetSearchTool" in t or "BrowseCompSearchTool" in t for t in available_tool_names):
            allowed.add(ActionType.WEB_SEARCH)
        if any("LocalFileSearchTool" in t for t in available_tool_names):
            allowed.add(ActionType.LOCAL_DOCUMENT_SEARCH)
        if any("SmartAnalysisTool" in t for t in available_tool_names):
            allowed.add(ActionType.DATA_ANALYSIS)

        if not allowed:
            allowed.add(ActionType.WEB_SEARCH)

        return allowed

    def _filter_supported_actions(self, actions: List[Action], tool_registry) -> List[Action]:
        """Drop actions that are not supported in the current run mode."""
        allowed_action_types = self._allowed_action_types(tool_registry)
        filtered: List[Action] = []

        for action in actions:
            if action.type not in allowed_action_types:
                logger.info(
                    "Dropping unsupported action for mode %s: %s (%s)",
                    self.run_config.report_style,
                    action.type.value,
                    action.description[:120],
                )
                continue
            filtered.append(action)

        return filtered

    def _generate_actions_for_task(self, task: Dict[str, Any], tool_registry, context: ResearchContext) -> List[Action]:
        """Generate specific actions for a research task"""

        # Extract fields from new research plan structure
        research_focus = task.get("research_focus", "")
        information_needs = task.get("information_needs", [])
        knowledge_sources = task.get("knowledge_sources", ["both"])
        research_approach = task.get("research_approach", "strategic_assessment")
        key_concepts = task.get("key_concepts", [])

        # Get meaningful task identifier - prioritize task_id, then area_id, then create descriptive fallback
        task_id = self._extract_task_id(task)

        # Generate dynamic tool guidelines from registry
        tool_guidelines = self._generate_tool_guidelines(tool_registry)
        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]
        task_context_block = context.get_task_context_block()

        task_prompt_header = "Generate specific executable actions for this research investigation area:"
        if self.run_config.report_style != "concise_qa":
            task_prompt_header = "Generate specific executable actions for this research investigation area with SOURCE PRIORITIZATION:"

        prompt = f"""
    {task_prompt_header}

    Research Focus: {research_focus}
    Information Needs: {information_needs}
    Knowledge Sources: {knowledge_sources}
    Research Approach: {research_approach}
    Key Concepts: {key_concepts}
    {task_context_block}
    Available Tools: {available_tool_names}

    Tool Selection Guidelines:
{tool_guidelines}

{_get_action_plan_guidelines(available_tool_names, self.run_config.report_style)}
    """

        self._log_prompt(prompt, f"action_prompt_iter{getattr(context, 'iteration', 0)}.txt")

        try:
            action_configs = prompt_llm_and_parse_json(
                self.llm_adapter,
                self.model,
                prompt,
                log_name="action_plan_initial",
            )
            action_configs = self._normalize_action_configs(action_configs, fallback_query=research_focus)

            actions = []
            for i, config in enumerate(action_configs):
                try:
                    action = Action(
                        id=f"{task_id}_action_{i}",
                        type=_coerce_action_type(config.get("type", "web_search")),
                        description=config.get("description", ""),
                        parameters=config.get("parameters", {}),
                        priority=config.get("priority", 0.5),
                        expected_output=config.get("expected_output", ""),
                        dependencies=[],  # Start empty
                        preferred_tool=config.get("preferred_tool"),
                    )
                    actions.append(action)
                except Exception as e:
                    logger.warning(f"Error parsing action config. Skipping action: {e}")
                    continue

            actions = self._filter_supported_actions(actions, tool_registry)

            # Skip dependency detection in QA mode to keep the action graph simple.
            if len(actions) > 1 and self.run_config.report_style != "concise_qa":
                detected_deps = self._detect_action_dependencies(actions, research_focus)

                # Apply the detected dependencies
                for action in actions:
                    if action.id in detected_deps:
                        action.dependencies = detected_deps[action.id]

            return actions

        except Exception as e:
            logger.error(f"Error generating actions for task: {e}")
            raise

    def _fallback_task_actions(self, task: Dict[str, Any], tool_registry) -> List[Action]:
        """Fallback action generation when LLM fails"""

        research_focus = task.get("research_focus", "")
        task_id = self._extract_task_id(task)
        available_tool_names = [tool.__class__.__name__.lower() for tool in tool_registry.tools]

        actions = []

        # Prioritize local document search if available
        if any("localfilesearchtool" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id=f"{task_id}_local_doc_search",
                    type=ActionType.LOCAL_DOCUMENT_SEARCH,
                    description=f"Search local documents for: {research_focus}",
                    parameters={"query": research_focus, "top_k": 10},
                    priority=0.9,  # High priority for local authoritative content
                    expected_output="Relevant information from local document collection",
                    preferred_tool="LocalFileSearchTool",
                )
            )

        # Always start with web search
        if any("search" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id=f"{task_id}_web_search",
                    type=ActionType.WEB_SEARCH,
                    description=f"Search the web for: {research_focus}",
                    parameters={"query": research_focus, "num_results": 10},
                    priority=0.7,
                    expected_output="Relevant web search results",
                    preferred_tool="InternetSearchTool",
                )
            )

        # Add enterprise API search if available
        if self.run_config.report_style != "concise_qa" and any("enterprise" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id=f"{task_id}_enterprise_search",
                    type=ActionType.ENTERPRISE_API,
                    description=f"Search enterprise services for: {research_focus}",
                    parameters={"query": research_focus},
                    priority=0.8,
                    expected_output="Enterprise data related to query",
                    preferred_tool="EnterpriseAPITool",
                )
            )

        return self._filter_supported_actions(actions, tool_registry)

    def _generate_fallback_actions(self, context: ResearchContext, tool_registry) -> List[Action]:
        """Generate basic actions when no research plan is available"""

        query = context.original_question
        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]

        # Generate tool guidelines
        tool_guidelines = self._generate_tool_guidelines(tool_registry)
        task_context_block = context.get_task_context_block()

        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]

        task_prompt_header = "Generate specific executable actions for this research task:"
        if self.run_config.report_style != "concise_qa":
            task_prompt_header = "Generate specific executable actions for this research task with SOURCE PRIORITIZATION:"

        prompt = f"""
    {task_prompt_header}

    Task: {query}
    {task_context_block}
    Available Tools: {available_tool_names}

    Tool Selection Guidelines:
{tool_guidelines}

{_get_action_plan_guidelines(available_tool_names, self.run_config.report_style)}
        """

        self._log_prompt(prompt, f"fallback_prompt_iter{getattr(context, 'iteration', 0)}.txt")

        try:
            action_configs = prompt_llm_and_parse_json(
                self.llm_adapter,
                self.model,
                prompt,
                log_name="action_plan_fallback",
            )
            action_configs = self._normalize_action_configs(action_configs, fallback_query=query)

            actions = []
            for i, config in enumerate(action_configs):
                try:
                    action = Action(
                        id=f"fallback_action_{i}",
                        type=_coerce_action_type(config.get("type", "web_search")),
                        description=config.get("description", ""),
                        parameters=config.get("parameters", {}),
                        priority=config.get("priority", 0.5),
                        expected_output=config.get("expected_output", ""),
                        dependencies=[],
                        preferred_tool=config.get("preferred_tool"),
                        created_from_research_step="fallback_generation",
                    )
                    actions.append(action)
                except Exception as e:
                    logger.warning(f"Error parsing fallback action config: {e}")
                    continue

            return self._filter_supported_actions(actions, tool_registry)

        except Exception as e:
            logger.error(f"Error generating fallback actions: {e}")
            # Return basic fallback actions
            return self._get_basic_fallback_actions(query, tool_registry)

    def _get_basic_fallback_actions(self, query: str, tool_registry) -> List[Action]:
        """Generate very basic fallback actions when LLM fails"""

        available_tool_names = [tool.__class__.__name__.lower() for tool in tool_registry.tools]
        actions = []

        # Always try web search if available
        if any("search" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id="basic_web_search",
                    type=ActionType.WEB_SEARCH,
                    description=f"Search the web for: {query}",
                    parameters={"query": query, "num_results": 10},
                    priority=0.7,
                    expected_output="Web search results",
                    preferred_tool="InternetSearchTool",
                    created_from_research_step="basic_fallback",
                )
            )

        # Try enterprise search if available
        if self.run_config.report_style != "concise_qa" and any("enterprise" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id="basic_enterprise_search",
                    type=ActionType.ENTERPRISE_API,
                    description=f"Search enterprise systems for: {query}",
                    parameters={"query": query},
                    priority=0.8,
                    expected_output="Enterprise search results",
                    preferred_tool="EnterpriseAPITool",
                    created_from_research_step="basic_fallback",
                )
            )

        return self._filter_supported_actions(actions, tool_registry)

    def _generate_adaptive_actions(
        self,
        action_plan: ActionPlan,
        new_findings: Dict[str, Any],
        context: ResearchContext,
        tool_registry,
    ) -> List[Action]:
        """Generate new actions based on findings from completed actions"""

        completed_actions = [a for a in action_plan.actions if a.status == ActionStatus.COMPLETED]

        try:
            findings_json = json.dumps(new_findings, indent=2)
        except TypeError:
            # Fallback: try to convert non-serializable objects to string, or omit them
            def safe_serialize(obj):
                try:
                    json.dumps(obj)
                    return obj
                except TypeError:
                    return str(obj)

            if isinstance(new_findings, dict):
                findings_json = json.dumps({k: safe_serialize(v) for k, v in new_findings.items()}, indent=2)
            else:
                findings_json = str(new_findings)

        if self.run_config.report_style == "concise_qa":
            prompt = self._build_qa_adaptive_prompt(
                action_plan,
                completed_actions,
                findings_json,
                tool_registry,
                context,
            )
        else:
            prompt = self._build_report_adaptive_prompt(
                action_plan, completed_actions, findings_json, new_findings, tool_registry
            )


        self._log_prompt(prompt, f"adaptive_prompt_iter{action_plan.current_iteration}.txt")

        try:
            action_configs = prompt_llm_and_parse_json(
                self.llm_adapter,
                self.model,
                prompt,
                log_name=f"adaptive_iter{action_plan.current_iteration}",
            )
            action_configs = self._normalize_action_configs(
                action_configs,
                fallback_query=action_plan.research_query,
            )

            new_actions = []
            for i, config in enumerate(action_configs):
                # Create more descriptive adaptive action ID
                action_type = config.get("type", "web_search")
                desc_snippet = config.get("description", "")[:15].replace(" ", "_").lower()
                action_id = (
                    f"adaptive_{action_plan.current_iteration}_{action_type}_{i}"
                    if not desc_snippet
                    else f"adaptive_{action_plan.current_iteration}_{action_type}_{desc_snippet}_{i}"
                )
                try:
                    action = Action(
                        id=action_id,
                        type=_coerce_action_type(config.get("type", "web_search")),
                        description=config.get("description", ""),
                        parameters=config.get("parameters", {}),
                        priority=config.get("priority", 0.5),
                        expected_output=config.get("expected_output", ""),
                        dependencies=config.get("dependencies", []),
                        preferred_tool=config.get("preferred_tool"),  # Now includes preferred tool
                    )
                    new_actions.append(action)
                except Exception as e:
                    logger.warning(f"Error parsing adaptive action config. Skipping action: {e}")
                    continue

            return self._filter_supported_actions(new_actions, tool_registry)

        except Exception as e:
            logger.warning(f"Adaptive action generation failed (continuing with existing actions): {e}")
            return []

    def _build_qa_adaptive_prompt(self, action_plan, completed_actions, findings_json, tool_registry, context):
        """Build a simple adaptive prompt for QA mode with lightweight source awareness."""
        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]
        allowed_action_types = sorted(
            action_type.value for action_type in self._allowed_action_types(tool_registry)
        )
        task_context_block = context.get_task_context_block()

        # Lightweight action-type summary instead of full source composition analysis
        type_counts = {}
        for a in completed_actions:
            t = a.type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        action_type_summary = ", ".join(f"{v} {k}" for k, v in type_counts.items()) or "none"

        return f"""You are continuing a multihop research task made up of numbered questions.

Original Questions: {action_plan.research_query}
{task_context_block}
Completed Actions: {len(completed_actions)} ({action_type_summary})
Available Tools: {available_tool_names}
Latest Findings: {findings_json}

Suggest the next 0-3 actions that seem most useful for improving the overall final answer.
- Later questions may depend on earlier findings, but you do not need to work strictly in order
- Prefer actions that resolve an important uncertainty, dependency, or ambiguity
- Avoid repeating prior searches unless you are meaningfully refining them
- Action types should be one of: {', '.join(allowed_action_types)}
- Do NOT generate URL_FETCH, FILE_DOWNLOAD, ENTERPRISE_API, MCP_QUERY, or CONTEXT_SYNTHESIS actions

Return a JSON array of 0-3 new actions, or [] if the current evidence already seems sufficient.
Each action: {{"type": "...", "description": "...", "parameters": {{}}, "priority": 0.8, "expected_output": "...", "preferred_tool": "..."}}

Just return valid JSON, no other text.
"""

    def _build_report_adaptive_prompt(self, action_plan, completed_actions, findings_json, new_findings, tool_registry):
        """Build the full adaptive prompt for report mode with source complementarity."""
        # Analyze findings to understand source composition
        internal_findings = self._analyze_source_composition(new_findings)

        # Generate tool guidelines and available tools for adaptive actions
        tool_guidelines = self._generate_tool_guidelines(tool_registry)
        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]

        return f"""
Based on the research progress so far, suggest new actions with INTELLIGENT SOURCE COMPLEMENTARITY:

Original Research Query: {action_plan.research_query}
Completed Actions: {len(completed_actions)}
Latest Findings: {findings_json}

Available Tools: {available_tool_names}

Tool Selection Guidelines:
{tool_guidelines}

Current Action Plan Status:
{json.dumps(action_plan.get_stats(), indent=2)}

SOURCE ANALYSIS:
- Internal/Enterprise Sources Found: {internal_findings['has_internal']}
- External Sources Found: {internal_findings['has_external']}
- Enterprise Chat/Files: {internal_findings['has_enterprise']}
- Key Internal Insights: {internal_findings['internal_insights']}

ADAPTIVE STRATEGY - Based on what we've found:

If we have STRONG INTERNAL FINDINGS:
- Validate/support with external industry research (priority 0.6-0.7)
- Search for related external trends that confirm internal direction
- Look for case studies and best practices from similar organizations

If we have LIMITED INTERNAL FINDINGS:
- Intensify internal search with different keywords (priority 0.8-0.9)
- Try alternative enterprise tools and search approaches
- Search for internal documentation using related terminology

If we have MIXED FINDINGS:
- Fill gaps between internal and external perspectives
- Look for specific URLs/resources mentioned in findings
- Deep-dive into promising leads with targeted searches

Suggest 0-5 new actions that would:
1. COMPLEMENT existing source types (internal findings -> external validation, limited internal -> more internal search)
2. Follow up on promising leads from the findings
3. Fill gaps in the research methodology
4. Explore new angles discovered in the findings
5. Download or fetch specific resources mentioned
6. Use available tools effectively based on source gaps
7. Avoid repeating previous actions
8. Prioritize based on source completeness and findings quality

For each action, specify:
    1. Action type ({', '.join([action_type.value for action_type in ActionType])})
    2. Specific parameters needed
    3. Priority (0.0 to 1.0) - HIGHER for filling source gaps
    4. Expected output description
    5. Preferred tool (choose from available tools above)
    6. Rationale (why this action complements existing findings)

Return a JSON array of new actions:
[
  {{
    "type": "enterprise_api",
    "description": "Search for additional internal documentation using alternative keywords",
    "parameters": {{"query": "alternative search terms", "service": "auto"}},
    "priority": 0.8,
    "expected_output": "More internal documents and enterprise data",
    "preferred_tool": "EnterpriseAPITool",
    "rationale": "Filling gap in internal sources discovered in analysis"
  }},
  {{
    "type": "web_search",
    "description": "Validate internal findings with external industry research",
    "parameters": {{"query": "industry validation terms", "num_results": 8}},
    "priority": 0.6,
    "expected_output": "External validation data for internal insights",
    "preferred_tool": "InternetSearchTool",
    "rationale": "Supporting strong internal findings with external context"
  }}
]

Create valid JSON only, no other text.
"""

    def _deduplicate_actions(self, new_actions: List[Action], existing_actions: List[Action]) -> List[Action]:
        """Remove duplicate or highly similar actions"""
        unique_actions = []

        for new_action in new_actions:
            is_duplicate = False

            # Check against existing actions
            for existing in existing_actions:
                if self._is_similar_action(new_action, existing):
                    logger.debug(f"Skipping duplicate action: {new_action.description}")
                    is_duplicate = True
                    break

            # Check against already added unique actions
            if not is_duplicate:
                for unique in unique_actions:
                    if self._is_similar_action(new_action, unique):
                        logger.debug(f"Skipping duplicate action: {new_action.description}")
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_actions.append(new_action)

        logger.info(f"Deduplication: {len(new_actions)} actions -> {len(unique_actions)} unique actions")
        return unique_actions

    def _is_similar_action(self, action1: Action, action2: Action) -> bool:
        """Check if two actions are similar enough to be considered duplicates"""
        # Same type and very similar descriptions
        if action1.type != action2.type:
            return False

        # Normalize descriptions for comparison
        desc1 = action1.description.lower().strip()
        desc2 = action2.description.lower().strip()

        # Check for exact match
        if desc1 == desc2:
            return True

        # Check for similar parameters (especially for searches)
        if action1.type in [ActionType.WEB_SEARCH, ActionType.ENTERPRISE_API]:
            query1 = action1.parameters.get("query", "").lower()
            query2 = action2.parameters.get("query", "").lower()

            # Check if queries are very similar
            if query1 and query2:
                # Simple similarity check - can be enhanced with proper string similarity
                words1 = set(query1.split())
                words2 = set(query2.split())

                # If 80% of words overlap, consider it similar
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1.intersection(words2))
                    similarity = overlap / min(len(words1), len(words2))
                    if similarity > 0.8:
                        return True

        # Check for URL fetches with same URLs
        if action1.type == ActionType.URL_FETCH:
            urls1 = set(action1.parameters.get("urls", []))
            urls2 = set(action2.parameters.get("urls", []))
            if urls1 and urls2 and urls1 == urls2:
                return True

        return False

    def _analyze_source_composition(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze findings to understand source type composition"""

        analysis = {
            "has_internal": False,
            "has_external": False,
            "has_enterprise": False,
            "internal_insights": [],
            "external_insights": [],
            "enterprise_insights": [],
            "gaps": [],
        }

        # Analyze findings for source indicators
        for key, finding in findings.items():
            if not isinstance(finding, dict):
                continue

            # Check for internal/enterprise source indicators
            if finding.get("success"):
                # Enterprise API results
                if finding.get("tool") == "enhanced_enterprise_api":
                    analysis["has_enterprise"] = True
                    if finding.get("content_stored_in_vector", 0) > 0:
                        analysis["enterprise_insights"].append(
                            f"Found {finding.get('content_stored_in_vector', 0)} enterprise sources"
                        )

                # File processing results
                if finding.get("processed_files"):
                    analysis["has_internal"] = True
                    analysis["internal_insights"].append(
                        f"Processed {len(finding.get('processed_files', []))} internal files"
                    )

                # Web search results
                if finding.get("tool") == "internet_search":
                    analysis["has_external"] = True
                    if finding.get("urls_processed", 0) > 0:
                        analysis["external_insights"].append(f"Found {finding.get('urls_processed', 0)} web sources")

                # URL fetch results
                if finding.get("url") and finding.get("success"):
                    analysis["has_external"] = True
                    analysis["external_insights"].append(f"Fetched content from {finding.get('url')}")

        # Identify gaps
        if not analysis["has_internal"] and not analysis["has_enterprise"]:
            analysis["gaps"].append("Missing internal/enterprise sources")
        if not analysis["has_external"]:
            analysis["gaps"].append("Missing external validation sources")
        if analysis["has_internal"] and not analysis["has_external"]:
            analysis["gaps"].append("Need external context to validate internal findings")
        if analysis["has_external"] and not analysis["has_internal"]:
            analysis["gaps"].append("Need internal perspective to complement external research")

        return analysis

    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get the status of an action plan"""
        action_plan = self.active_plans.get(plan_id)
        if not action_plan:
            return {"error": "Plan not found"}

        return {
            "plan_id": plan_id,
            "query": action_plan.research_query,
            "stats": action_plan.get_stats(),
            "is_complete": action_plan.is_complete(),
            "actions": [
                {
                    "id": a.id,
                    "type": a.type.value,
                    "description": a.description,
                    "status": a.status.value,
                    "priority": a.priority,
                }
                for a in action_plan.actions
            ],
        }

    def cleanup_completed_plans(self):
        """Remove completed action plans to save memory"""
        completed_plans = [plan_id for plan_id, plan in self.active_plans.items() if plan.is_complete()]

        for plan_id in completed_plans:
            del self.active_plans[plan_id]

        return len(completed_plans)

    def _detect_action_dependencies(self, actions: List[Action], task_context: str = "") -> Dict[str, List[str]]:
        """Use LLM to detect dependencies between actions"""

        if len(actions) <= 1:
            return {}

        # Create a simple summary for the LLM
        actions_info = []
        for action in actions:
            actions_info.append(
                {
                    "id": action.id,
                    "type": action.type.value,
                    "description": action.description,
                    "expected_output": action.expected_output,
                }
            )

        prompt = f"""
    Task Context: {task_context}

    Analyze these actions and determine which ones depend on others:

    {json.dumps(actions_info, indent=2)}

    Return a JSON object where each action ID maps to a list of action IDs it depends on.
    Only create dependencies where an action truly needs the output/result from another action.

    Rules:
    - No circular dependencies
    - No self-dependencies  
    - Only use action IDs from the list above
    - Prefer parallel execution when possible

    Example format:
    {{
    "task_action_0": [],
    "task_action_1": ["task_action_0"],
    "task_action_2": ["task_action_0", "task_action_1"]
    }}

    Return only the JSON:
    """

        try:
            dependencies = prompt_llm_and_parse_json(
                self.llm_adapter,
                self.model,
                prompt,
                log_name="action_dependencies",
            )

            # Validate the dependencies
            valid_action_ids = {action.id for action in actions}
            validated_deps = {}

            for action_id, deps in dependencies.items():
                if action_id in valid_action_ids:
                    # Only keep valid dependencies
                    valid_deps = [dep for dep in deps if dep in valid_action_ids and dep != action_id]
                    validated_deps[action_id] = valid_deps

            return validated_deps

        except Exception as e:
            logger.error(f"Error detecting dependencies: {e}")
            return {}


def prompt_llm_and_parse_json(
    llm_adapter: Any,
    model: str,
    prompt: str,
    retries: int = 3,
    log_name: str = "llm_parse",
):
    """
    Call the rollout LLM and robustly parse a JSON response, handling code blocks and cleaning.
    Retries on JSONDecodeError or ValueError.
    """

    @retry(
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
    )
    def _call():
        response = llm_adapter.generate_text(prompt, model=model, log_name=log_name)
        return extract_json(response)

    return _call()
