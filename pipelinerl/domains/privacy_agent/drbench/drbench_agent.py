import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Optional, Tuple

from .action_planning_system import ActionPlanner, ActionStatus, ActionType
from .agent_call_logger import log_tool_call
from .base_agent import BaseAgent, InsightDict
from .config import RunConfig, SERPER_API_KEY
from .agent_tools import (
    FileManager,
    InternetSearchTool,
    QueryPlanner,
    ResearchContext,
    ToolRegistry,
)
from .agent_tools.analysis_tools import SmartAnalysisTool
from .agent_tools.browsecomp_search_tool import BrowseCompSearchTool
from .agent_tools.content_processor import ContentProcessor
from .agent_tools.enterprise_tools import EnterpriseAPITool
from .agent_tools.local_document_tool import (
    LocalDocumentIngestionTool,
    LocalFileSearchTool,
)
from .agent_tools.model_config import CapacityTier
from .agent_tools.report_tools import ReportAssembler
from .agent_tools.web_tools import EnhancedURLFetchTool
from .session_cache import SessionCache
from .utils import break_report_to_insights
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class DrBenchAgent(BaseAgent):
    """Deep research agent with enhanced action planning and enterprise integration"""

    def __init__(
        self,
        model: str,
        run_config: RunConfig,
        workspace_dir: str = "./outputs/research_workspace",
        max_iterations: int = 10,
        vector_store_base_dir: str = "./outputs/vector_stores",
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        early_stopping: bool = False,
        use_research_plan: bool = True,
        use_adaptive_actions: bool = True,
        local_document_folders: Optional[List[Path | str]] = None,
        local_files: Optional[List[Path | str]] = None,
        local_file_extensions: Optional[List[str]] = None,
        concurrent_actions: int = 1,
        verbose: bool = False,
        shared_browsecomp=None,
        helper_client=None,
        use_remote_browsecomp: bool = False,
        llm_adapter=None,
        vector_store=None,
        session_cache: SessionCache | None = None,
        preindexed_local_documents: bool = False,
        **kwargs,
    ):
        self.run_config = run_config
        self._shared_browsecomp = shared_browsecomp
        self.helper_client = helper_client
        self.use_remote_browsecomp = use_remote_browsecomp
        self.model = model or self.run_config.model
        self.workspace_dir = workspace_dir
        self.max_iterations = max_iterations
        self.concurrent_actions = concurrent_actions
        self.vector_store_base_dir = vector_store_base_dir
        self.embedding_model = embedding_model or self.run_config.get_embedding_model()
        self.embedding_provider = embedding_provider or self.run_config.get_embedding_provider()
        self.early_stopping = early_stopping
        self.use_research_plan = use_research_plan
        self.use_adaptive_actions = use_adaptive_actions
        self.local_document_folders = local_document_folders or []
        self.local_files = local_files or []
        self.local_file_extensions = local_file_extensions
        self.verbose = verbose
        self.llm_adapter = llm_adapter
        self.preindexed_local_documents = preindexed_local_documents

        # Initialize core components
        self.planner = QueryPlanner(
            self.model,
            llm_adapter=self.llm_adapter,
            run_config=self.run_config,
        )
        self.action_planner = ActionPlanner(
            self.model,
            llm_adapter=self.llm_adapter,
            run_config=self.run_config,
        )
        self.tool_registry = ToolRegistry()
        self.file_manager = FileManager(self.workspace_dir)

        # Vector store will be created per research session
        if session_cache is None and vector_store is not None:
            session_cache = getattr(vector_store, "session_cache", None)
        self.session_cache = session_cache
        self.vector_store = vector_store or self._create_vector_store()

        # Store last generated report and metadata
        self._last_report = None
        self._last_report_metadata = {}

        # Initialize content processor with vector store
        self.content_processor = ContentProcessor(
            model=self.model,
            workspace_dir=self.workspace_dir,
            llm_adapter=self.llm_adapter,
            run_config=self.run_config,
            vector_store=self.vector_store,
            session_cache=self.session_cache,
        )

        # Initialize enhanced report assembler
        self.report_assembler = ReportAssembler(
            model=self.model,
            llm_adapter=self.llm_adapter,
            vector_store=self.vector_store,
            report_style=self.run_config.report_style,
        )

        # Register base tools
        self._register_base_tools()

    def _register_base_tools(self):
        """Register base tools that don't depend on environment"""
        tools = []

        # Web search tool: BrowseComp (offline) or InternetSearch (live)
        if self.run_config.browsecomp_enabled:
            logger.info("Using BrowseComp-Plus offline search (replaces InternetSearchTool)")
            if self.use_remote_browsecomp and self.helper_client is not None:
                tools.append(
                    BrowseCompSearchTool(
                        config=self.run_config,
                        vector_store=self.vector_store,
                        device="cpu",
                        helper_client=self.helper_client,
                        use_remote=True,
                    )
                )
            elif self._shared_browsecomp is not None:
                # Shallow copy: shares FAISS/BM25/model, gets own vector_store
                tool = BrowseCompSearchTool.__new__(BrowseCompSearchTool)
                tool.__dict__.update(self._shared_browsecomp.__dict__)
                tool.config = self.run_config
                tool.vector_store = self.vector_store
                tools.append(tool)
            else:
                tools.append(BrowseCompSearchTool(
                    config=self.run_config,
                    vector_store=self.vector_store,
                    device="cpu",
                ))
        elif not self.run_config.no_web and SERPER_API_KEY:
            # Use live Serper web search
            tools.append(InternetSearchTool(
                SERPER_API_KEY,
                run_config=self.run_config,
                vector_store=self.vector_store,
                content_processor=self.content_processor,
            ))

        # URL fetching (only useful with live web)
        if not self.run_config.browsecomp_enabled and not self.run_config.no_web:
            tools.append(EnhancedURLFetchTool(self.content_processor, self.model, llm_adapter=self.llm_adapter))

        # Supporting analysis
        tools.append(SmartAnalysisTool(
            self.model,
            llm_adapter=self.llm_adapter,
            vector_store=self.vector_store,
            capacity_tier=CapacityTier.ULTRA_CAPACITY,
            workspace_dir=self.workspace_dir,
        ))

        for tool in tools:
            self.tool_registry.register_tool(tool)

    def _create_vector_store(self) -> VectorStore:
        """Create an isolated vector store for this research session"""

        # Create unique identifier for this research session
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session_cache = SessionCache(session_id=session_id)

        # Create isolated directory
        isolated_dir = os.path.join(self.vector_store_base_dir, f"session_{timestamp}_{session_id}")

        # Initialize fresh vector store
        vector_store = VectorStore(
            run_config=self.run_config,
            storage_dir=isolated_dir,
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            session_cache=self.session_cache,
            helper_client=self.helper_client,
        )

        if self.verbose:
            logger.info(f"Created isolated vector store: {isolated_dir}")
        return vector_store

    def _register_enterprise_tools(self, env: Any):
        """Register enterprise-specific tools based on available services"""

        # Add enhanced intelligent enterprise API tool
        intelligent_api_tool = EnterpriseAPITool(env, self.content_processor, self.model, self.session_cache)
        self.tool_registry.register_tool(intelligent_api_tool)

    def _ingest_local_documents(self):
        """Ingest local documents into vector store"""

        if not self.local_document_folders and not self.local_files:
            return

        if self.verbose:
            logger.info(
                f"📁 Ingesting local documents from {len(self.local_document_folders)} folders and {len(self.local_files)} files"
            )

        ingestion_tool = LocalDocumentIngestionTool(self.content_processor)

        try:
            stats = ingestion_tool.ingest_paths(
                folder_paths=(self.local_document_folders if self.local_document_folders else None),
                file_paths=self.local_files if self.local_files else None,
                file_extensions=self.local_file_extensions,
                recursive=True,
            )

            if self.verbose:
                logger.info(f"✅ Successfully ingested {stats.processed_files}/{stats.total_files} documents")
                logger.info(f"📊 Processing time: {stats.processing_time_seconds:.2f}s")
                if stats.supported_formats:
                    logger.info(f"📝 File types processed: {', '.join(stats.supported_formats)}")

        except Exception as e:
            logger.error(f"❌ Document ingestion error: {e}")

    def generate_report(
        self,
        query: str,
        env: Optional[Any] = None,
        extract_insights: bool = True,
        results_dir: Optional[str] = None,
        as_dict=True,
        **kwargs,
    ) -> Tuple[str, Optional[List[InsightDict]]] | Dict[str, Any]:
        """
        Generate comprehensive research report with enhanced action planning and enterprise integration

        Args:
            query: Research question to investigate
            env: Enterprise search space providing access to available services
            extract_insights: If True, extract and return insights with citations from report; if False, return None
            **kwargs: Additional arguments

        Returns:
            tuple: (report_text, insights) where insights is a list or a dict if as_dict=True
        """
        if self.verbose:
            logger.info(f"🚀 Starting enhanced research for: {query}")
            logger.debug(f"Enterprise environment: {env}")

        # Step 2: Register enterprise-specific tools
        if env:
            self._register_enterprise_tools(env)

        # Step 3: Ingest local documents if configured
        self.local_document_folders = kwargs.get("local_document_folders", self.local_document_folders)
        self.local_files = kwargs.get("local_files", self.local_files)

        if self.local_document_folders or self.local_files:
            # Add local file search tool
            self.tool_registry.register_tool(
                LocalFileSearchTool(
                    self.vector_store,
                    self.model,
                    llm_adapter=self.llm_adapter,
                    run_config=self.run_config,
                )
            )

            if not self.preindexed_local_documents:
                self._ingest_local_documents()

        # Step 4: Create research plan (if enabled) or proceed directly to action planning
        context = ResearchContext(
            original_question=query,
            task_id=kwargs.get("task_id"),
            company_name=kwargs.get("company_name"),
            company_description=kwargs.get("company_description"),
            vector_store=self.vector_store,
        )

        if self.use_research_plan:
            context.plan = self.planner.create_research_plan(
                query,
                tool_registry=self.tool_registry,
                env=env,
                context=context,
            )
            if self.verbose:
                logger.info(
                    f"📋 Research plan created with {len(context.plan.get('research_investigation_areas', []))} investigation areas"  # noqa: E501
                )
                logger.debug(f"Research plan: {json.dumps(context.plan, indent=2)}")
            # Save research plan to file
            self._save_research_plan(context.plan, query)
        else:
            context.plan = None
            if self.verbose:
                logger.info("📋 Skipping research plan - proceeding directly to action planning")

        # Step 5: Create action plan from research plan (or None if disabled)
        action_plan = self.action_planner.create_action_plan(context.plan, context, self.tool_registry)

        if self.verbose:
            logger.info(f"⚡ Action plan created with {len(action_plan.actions)} initial actions")

        # Save initial action plan to file
        self._save_initial_action_plan(action_plan)

        # Step 6: Execute action plan with iterative evolution
        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.info(f"\n--- Iteration {iteration + 1} ---")

            # Get next actions to execute
            next_actions = action_plan.get_next_actions(max_concurrent=self.concurrent_actions)

            if not next_actions:
                if self.verbose:
                    logger.info("✅ No more actions to execute")
                break

            # Show source prioritization in action
            if self.verbose:
                logger.info(f"🔄 Executing {len(next_actions)} actions (prioritized by source type):")
            for action in next_actions:
                source_type = "🌐 External"
                if action.type in [ActionType.ENTERPRISE_API, ActionType.MCP_QUERY]:
                    source_type = "🏢 Enterprise"
                elif action.type in [
                    ActionType.LOCAL_DOCUMENT_SEARCH,
                    ActionType.LOCAL_FILE_ANALYSIS,
                ]:
                    source_type = "📁 Local Docs"
                elif action.type in [
                    ActionType.CONTEXT_SYNTHESIS,
                    ActionType.DATA_ANALYSIS,
                ]:
                    source_type = "🧪 Analysis"

                logger.info(
                    f"   {source_type} | Priority: {action.priority:.1f} | {action.type.value}: {action.description[:60]}..."  # noqa: E501
                )

            # Execute actions
            iteration_findings = {}
            for action in next_actions:
                action.status = ActionStatus.IN_PROGRESS
                start_time = datetime.datetime.now()

                try:
                    # Execute the action based on its type
                    result = self._execute_action(action, context)

                    # Calculate execution time
                    execution_time = (datetime.datetime.now() - start_time).total_seconds()

                    # Store results with timing information
                    action_plan.mark_completed(
                        action.id,
                        result,
                        execution_time=execution_time,
                        iteration=iteration,
                    )
                    # Use new add_finding method with category based on action type
                    category = action.type.value if hasattr(action, "type") else "general"
                    context.add_finding(f"action_{action.id}", result, category=category)
                    # Ensure result is JSON serializable before storing in iteration_findings
                    try:
                        json.dumps(result)
                        iteration_findings[action.id] = result
                    except TypeError:
                        # If not serializable, convert non-serializable objects to string
                        def make_serializable(obj):
                            if isinstance(obj, dict):
                                return {k: make_serializable(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [make_serializable(i) for i in obj]
                            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                                return obj
                            else:
                                return str(obj)

                        serializable_result = make_serializable(result)
                        iteration_findings[action.id] = serializable_result

                except Exception as e:
                    # Calculate execution time even for failed actions
                    execution_time = (datetime.datetime.now() - start_time).total_seconds()
                    action_plan.mark_failed(
                        action.id,
                        str(e),
                        execution_time=execution_time,
                        iteration=iteration,
                    )
                    context.add_finding(f"error_{action.id}", str(e), category="errors")

            # Step 7: Evolve action plan based on findings (if adaptive actions enabled)
            if iteration_findings and self.use_adaptive_actions:
                # In QA mode, skip source composition analysis (handled by lightweight
                # action-type counts in the adaptive prompt itself)
                if self.run_config.report_style != "concise_qa":
                    if self.verbose:
                        logger.debug("📊 Analyzing source composition for adaptive planning...")
                    source_analysis = self.action_planner._analyze_source_composition(iteration_findings)
                    if self.verbose:
                        logger.debug(
                            f"   Internal/Enterprise: {'✅' if source_analysis['has_internal'] or source_analysis['has_enterprise'] else '❌'}"  # noqa: E501
                        )
                        logger.debug(f"   External Sources: {'✅' if source_analysis['has_external'] else '❌'}")
                        if source_analysis["gaps"]:
                            logger.debug(f"   🎯 Identified gaps: {', '.join(source_analysis['gaps'])}")

                new_actions = self.action_planner.evolve_action_plan(
                    action_plan.id, iteration_findings, context, self.tool_registry
                )
                if new_actions and self.verbose:
                    logger.info(f"🔄 Added {len(new_actions)} new adaptive actions")
                    for action in new_actions:
                        logger.info(
                            f"   {action.type.value} | Priority: {action.priority:.1f} | {action.description[:100]}..."
                        )

            action_plan.current_iteration = iteration + 1
            # Save incrementally so partial results survive crashes
            self._save_final_action_plan(action_plan)

            # Check if we should continue with enhanced completion criteria
            if action_plan.is_complete():
                if self.verbose:
                    logger.info("✅ Action plan completed")
                break

            # Check early stopping criteria (if enabled)
            if self.early_stopping and self._should_stop_research(action_plan, context, iteration + 1):
                if self.verbose:
                    logger.info("🎯 Research goals achieved - stopping early")
                break
        # Step 8: Save final action plan with execution results
        self._save_final_action_plan(action_plan)

        # Step 9: Generate enhanced report
        if self.verbose:
            logger.info("📝 Generating comprehensive report...")
        final_report = self.report_assembler.generate_comprehensive_report(context, action_plan)

        # Step 10: Store the final report in vector store
        report_doc_id = self.vector_store.store_document(
            content=final_report,
            metadata={
                "type": "research_report",
                "question": query,
                "timestamp": datetime.datetime.now().isoformat(),
                "findings_count": len(context.findings),
                "action_plan_stats": action_plan.get_stats(),
                "enterprise_services": (list(env.get_available_apps().keys()) if env else []),
            },
        )

        # Step 11: Save report to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"research_report_{timestamp}.md"
        report_path = self.file_manager.save_file(report_filename, final_report)

        context.files_created.append(report_path)

        # Step 12: Save report to results folder
        if results_dir:
            report_path_results = os.path.join(results_dir, "research_report.md")
            copy2(report_path, report_path_results)

        # Display final statistics
        plan_stats = action_plan.get_stats()
        vector_stats = self.vector_store.get_stats()
        cache_stats = self.session_cache.get_stats()

        if self.verbose:
            logger.info("✅ Research completed!")
            logger.info(f"📊 Action Plan: {plan_stats['completed']}/{plan_stats['total_actions']} actions completed")
            logger.info(f"📚 Knowledge Base: {vector_stats['total_documents']} documents stored")
            if cache_stats["duplicate_preventions"] > 0:
                logger.info(f"🔄 Deduplication: {cache_stats['duplicate_preventions']} duplicates prevented")
            logger.info(f"💾 Report saved: {report_filename}")

        # Store metadata
        self._last_report = final_report
        self._last_report_metadata = {
            "query": query,
            "report_file": report_path,
            "doc_id": report_doc_id,
            "timestamp": timestamp,
            "action_plan_stats": plan_stats,
            "vector_store_path": self.vector_store.storage_dir,
            "enterprise_services": list(env.get_available_apps().keys()) if env else [],
            "files_created": context.files_created,
            # Enhanced metadata from report assembler
            **self.report_assembler.get_evidence_metadata(),
        }
        self._last_context = context

        insights = None
        if extract_insights:
            # Extract insights from the generated report
            try:
                insights = break_report_to_insights(
                    final_report,
                    model=self.model,
                    llm_adapter=self.llm_adapter,
                )
            except Exception as e:
                logger.warning(f"Failed to extract insights from report: {e}")
                insights = []

        if as_dict:
            return {
                "report_text": final_report,
                "report_insights": insights,
            }
        return final_report, insights

    def _save_research_plan(self, research_plan: Optional[Dict[str, Any]], query: str) -> None:
        """Save research plan to JSON file in session directory"""
        if not research_plan:
            return

        try:
            session_dir = self.vector_store.storage_dir
            plan_path = os.path.join(session_dir, "research_plan.json")

            plan_data = {
                "query": query,
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": (self.session_cache.session_id if self.session_cache else None),
                "plan": research_plan,
                "metadata": {
                    "model": self.model,
                    "use_research_plan": self.use_research_plan,
                    "use_adaptive_actions": self.use_adaptive_actions,
                },
            }

            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan_data, f, indent=2, ensure_ascii=False)

            if self.verbose:
                logger.info(f"📋 Research plan saved to: {plan_path}")

        except Exception as e:
            logger.warning(f"Warning: Could not save research plan: {e}")

    def _save_initial_action_plan(self, action_plan) -> None:
        """Save initial action plan to JSON file in session directory"""
        try:
            session_dir = self.vector_store.storage_dir
            plan_path = os.path.join(session_dir, "action_plan_initial.json")

            action_plan.save_to_file(plan_path)
            if self.verbose:
                logger.info(f"⚡ Initial action plan saved to: {plan_path}")

        except Exception as e:
            logger.warning(f"Warning: Could not save initial action plan: {e}")

    def _save_final_action_plan(self, action_plan) -> None:
        """Save final action plan with execution results to JSON file in session directory"""
        try:
            session_dir = self.vector_store.storage_dir
            plan_path = os.path.join(session_dir, "action_plan_final.json")

            action_plan.save_to_file(plan_path)
            if self.verbose:
                logger.info(f"✅ Final action plan saved to: {plan_path}")

        except Exception as e:
            logger.warning(f"Warning: Could not save final action plan: {e}")

    def _execute_action(self, action, context: ResearchContext) -> Dict[str, Any]:
        """Execute action using intelligent tool selection via action planner and enhanced purpose descriptions"""

        try:
            candidate_tools = self._candidate_tools_for_action(action)
            if not candidate_tools:
                return {
                    "action": action.id,
                    "success": False,
                    "error": f"No compatible tool available for action type {action.type.value}",
                }

            # Step 1: Use preferred tool if specified by action planner
            preferred_tool = getattr(action, "preferred_tool", None)
            if preferred_tool:
                tool = self._get_tool_by_name(preferred_tool)
                if tool and any(candidate is tool for candidate in candidate_tools):
                    if self.verbose:
                        logger.info(f"🎯 Using preferred tool: {preferred_tool}")
                    query = self._get_query_for_tool(action)
                    result = tool.execute(query, context)
                    log_tool_call(
                        self.run_config,
                        tool.__class__.__name__,
                        query[:500] if query else "",
                        status="success" if result.get("success", True) else "error",
                    )
                    self._process_action_results(action, result, context)
                    return result
                else:
                    if self.verbose:
                        logger.warning(
                            f"⚠️ Preferred tool {preferred_tool} is unavailable or incompatible with {action.type.value}, using constrained selection"
                        )

            # Step 2: Constrained tool selection based on the action type
            tool = self._select_best_tool(candidate_tools, action)
            query = self._get_query_for_tool(action)
            result = tool.execute(query, context)
            log_tool_call(
                self.run_config,
                tool.__class__.__name__,
                query[:500] if query else "",
                status="success" if result.get("success", True) else "error",
            )
            self._process_action_results(action, result, context)
            return result

        except Exception as e:
            tool_name = getattr(action, "preferred_tool", None) or action.type.value
            log_tool_call(
                self.run_config,
                str(tool_name),
                str(action.parameters.get("query", ""))[:500],
                status="error",
                extra=str(e),
            )
            error_result = {"action": action.id, "success": False, "error": str(e)}
            context.findings[f"error_{action.id}"] = error_result
            return error_result

    def _get_tool_by_name(self, tool_name: str):
        """Get a tool instance by its class name.

        When BrowseComp is enabled, maps InternetSearchTool -> BrowseCompSearchTool
        so action planner's web search actions work transparently.
        """
        # Map InternetSearchTool to BrowseCompSearchTool when browsecomp is enabled
        if self.run_config.browsecomp_enabled and tool_name == "InternetSearchTool":
            tool_name = "BrowseCompSearchTool"

        for tool in self.tool_registry.tools:
            if tool.__class__.__name__ == tool_name:
                return tool
        return None

    def _get_query_for_tool(self, action) -> str:
        """Extract the appropriate query string for a tool based on action parameters"""
        # Priority 1: Check for 'query' parameter (most common)
        query = action.parameters.get("query")
        if query and not query.startswith("[insert"):  # Skip placeholder text
            return query

        # Priority 2: Check for 'operation' parameter (enterprise discovery)
        operation = action.parameters.get("operation")
        if operation:
            return operation

        # Priority 3: Handle specific parameter types for different action types
        if action.type == ActionType.URL_FETCH:
            urls = action.parameters.get("urls", action.parameters.get("url", ""))
            if isinstance(urls, list):
                return " ".join(urls)  # Space-separated URLs for regex extraction
            elif isinstance(urls, str) and urls and not urls.startswith("[insert"):
                return urls  # Single URL string (skip placeholders)
            else:
                return action.description  # Fallback for placeholder URLs

        elif action.type == ActionType.FILE_DOWNLOAD:
            file_path = action.parameters.get("file_path", "")
            return f"download {file_path}" if file_path else action.description

        # Default to action description
        return action.description

    def _select_best_tool(self, candidate_tools: List, action):
        """Select the best tool from candidates - trust the action planner's intelligence"""
        if len(candidate_tools) == 1:
            return candidate_tools[0]

        # If action has preferred tool, try to match it
        if hasattr(action, "preferred_tool") and action.preferred_tool:
            preferred = action.preferred_tool
            # Map InternetSearchTool to BrowseCompSearchTool when browsecomp is enabled
            if self.run_config.browsecomp_enabled and preferred == "InternetSearchTool":
                preferred = "BrowseCompSearchTool"

            for tool in candidate_tools:
                if tool.__class__.__name__ == preferred:
                    return tool

        # Default to first available tool - the action planner should have made the right choice
        # or specified a preferred tool if it mattered
        return candidate_tools[0]

    def _candidate_tools_for_action(self, action) -> List:
        """Return only the tools that are valid for this action type in the current mode."""
        action_type = action.type

        if self.run_config.report_style == "concise_qa":
            allowed_action_types = {
                ActionType.WEB_SEARCH,
                ActionType.LOCAL_DOCUMENT_SEARCH,
                ActionType.DATA_ANALYSIS,
            }
            if action_type not in allowed_action_types:
                return []

        compatible_tool_names: List[str] = []

        if action_type == ActionType.WEB_SEARCH:
            compatible_tool_names = ["BrowseCompSearchTool", "InternetSearchTool"]
        elif action_type == ActionType.LOCAL_DOCUMENT_SEARCH:
            compatible_tool_names = ["LocalFileSearchTool"]
        elif action_type in [ActionType.DATA_ANALYSIS, ActionType.CONTEXT_SYNTHESIS]:
            compatible_tool_names = ["SmartAnalysisTool"]
        elif action_type == ActionType.URL_FETCH:
            compatible_tool_names = ["EnhancedURLFetchTool"]
        elif action_type == ActionType.ENTERPRISE_API:
            compatible_tool_names = ["EnterpriseAPITool"]
        elif action_type == ActionType.MCP_QUERY:
            compatible_tool_names = ["MCPTool"]
        elif action_type == ActionType.FILE_DOWNLOAD:
            compatible_tool_names = []

        candidates = [
            tool for tool in self.tool_registry.tools if tool.__class__.__name__ in compatible_tool_names
        ]

        if action_type == ActionType.URL_FETCH and self.run_config.browsecomp_enabled:
            return []

        return candidates

    def _is_significant_finding(self, result: Dict[str, Any]) -> bool:
        """
        Determine if a finding is significant enough to store in vector store
        """
        if not isinstance(result, dict):
            return False

        # With standardized output, we can rely on consistent fields
        # A finding is significant if:

        # 1. Operation was successful and data was retrieved
        if result.get("success") and result.get("data_retrieved"):
            return True

        # 2. Content was stored in vector store by the tool
        if result.get("stored_in_vector") or result.get("content_stored_in_vector", 0) > 0:
            return True

        # 3. Files were processed
        if result.get("processed_files") and len(result["processed_files"]) > 0:
            return True

        # 4. Substantial synthesis or content was generated
        if result.get("synthesis") and len(str(result["synthesis"])) > 100:
            return True

        # 5. Multiple items found/processed
        items_found = (
            result.get("documents_found", 0)
            + result.get("urls_processed", 0)
            + result.get("successful_actions", 0)
            + len(result.get("results", []))
        )
        if items_found > 0:
            return True

        # 6. Backward compatibility - check for substantial content
        if len(str(result)) > 500:  # Substantial content
            return True

        return False

    def _store_finding_in_vector_store(self, result: Dict[str, Any], original_question: str, action_description: str):
        """
        Store a research finding in the vector store with enhanced processing
        """
        try:
            # Check if content was already stored by enhanced tools
            if result.get("stored_in_vector") or result.get("content_stored_in_vector", 0) > 0:
                if not result.get("stored_in_vector") and self.verbose:
                    logger.debug(f"📄 Content already stored in vector store by {result.get('tool', 'tool')}")
                return

            # Check if this finding has substantial content to store
            if not result.get("success") or not result.get("data_retrieved"):
                # Don't store failed operations or operations without data
                return

            # Create comprehensive content for findings not already stored
            content_parts = []

            # Add the action and tool information
            tool_name = result.get("tool", "unknown")
            content_parts.append(f"Research Action: {action_description}")
            content_parts.append(f"Tool Used: {tool_name}")
            content_parts.append(f"Status: {'Success' if result.get('success') else 'Failed'}")

            # Add summary if available
            if result.get("summary"):
                content_parts.append(f"Summary: {result['summary']}")

            # Add findings summary if available (for enhanced tools)
            if result.get("findings_summary"):
                content_parts.append(f"Findings: {result['findings_summary']}")

            # Add synthesis if available
            if result.get("synthesis"):
                content_parts.append(f"Synthesis: {result['synthesis']}")

            # Add main results
            if result.get("results"):
                results_data = result["results"]
                if isinstance(results_data, list):
                    for i, item in enumerate(results_data[:5]):  # Top 5 results
                        if isinstance(item, dict):
                            item_str = json.dumps(item, indent=2)[:500]  # Limit size
                            content_parts.append(f"Result {i+1}: {item_str}")
                        else:
                            content_parts.append(f"Result {i+1}: {str(item)[:500]}")
                else:
                    content_parts.append(f"Results: {str(results_data)[:1000]}")

            # Add other significant content fields
            content_fields = ["content", "text_content", "data"]
            for field in content_fields:
                if result.get(field) and len(str(result[field])) > 50:
                    content_parts.append(f"{field.replace('_', ' ').title()}: {str(result[field])[:1000]}")

            # Handle extracted_content with lazy loading
            if result.get("extracted_path") and os.path.exists(str(result["extracted_path"])):
                try:
                    with open(result["extracted_path"], "r", encoding="utf-8", errors="ignore") as f:
                        extracted_content = f.read()
                    if len(extracted_content) > 50:
                        content_parts.append(f"Extracted Content: {extracted_content[:1000]}")
                except Exception:
                    pass

            if not content_parts:
                return  # Nothing substantial to store

            content = "\n\n".join(content_parts)

            # Store in vector store with enhanced metadata
            doc_id = self.vector_store.store_document(
                content=content,
                metadata={
                    "type": "research_finding",
                    "tool_used": tool_name,
                    "original_question": original_question,
                    "action_description": action_description,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "success": result.get("success", False),
                    "data_retrieved": result.get("data_retrieved", False),
                    "processed_files": result.get("processed_files", []),
                    "documents_found": result.get("documents_found", 0),
                    "urls_processed": result.get("urls_processed", 0),
                },
            )

            if self.verbose:
                logger.info(f"🗄️ Stored finding in vector store: {doc_id}")

        except Exception as e:
            logger.warning(f"Warning: Could not store finding in vector store: {e}")

    def get_report_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the generated report.

        Returns:
            dict: Metadata about the report, including document references
        """
        return self._last_report_metadata

    def save_report(self, save_path: str, **kwargs) -> Dict[str, Any]:
        """
        Save the report to a file.

        Args:
            save_path: Path to save the report
            **kwargs: Additional arguments specific to the agent implementation

        Returns:
            dict: Result of the save operation
        """
        if not self._last_report:
            return {
                "success": False,
                "error": "No report available to save. Generate a report first.",
            }

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(self._last_report)

            return {
                "success": True,
                "file_path": save_path,
                "metadata": self._last_report_metadata,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _should_stop_research(self, action_plan, context: ResearchContext, iteration: int) -> bool:
        """Determine if research should stop early based on various criteria"""

        # Never stop before minimum iterations
        if iteration < 2:
            return False

        # Stop if we've completed a good number of actions and hit diminishing returns
        completed_count = sum(1 for a in action_plan.actions if a.status == ActionStatus.COMPLETED)

        # Require at least 10 completed actions unless we have very few pending
        pending_count = sum(1 for a in action_plan.actions if a.status == ActionStatus.PENDING)
        if completed_count < 10 and pending_count > 5:
            return False

        # Calculate research coverage
        coverage_score = self._calculate_research_coverage(context, action_plan)
        if coverage_score > 0.85:  # 85% coverage achieved
            if self.verbose:
                logger.info(f"📊 Research coverage: {coverage_score:.2f} - sufficient coverage achieved")
            return True

        # Check for diminishing returns in recent iterations
        if iteration >= 3:
            recent_info_gain = self._calculate_recent_information_gain(action_plan, window=2)
            if recent_info_gain < 0.2:  # Less than 20% new information in recent iterations
                if self.verbose:
                    logger.info(f"📈 Diminishing returns detected: {recent_info_gain:.2f} - stopping research")
                return True

        # Stop if we have very few pending actions and good coverage
        if pending_count <= 2 and coverage_score > 0.6:
            if self.verbose:
                logger.info(f"🎯 Few pending actions ({pending_count}) with good coverage ({coverage_score:.2f})")
            return True

        return False

    def _calculate_research_coverage(self, context: ResearchContext, action_plan) -> float:
        """Calculate how well we've covered the research areas"""
        # Check if we have research plan tasks
        plan_tasks = context.plan.get("research_investigation_areas", []) if context.plan else []

        if not plan_tasks:
            # If no research plan, calculate coverage based on action completion and data retrieval
            completed_actions = [a for a in action_plan.actions if a.status == ActionStatus.COMPLETED]
            if not completed_actions:
                return 0.0

            # Calculate coverage based on successful data retrieval
            successful_data_actions = [
                a for a in completed_actions if a.actual_output and a.actual_output.get("data_retrieved", False)
            ]

            coverage = len(successful_data_actions) / len(completed_actions) if completed_actions else 0.0
            # Scale coverage based on number of completed actions (more actions = better coverage)
            action_bonus = min(len(completed_actions) / 10.0, 0.3)  # Up to 30% bonus for having many actions
            return min(coverage + action_bonus, 1.0)

        # Original logic for research plan-based coverage
        covered_tasks = 0
        for task in plan_tasks:
            task_id = task.get("task_id", str(task.get("query", "")))
            # Check if we have completed actions for this task
            task_actions = [
                a
                for a in action_plan.actions
                if a.created_from_research_step == task_id and a.status == ActionStatus.COMPLETED
            ]
            if task_actions:
                covered_tasks += 1

        return covered_tasks / len(plan_tasks) if plan_tasks else 0.5

    def _calculate_recent_information_gain(self, action_plan, window: int = 2) -> float:
        """Calculate information gain in recent iterations"""
        # Simple heuristic: ratio of successful vs total recent actions
        recent_actions = []
        max_iteration = action_plan.current_iteration
        min_iteration = max(0, max_iteration - window)

        # This is a simplified approach - in practice you'd want to analyze content similarity
        for action in action_plan.actions:
            if (
                action.status == ActionStatus.COMPLETED
                and hasattr(action, "iteration_completed")
                and min_iteration <= getattr(action, "iteration_completed", 0) <= max_iteration
            ):
                recent_actions.append(action)

        if not recent_actions:
            return 0.0

        # Simple scoring based on data retrieved
        successful_actions = sum(
            1 for a in recent_actions if a.actual_output and a.actual_output.get("data_retrieved", False)
        )

        return successful_actions / len(recent_actions) if recent_actions else 0.0

    def _process_action_results(self, action, result: Dict[str, Any], context: ResearchContext):
        """
        Process action results with enhanced content handling
        """

        # Store results in context
        context.findings[f"action_{action.id}"] = result

        # Process any files that were created/downloaded
        processed_files = result.get("processed_files", [])
        for file_path in processed_files:
            if file_path not in context.files_created:
                context.files_created.append(file_path)

        # If this is a significant finding, ensure it's stored in vector store
        if self._is_significant_finding(result):
            self._store_finding_in_vector_store(result, context.original_question, action.description)

        # Enhanced logging with standardized fields
        tool_name = result.get("tool", "unknown")
        success = result.get("success", False)
        data_retrieved = result.get("data_retrieved", False)

        if success:
            if self.verbose:
                status_msg = f"✅ Completed: {action.description} ({tool_name})"

                # Add details about what was retrieved
                details = []
                if data_retrieved:
                    details.append("📊 Data retrieved")
                if processed_files:
                    details.append(f"📁 {len(processed_files)} files processed")
                if result.get("stored_in_vector") or result.get("content_stored_in_vector", 0) > 0:
                    details.append("🗄️ Content stored in vector store")
                if result.get("documents_found", 0) > 0:
                    details.append(f"📄 {result['documents_found']} documents found")
                if result.get("urls_processed", 0) > 0:
                    details.append(f"🌐 {result['urls_processed']} URLs processed")

                if details:
                    status_msg += f" - {', '.join(details)}"

                logger.info(status_msg)
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"❌ Failed: {action.description} ({tool_name}) - {error_msg}")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the research session"""

        vector_stats = self.vector_store.get_stats() if self.vector_store else {}
        content_stats = self.content_processor.get_stats() if self.content_processor else {}

        return {
            "vector_store": vector_stats,
            "content_processing": content_stats,
            "files_in_workspace": (
                len(list(Path(self.workspace_dir).glob("**/*"))) if os.path.exists(self.workspace_dir) else 0
            ),
            "workspace_size_mb": self._get_workspace_size(),
            "last_report_metadata": self._last_report_metadata,
        }

    def _get_workspace_size(self) -> float:
        """Calculate workspace size in MB"""
        try:
            total_size = 0
            for path in Path(self.workspace_dir).rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size
            return round(total_size / (1024 * 1024), 2)
        except Exception as e:
            logger.error(f"Error calculating workspace size: {e}")
            return 0.0

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of research execution with standardized metrics
        New method to provide insights into tool performance
        """

        if not hasattr(self, "_last_report_metadata") or not self._last_report_metadata:
            return {"error": "No research execution data available"}

        # Analyze findings for standardized metrics
        findings_analysis = {
            "total_operations": 0,
            "successful_operations": 0,
            "operations_with_data": 0,
            "failed_operations": 0,
            "tools_used": set(),
            "total_files_processed": 0,
            "total_urls_processed": 0,
            "total_documents_found": 0,
            "vector_storage_operations": 0,
        }

        # Get the research context from last execution
        if hasattr(self, "_last_context"):
            context = self._last_context

            for key, finding in context.findings.items():
                if isinstance(finding, dict):
                    findings_analysis["total_operations"] += 1

                    tool_name = finding.get("tool", "unknown")
                    findings_analysis["tools_used"].add(tool_name)

                    if finding.get("success", True):
                        findings_analysis["successful_operations"] += 1

                        if finding.get("data_retrieved"):
                            findings_analysis["operations_with_data"] += 1

                        # Aggregate data collection metrics
                        findings_analysis["total_files_processed"] += len(finding.get("processed_files", []))
                        findings_analysis["total_urls_processed"] += finding.get("urls_processed", 0)
                        findings_analysis["total_documents_found"] += finding.get("documents_found", 0)

                        if finding.get("stored_in_vector") or finding.get("content_stored_in_vector", 0) > 0:
                            findings_analysis["vector_storage_operations"] += 1
                    else:
                        findings_analysis["failed_operations"] += 1

        # Convert set to list for JSON serialization
        findings_analysis["tools_used"] = list(findings_analysis["tools_used"])

        # Get current stats
        vector_stats = self.vector_store.get_stats() if self.vector_store else {}
        content_stats = self.content_processor.get_stats() if hasattr(self, "content_processor") else {}

        return {
            "execution_analysis": findings_analysis,
            "vector_store_stats": vector_stats,
            "content_processing_stats": content_stats,
            "workspace_info": {
                "workspace_dir": self.workspace_dir,
                "files_created": len(self._last_report_metadata.get("files_created", [])),
                "workspace_size_mb": self._get_workspace_size(),
            },
            "performance_metrics": {
                "success_rate": (
                    findings_analysis["successful_operations"] / max(findings_analysis["total_operations"], 1)
                )
                * 100,
                "data_retrieval_rate": (
                    findings_analysis["operations_with_data"] / max(findings_analysis["successful_operations"], 1)
                )
                * 100,
                "avg_files_per_operation": findings_analysis["total_files_processed"]
                / max(findings_analysis["successful_operations"], 1),
                "vector_utilization_rate": (
                    findings_analysis["vector_storage_operations"] / max(findings_analysis["successful_operations"], 1)
                )
                * 100,
            },
        }
