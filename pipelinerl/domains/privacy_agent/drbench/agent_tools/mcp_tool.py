import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from .base import ResearchContext, Tool
from ..utils import extract_json

logger = logging.getLogger(__name__)


class MCPTool(Tool):
    """Tool for discovering and interacting with MCP servers with standardized output"""

    @property
    def purpose(self) -> str:
        return """Discovery and interaction with Model Context Protocol (MCP) servers for specialized data access and integrations.
        IDEAL FOR: Accessing specialized data sources, third-party integrations, protocol-based services, and external systems that expose MCP-compatible interfaces.
        USE WHEN: Research requires specialized data sources not available through standard web or enterprise tools, or when specific protocol-based integrations are needed.
        PARAMETERS: query (MCP-related requests - e.g., 'discover available MCP servers', 'query financial data via MCP', 'access specialized databases')
        OUTPUTS: Data from MCP servers, protocol-based integrations, specialized data sources, and discovered server capabilities with structured metadata."""

    def __init__(self, model: str, llm_adapter: Any):
        self.model = model
        self.llm_adapter = llm_adapter
        self.discovered_servers = {}  # Cache for discovered MCP servers
        self.session = requests.Session()


    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute MCP server discovery and interaction with standardized output"""

        try:
            # Step 1: Discover available MCP servers
            mcp_servers = self._discover_mcp_servers(context)

            if not mcp_servers:
                return self.create_success_output(
                    tool_name="mcp",
                    query=query,
                    results=[],
                    data_retrieved=False,
                    servers_discovered=0,
                    message="No MCP servers found in environment",
                )

            # Step 2: Generate action candidates for each server
            all_candidates = []
            for server in mcp_servers:
                candidates = self._generate_mcp_actions(server, query)
                all_candidates.extend(candidates)

            if not all_candidates:
                return self.create_success_output(
                    tool_name="mcp",
                    query=query,
                    results=mcp_servers,
                    data_retrieved=len(mcp_servers) > 0,
                    servers_discovered=len(mcp_servers),
                    candidate_actions=[],
                    message="MCP servers found but no applicable actions generated",
                )

            # Step 3: Execute the most promising actions
            results = []
            successful_actions = 0

            for action in all_candidates[:3]:  # Execute top 3 candidates
                try:
                    result = self._execute_mcp_action(action, query)
                    results.append(result)

                    if result.get("success"):
                        successful_actions += 1

                except Exception as e:
                    results.append({"action": action, "success": False, "error": str(e)})

            return self.create_success_output(
                tool_name="mcp",
                query=query,
                results=results,
                data_retrieved=successful_actions > 0,
                servers_discovered=len(mcp_servers),
                candidate_actions=all_candidates,
                executed_actions=len(results),
                successful_actions=successful_actions,
                discovered_servers=mcp_servers,
            )

        except Exception as e:
            return self.create_error_output("mcp", query, f"MCP tool execution failed: {str(e)}")

    def _discover_mcp_servers(self, context: ResearchContext) -> List[Dict[str, Any]]:
        """Discover available MCP servers in the environment"""

        servers = []

        # Check for common MCP server configurations
        common_mcp_ports = [8001, 8002, 8003, 9001, 9002]
        common_hosts = ["localhost", "127.0.0.1"]

        for host in common_hosts:
            for port in common_mcp_ports:
                server_info = self._probe_mcp_server(host, port)
                if server_info:
                    servers.append(server_info)
                    # Cache discovered server
                    self.discovered_servers[server_info["name"]] = server_info

        # Also check for environment-specified servers
        mcp_servers_env = os.getenv("MCP_SERVERS", "")
        if mcp_servers_env:
            for server_spec in mcp_servers_env.split(","):
                parts = server_spec.strip().split(":")
                if len(parts) >= 3:
                    name, host, port = parts[0], parts[1], int(parts[2])
                    server_info = self._probe_mcp_server(host, port, name)
                    if server_info:
                        servers.append(server_info)
                        self.discovered_servers[server_info["name"]] = server_info

        return servers

    def _probe_mcp_server(self, host: str, port: int, name: str = None) -> Optional[Dict[str, Any]]:
        """Probe a potential MCP server to check if it's available"""

        server_name = name or f"mcp_server_{host}_{port}"
        base_url = f"http://{host}:{port}"

        # Try common MCP endpoints
        mcp_endpoints = ["/mcp/capabilities", "/mcp/tools", "/mcp/status", "/capabilities", "/tools", "/status", "/"]

        for endpoint in mcp_endpoints:
            try:
                url = f"{base_url}{endpoint}"
                response = self.session.get(url, timeout=3)

                if response.status_code == 200:
                    # Try to determine if this is actually an MCP server
                    content = response.text.lower()
                    response_data = None

                    try:
                        response_data = response.json()
                    except:
                        pass

                    # Look for MCP-like indicators
                    mcp_indicators = ["mcp", "model context protocol", "tools", "capabilities"]
                    is_mcp_like = any(indicator in content for indicator in mcp_indicators)

                    if is_mcp_like or response_data:
                        return {
                            "name": server_name,
                            "host": host,
                            "port": port,
                            "base_url": base_url,
                            "discovered_endpoint": endpoint,
                            "capabilities": response_data if response_data else {},
                            "status": "active",
                            "probe_successful": True,
                        }

            except Exception:
                continue

        return None

    def _generate_mcp_actions(self, server: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Generate action candidates for an MCP server"""

        server_name = server.get("name", "unknown")
        base_url = server.get("base_url", "")
        capabilities = server.get("capabilities", {})

        # Use LLM to generate smart actions based on server capabilities
        prompt = f"""
You are an MCP (Model Context Protocol) interaction specialist. Given a research query and an MCP server's capabilities, generate specific actionable requests.

Query: "{query}"

MCP Server: {server_name}
Base URL: {base_url}
Discovered Capabilities: {json.dumps(capabilities, indent=2)}

Generate 2-3 candidate actions for this MCP server that could help answer the research query. For each action, specify:

1. What MCP method/tool to call
2. What parameters to send
3. What type of data you expect to retrieve
4. How this relates to the research query

Common MCP operations include:
- list_tools: Get available tools
- call_tool: Execute a specific tool
- get_capabilities: Get server capabilities
- query_data: Query for specific data

Return a JSON array of actions in this format:
[
  {{
    "server": "{server_name}",
    "operation": "list_tools|call_tool|get_capabilities|query_data",
    "endpoint": "suggested_endpoint_path",
    "method": "GET|POST",
    "parameters": {{"key": "value"}},
    "description": "What this action will attempt to do",
    "expected_data": "Type of data expected",
    "confidence": 0.8
  }}
]
"""

        try:
            response = self.llm_adapter.generate_text(
                prompt,
                model=self.model,
                log_name="mcp_action_generation",
            )
            actions = extract_json(response)
            return actions if isinstance(actions, list) else []
        except Exception as e:
            logger.error(f"Error generating MCP actions: {e}")
            # Fallback to basic actions
            return self._fallback_mcp_actions(server, query)

    def _fallback_mcp_actions(self, server: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Fallback MCP action generation when LLM fails"""

        server_name = server.get("name", "unknown")
        base_url = server.get("base_url", "")

        actions = [
            {
                "server": server_name,
                "operation": "list_tools",
                "endpoint": "/mcp/tools",
                "method": "GET",
                "parameters": {},
                "description": f"List available tools on {server_name}",
                "expected_data": "List of available MCP tools",
                "confidence": 0.9,
            },
            {
                "server": server_name,
                "operation": "get_capabilities",
                "endpoint": "/mcp/capabilities",
                "method": "GET",
                "parameters": {},
                "description": f"Get capabilities of {server_name}",
                "expected_data": "Server capabilities and supported operations",
                "confidence": 0.8,
            },
            {
                "server": server_name,
                "operation": "query_data",
                "endpoint": "/mcp/query",
                "method": "POST",
                "parameters": {"query": query},
                "description": f"Query {server_name} for data related to: {query}",
                "expected_data": "Relevant data matching the query",
                "confidence": 0.6,
            },
        ]

        return actions

    def _execute_mcp_action(self, action: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Execute a specific MCP action with standardized output"""

        server_name = action.get("server", "unknown")
        operation = action.get("operation", "")
        endpoint = action.get("endpoint", "")
        method = action.get("method", "GET")
        parameters = action.get("parameters", {})

        # Get the server info to build the full URL
        server_info = self.discovered_servers.get(server_name)
        if server_info:
            base_url = server_info.get("base_url", "")
        else:
            # Try to extract from action if available
            base_url = action.get("base_url", "")

        if not base_url:
            return {
                "action": action,
                "success": False,
                "error": "Server base URL not available",
                "server": server_name,
                "operation": operation,
            }

        full_url = f"{base_url.rstrip('/')}{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(full_url, params=parameters, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(full_url, json=parameters, timeout=10)
            else:
                response = self.session.request(method, full_url, json=parameters, timeout=10)

            if response.status_code == 200:
                try:
                    data = response.json()
                except:
                    data = {"content": response.text[:500]}

                return {
                    "action": action,
                    "success": True,
                    "server": server_name,
                    "operation": operation,
                    "data": data,
                    "data_retrieved": True,
                    "response_size": len(response.content),
                    "status_code": response.status_code,
                }
            else:
                return {
                    "action": action,
                    "success": False,
                    "server": server_name,
                    "operation": operation,
                    "error": f"HTTP {response.status_code}",
                    "response_text": response.text[:200],
                }

        except requests.exceptions.Timeout:
            return {
                "action": action,
                "success": False,
                "server": server_name,
                "operation": operation,
                "error": "Request timeout",
            }
        except requests.exceptions.RequestException as e:
            return {
                "action": action,
                "success": False,
                "server": server_name,
                "operation": operation,
                "error": f"Network error: {str(e)}",
            }
        except Exception as e:
            return {"action": action, "success": False, "server": server_name, "operation": operation, "error": str(e)}

    def list_discovered_servers(self) -> List[Dict[str, Any]]:
        """Get list of all discovered MCP servers"""
        return list(self.discovered_servers.values())

    def get_server_tools(self, server_name: str) -> Dict[str, Any]:
        """Get available tools for a specific MCP server with standardized output"""

        server_info = self.discovered_servers.get(server_name)
        if not server_info:
            return {"success": False, "error": "Server not found", "server": server_name}

        base_url = server_info.get("base_url", "")
        tools_endpoint = f"{base_url}/mcp/tools"

        try:
            response = self.session.get(tools_endpoint, timeout=10)
            if response.status_code == 200:
                try:
                    tools_data = response.json()
                    return {"success": True, "server": server_name, "tools": tools_data, "data_retrieved": True}
                except:
                    return {
                        "success": True,
                        "server": server_name,
                        "tools": {"raw_response": response.text},
                        "data_retrieved": True,
                    }
            else:
                return {"success": False, "server": server_name, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "server": server_name, "error": str(e)}
