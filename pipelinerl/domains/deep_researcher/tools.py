"""Tools for DeepResearcher domain (web search and reader)."""
import os
import logging
import asyncio
from typing import Dict, List, Union
import aiohttp

logger = logging.getLogger(__name__)


class AsyncJinaSearch:
    
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            logger.warning("JINA_API_KEY not set, search functionality will be limited")
        
        self.base_url = "https://s.jina.ai/"
        logger.info("[AsyncJinaSearch] Initialized with Jina Search API")
    
    async def search(
        self,
        query: str,
        session: aiohttp.ClientSession = None
    ) -> str:
        if not self.api_key:
            return f"Error: JINA_API_KEY not set. Cannot perform search for '{query}'."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Return-Format": "markdown"
        }
        
        search_url = f"{self.base_url}{query}"
        
        try:
            # Use provided session or create new one
            if session:
                async with session.get(
                    search_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return await self._process_response(response, query)
            else:
                async with aiohttp.ClientSession() as new_session:
                    async with new_session.get(
                        search_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        return await self._process_response(response, query)
            
        except aiohttp.ClientError as e:
            error_msg = f"Search API error for '{query}': {str(e)}"
            logger.error(f"[AsyncJinaSearch] {error_msg}")
            return error_msg
        except asyncio.TimeoutError:
            error_msg = f"Search timeout for '{query}'"
            logger.error(f"[AsyncJinaSearch] {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error for '{query}': {str(e)}"
            logger.error(f"[AsyncJinaSearch] {error_msg}")
            return error_msg
    
    async def _process_response(self, response, query: str) -> str:
        if response.status == 200:
            content = await response.text()
            
            max_length = 8000
            if len(content) > max_length:
                content = content[:max_length] + "\n\n[Results truncated...]"
            
            formatted = f"Search results for '{query}':\n\n{content}"
            logger.info(f"[AsyncJinaSearch] Found results for: {query[:50]}... ({len(content)} chars)")
            return formatted
        else:
            error_msg = f"Search API error for '{query}': HTTP {response.status}"
            logger.error(f"[AsyncJinaSearch] {error_msg}")
            return error_msg


class AsyncJinaReader:
    
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            logger.warning("JINA_API_KEY not set, reader functionality will be limited")
        
        logger.info("[AsyncJinaReader] Initialized with Jina Reader API")
    
    async def read(
        self,
        url: str,
        goal: str = "Extract relevant information",
        session: aiohttp.ClientSession = None
    ) -> str:
        if not self.api_key:
            return f"Error: JINA_API_KEY not set. Cannot read {url}."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Return-Format": "text"
        }
        
        jina_url = f"https://r.jina.ai/{url}"
        
        try:
            # Use provided session or create new one
            if session:
                async with session.get(
                    jina_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=100)
                ) as response:
                    return await self._process_response(response, url, goal)
            else:
                async with aiohttp.ClientSession() as new_session:
                    async with new_session.get(
                        jina_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=100)
                    ) as response:
                        return await self._process_response(response, url, goal)
                        
        except aiohttp.ClientError as e:
            error_msg = f"Network error reading {url}: {str(e)}"
            logger.error(f"[AsyncJinaReader] {error_msg}")
            return error_msg
        except asyncio.TimeoutError:
            error_msg = f"Timeout reading {url}"
            logger.error(f"[AsyncJinaReader] {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error reading {url}: {str(e)}"
            logger.error(f"[AsyncJinaReader] {error_msg}")
            return error_msg
    
    async def _process_response(self, response, url: str, goal: str) -> str:
        if response.status == 200:
            content = await response.text()
            
            max_length = 10000
            if len(content) > max_length:
                content = content[:max_length] + "\n\n[Content truncated...]"
            
            formatted = f"Content from {url} (Goal: {goal}):\n\n{content}"
            logger.info(f"[AsyncJinaReader] Successfully read {url} ({len(content)} chars)")
            return formatted
        else:
            error_msg = f"Error reading {url}: HTTP {response.status}"
            logger.error(f"[AsyncJinaReader] {error_msg}")
            return error_msg


class MockWebSearchTool:
    
    def __init__(self):
        self.knowledge_base = {
            "paris": "Paris is the capital and largest city of France. Population: approximately 2.2 million in the city proper, 12 million in the metropolitan area.",
            "eiffel tower": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. Height: 330 meters (1,083 ft). Built in 1889.",
            "france": "France is a country in Western Europe. Capital: Paris. Population: 67 million. Official language: French.",
            "tokyo": "Tokyo is the capital of Japan. Population: approximately 14 million in the city proper, 37 million in the metropolitan area.",
            "japan": "Japan is an island country in East Asia. Capital: Tokyo. Population: 125 million.",
            "python": "Python is a high-level programming language. Created by Guido van Rossum in 1991. Known for readability and versatility.",
            "machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Common algorithms include neural networks, decision trees, and support vector machines.",
        }
        self.last_search_results = []
        logger.info("[MockWebSearchTool] Initialized with mock knowledge base")
    
    async def search(
        self,
        query: str,
        num_results: int = 3,
        session: aiohttp.ClientSession = None
    ) -> str:
        await asyncio.sleep(0.1)
        
        query_lower = query.lower()
        results = []
        
        for key, value in self.knowledge_base.items():
            if key in query_lower or any(word in key for word in query_lower.split()):
                results.append({
                    "title": key.title(),
                    "snippet": value,
                    "url": f"https://example.com/{key.replace(' ', '_')}"
                })
        
        if not results:
            results = [{
                "title": "No specific results found",
                "snippet": f"No information found for '{query}'. Try a different search term.",
                "url": "https://example.com/not_found"
            }]
        
        results = results[:num_results]
        self.last_search_results = results
        
        formatted = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   URL: {result['url']}\n\n"
        
        logger.info(f"[MockWebSearchTool] Mock search for '{query}' returned {len(results)} results")
        return formatted.strip()


class MockWebLookupTool:
    
    def __init__(self, search_tool: MockWebSearchTool):
        self.search_tool = search_tool
        logger.info("[MockWebLookupTool] Initialized")
    
    async def lookup(
        self,
        term: str,
        session: aiohttp.ClientSession = None
    ) -> str:
        await asyncio.sleep(0.1)
        
        term_lower = term.lower()
        
        if self.search_tool.last_search_results:
            for result in self.search_tool.last_search_results:
                if term_lower in result['snippet'].lower() or term_lower in result['title'].lower():
                    return f"Found information about '{term}':\n{result['snippet']}"
        
        for key, value in self.search_tool.knowledge_base.items():
            if term_lower in key or term_lower in value.lower():
                return f"Information about '{term}':\n{value}"
        
        return f"No specific information found for '{term}' in recent results. Try a new search."


class ToolRegistry:
    
    def __init__(self, use_mock: bool = None):
        if use_mock is None:
            use_mock = not bool(os.getenv("JINA_API_KEY"))
        
        if use_mock:
            logger.info("[ToolRegistry] Using MOCK tools (no API key required)")
            self.search_tool = MockWebSearchTool()
            self.lookup_tool = MockWebLookupTool(self.search_tool)
            self.reader_tool = None
        else:
            logger.info("[ToolRegistry] Using REAL Jina API tools")
            self.search_tool = AsyncJinaSearch()
            self.reader_tool = AsyncJinaReader()
            self.lookup_tool = None
        
        self.use_mock = use_mock
    
    def get_tool_descriptions(self) -> str:
        if self.use_mock:
            return """
Available Tools:

1. Search[query]: Search for information about a topic
   Returns: List of search results with titles and snippets
   Example: Search[Paris population]

2. Lookup[term]: Look up a specific term in recent search results
   Returns: Detailed information about the term
   Example: Lookup[population]
"""
        else:
            return """
Available Tools:

1. Search[query]: Search the web for information
   Returns: Markdown-formatted search results
   Example: Search[Paris population 2024]

2. Read[url]: Read and extract content from a webpage
   Returns: Extracted text content from the URL
   Example: Read[https://example.com/article]
"""
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        session: aiohttp.ClientSession
    ) -> str:
        try:
            if tool_name == "web_search":
                return await self.search_tool.search(
                    query=arguments.get("query", ""),
                    session=session
                )
            elif tool_name == "web_lookup" and self.lookup_tool:
                return await self.lookup_tool.lookup(
                    term=arguments.get("term", ""),
                    session=session
                )
            elif tool_name == "web_read" and self.reader_tool:
                return await self.reader_tool.read(
                    url=arguments.get("url", ""),
                    goal=arguments.get("goal", "Extract relevant information"),
                    session=session
                )
            else:
                return f"Error: Unknown tool '{tool_name}' or tool not available in current mode"
        except Exception as e:
            logger.error(f"[ToolRegistry] Tool execution failed: {e}")
            return f"Error executing {tool_name}: {str(e)}"
