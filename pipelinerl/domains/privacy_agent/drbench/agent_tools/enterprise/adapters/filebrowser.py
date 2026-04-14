"""FileBrowser service adapter implementation"""
import json
import logging
from typing import Any, Dict, List, Optional
import requests

from ..base import BaseServiceAdapter, ServiceCapabilities, AuthMethod

logger = logging.getLogger(__name__)


class FileBrowserAdapter(BaseServiceAdapter):
    """Adapter for FileBrowser REST API integration"""
    
    def __init__(self, config: Dict[str, Any], session: Optional[requests.Session] = None):
        super().__init__("filebrowser", config, session or requests.Session())
        self.auth_method = AuthMethod.CUSTOM  # Uses X-Auth header
        self.api_base = f"{self.base_url}/api"
        
    def discover_capabilities(self) -> Dict[str, Any]:
        """Discover FileBrowser capabilities and endpoints"""
        capabilities = []
        endpoints = {}
        working_credentials = None
        
        credentials = self.config.get("credentials", {})
        username = credentials.get("username") or self.config.get("username", "admin")
        password = credentials.get("password") or self.config.get("password", "admin")
        
        # Try to login and test access
        token = self._login(username, password)
        if token:
            working_credentials = {
                "username": username,
                "password": password,
                "token": token
            }
            
            # Test file listing to confirm access
            if self._test_file_access(token):
                capabilities.extend([
                    ServiceCapabilities.FILE_LISTING,
                    ServiceCapabilities.FILE_DOWNLOAD,
                    ServiceCapabilities.FILE_SEARCH,
                    ServiceCapabilities.FILE_UPLOAD
                ])
                
                endpoints.update({
                    "list_files": f"{self.api_base}/resources/",
                    "download_file": f"{self.api_base}/raw/",
                    "search_files": f"{self.api_base}/search/",
                    "upload_file": f"{self.api_base}/resources/"
                })
                
        self.capabilities = capabilities
        self.endpoints = endpoints
        self.credentials = working_credentials or {"username": username, "password": password}
        self.auth_method = AuthMethod.CUSTOM
        
        return {
            "capabilities": capabilities,
            "endpoints": endpoints,
            "auth_method": self.auth_method,
            "credentials": self.credentials
        }
    
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with FileBrowser"""
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            return False
            
        token = self._login(username, password)
        if token:
            self.credentials = {
                "username": username,
                "password": password,
                "token": token
            }
            return True
            
        return False
    
    def search(self, terms: List[str], context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for files in FileBrowser"""
        if not self.credentials.get("token"):
            return []
            
        # FileBrowser doesn't have a dedicated search API,
        # so we'll list files and filter
        all_files = self._list_all_files()
        
        matching_files = []
        for file_info in all_files:
            file_name = file_info.get("name", "").lower()
            if any(term.lower() in file_name for term in terms):
                file_info["relevance_reason"] = f"Filename matches: {terms}"
                matching_files.append(file_info)
                
        return matching_files[:20]  # Limit results
    
    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """List files at the given path"""
        if not self.credentials.get("token"):
            return []
            
        headers = self._get_auth_headers()
        
        try:
            # FileBrowser expects paths without leading slash in API
            api_path = path.lstrip("/")
            url = f"{self.api_base}/resources/{api_path}"
            
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_file_listing(data)
                
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            
        return []
    
    def download_file(self, file_path: str) -> Dict[str, Any]:
        """Download a file from FileBrowser"""
        if not self.credentials.get("token"):
            return {"success": False, "error": "Not authenticated"}
            
        headers = self._get_auth_headers()
        
        try:
            # FileBrowser download endpoint
            api_path = file_path.lstrip("/")
            url = f"{self.api_base}/raw/{api_path}"
            
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "content": response.content,
                    "content_type": response.headers.get("content-type", ""),
                    "content_size": len(response.content),
                    "file_path": file_path
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def parse_response(self, response: Any, operation: str) -> Any:
        """Parse FileBrowser-specific response formats"""
        # FileBrowser typically returns JSON
        return response
    
    # Private helper methods
    
    def _login(self, username: str, password: str) -> Optional[str]:
        """Login to FileBrowser and get auth token"""
        try:
            response = self.session.post(
                f"{self.api_base}/login",
                json={"username": username, "password": password},
                timeout=5
            )
            
            if response.status_code == 200:
                # Token is returned as a plain string in quotes
                token = response.text.strip('"')
                return token
                
        except Exception as e:
            logger.debug(f"FileBrowser login failed: {e}")
            
        return None
    
    def _test_file_access(self, token: str) -> bool:
        """Test if token provides file access"""
        try:
            headers = {"X-Auth": token}
            response = self.session.get(
                f"{self.api_base}/resources/",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get FileBrowser-specific auth headers"""
        headers = {}
        if self.credentials.get("token"):
            headers["X-Auth"] = self.credentials["token"]
        return headers
    
    def _parse_file_listing(self, data: Any) -> List[Dict[str, Any]]:
        """Parse FileBrowser file listing response"""
        files = []
        
        if isinstance(data, dict):
            # Single directory info with items
            items = data.get("items", [])
        elif isinstance(data, list):
            # Direct list of items
            items = data
        else:
            return files
            
        for item in items:
            file_info = {
                "name": item.get("name", ""),
                "path": item.get("path", ""),
                "type": "directory" if item.get("isDir", False) else "file",
                "size": item.get("size", 0),
                "modified": item.get("modified", "")
            }
            files.append(file_info)
            
        return files
    
    def _list_all_files(self, path: str = "/", max_depth: int = 3) -> List[Dict[str, Any]]:
        """Recursively list all files (with depth limit)"""
        if max_depth <= 0:
            return []
            
        files = []
        items = self.list_files(path)
        
        for item in items:
            if item["type"] == "file":
                files.append(item)
            elif item["type"] == "directory" and not item["name"].startswith("."):
                # Recurse into subdirectories
                subpath = item.get("path", f"{path}/{item['name']}".replace("//", "/"))
                files.extend(self._list_all_files(subpath, max_depth - 1))
                
        return files