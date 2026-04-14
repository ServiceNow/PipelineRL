"""Nextcloud service adapter implementation"""
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import requests

from ..base import BaseServiceAdapter, ServiceCapabilities, AuthMethod

logger = logging.getLogger(__name__)


class NextcloudAdapter(BaseServiceAdapter):
    """Adapter for Nextcloud WebDAV and OCS API integration"""
    
    def __init__(self, config: Dict[str, Any], session: Optional[requests.Session] = None):
        super().__init__("nextcloud", config, session or requests.Session())
        self.auth_method = AuthMethod.BASIC
        
    def discover_capabilities(self) -> Dict[str, Any]:
        """Discover Nextcloud capabilities and endpoints"""
        capabilities = []
        endpoints = {}
        working_credentials = None
        
        # Try different credential sources
        credential_options = [
            self.config.get("credentials", {}),
        ]
        
        for creds in credential_options:
            username = creds.get("username")
            password = creds.get("password")
            
            if not username or not password:
                continue
                
            logger.debug(f"Trying Nextcloud credentials: {username}")
            
            if self._test_webdav_access(username, password):
                working_credentials = {"username": username, "password": password}
                capabilities.extend([
                    ServiceCapabilities.FILE_LISTING,
                    ServiceCapabilities.FILE_DOWNLOAD,
                    ServiceCapabilities.FILE_SEARCH
                ])
                endpoints["list_files"] = f"{self.base_url}/remote.php/dav/files/{username}/"
                endpoints["download_file"] = endpoints["list_files"] + "{filepath}"
                endpoints["search_files"] = f"{self.base_url}/index.php/apps/files/api/v1/search"
                
                # Test OCS API for additional features
                if self._test_ocs_api(username, password):
                    capabilities.append(ServiceCapabilities.SHARING_MANAGEMENT)
                    endpoints["list_shares"] = f"{self.base_url}/ocs/v2.php/apps/files_sharing/api/v1/shares"
                    
                break
                
        self.capabilities = capabilities
        self.endpoints = endpoints
        self.credentials = working_credentials or {"username": "admin", "password": "admin"}
        
        return {
            "capabilities": capabilities,
            "endpoints": endpoints,
            "auth_method": self.auth_method,
            "credentials": self.credentials
        }
    
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Test authentication with provided credentials"""
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            return False
            
        if self._test_webdav_access(username, password):
            self.credentials = credentials
            return True
            
        return False
    
    def search(self, terms: List[str], context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for files in Nextcloud"""
        results = []
        
        # Try Search API first
        if "search_files" in self.endpoints:
            try:
                search_results = self._search_via_api(terms)
                results.extend(search_results)
            except Exception as e:
                logger.debug(f"Search API failed, falling back to WebDAV: {e}")
                
        # Fallback to WebDAV listing
        if not results:
            webdav_results = self._search_via_webdav(terms)
            results.extend(webdav_results)
            
        return results
    
    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """List files at the given path"""
        username = self.credentials.get("username")
        if not username:
            return []
            
        # Ensure path is relative to user's files
        if not path.startswith(f"/remote.php/dav/files/{username}/"):
            path = f"/remote.php/dav/files/{username}/{path.lstrip('/')}"
            
        full_url = self.base_url + path
        
        try:
            response = self._propfind_request(full_url)
            if response.status_code == 207:  # Multi-Status
                return self._parse_propfind_response(response.text)
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            
        return []
    
    def download_file(self, file_path: str) -> Dict[str, Any]:
        """Download a file from Nextcloud"""
        username = self.credentials.get("username")
        password = self.credentials.get("password")
        
        if not username or not password:
            return {"success": False, "error": "No credentials available"}
            
        # Construct proper download URL
        if file_path.startswith("/remote.php/dav/"):
            download_url = self.base_url + file_path
        else:
            download_url = f"{self.base_url}/remote.php/dav/files/{username}/{file_path.lstrip('/')}"
            
        try:
            response = self.session.get(
                download_url,
                auth=(username, password),
                timeout=30
            )
            
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
        """Parse Nextcloud-specific response formats"""
        if operation in ["list", "search"] and isinstance(response, str):
            # Parse WebDAV XML response
            try:
                return self._parse_propfind_response(response)
            except ET.ParseError:
                return []
        return response
    
    # Private helper methods
    
    def _test_webdav_access(self, username: str, password: str) -> bool:
        """Test WebDAV access with credentials"""
        try:
            webdav_url = f"{self.base_url}/remote.php/dav/files/{username}/"
            response = self._propfind_request(webdav_url, auth=(username, password))
            
            if response.status_code == 207:  # Multi-Status
                return True
            elif response.status_code == 200 and "<?xml" in response.text:
                return True  # Some instances return 200
                
        except Exception as e:
            logger.debug(f"WebDAV test failed: {e}")
            
        return False
    
    def _test_ocs_api(self, username: str, password: str) -> bool:
        """Test OCS API access"""
        try:
            ocs_url = f"{self.base_url}/ocs/v2.php/apps/files_sharing/api/v1/shares"
            response = self.session.get(
                ocs_url,
                auth=(username, password),
                headers={"OCS-APIRequest": "true"},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def _propfind_request(self, url: str, auth: Optional[tuple] = None) -> requests.Response:
        """Execute a PROPFIND request"""
        if not auth:
            auth = (self.credentials.get("username"), self.credentials.get("password"))
            
        propfind_body = """<?xml version="1.0" encoding="utf-8" ?>
        <D:propfind xmlns:D="DAV:">
            <D:prop>
                <D:displayname/>
                <D:resourcetype/>
                <D:getcontentlength/>
                <D:getlastmodified/>
            </D:prop>
        </D:propfind>"""
        
        headers = {
            "Depth": "1",
            "Content-Type": "application/xml"
        }
        
        return self.session.request(
            "PROPFIND",
            url,
            data=propfind_body,
            headers=headers,
            auth=auth,
            timeout=10
        )
    
    def _parse_propfind_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse WebDAV PROPFIND XML response"""
        files = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for response in root.findall(".//{DAV:}response"):
                href = response.find(".//{DAV:}href")
                displayname = response.find(".//{DAV:}displayname")
                resourcetype = response.find(".//{DAV:}resourcetype")
                contentlength = response.find(".//{DAV:}getcontentlength")
                lastmodified = response.find(".//{DAV:}getlastmodified")
                
                if href is not None:
                    file_info = {
                        "path": href.text,
                        "name": displayname.text if displayname is not None else href.text.split("/")[-1],
                        "type": "directory" if resourcetype is not None and resourcetype.find(".//{DAV:}collection") is not None else "file",
                        "size": int(contentlength.text) if contentlength is not None and contentlength.text else 0,
                        "modified": lastmodified.text if lastmodified is not None else ""
                    }
                    files.append(file_info)
                    
        except Exception as e:
            logger.error(f"Failed to parse WebDAV response: {e}")
            
        return files
    
    def _search_via_api(self, terms: List[str]) -> List[Dict[str, Any]]:
        """Search using Nextcloud Search API"""
        if "search_files" not in self.endpoints:
            return []
            
        search_query = " ".join(terms)
        params = {
            "query": search_query,
            "limit": 50,
            "type": "files"
        }
        
        auth = self._get_auth_tuple()
        response = self.session.get(
            self.endpoints["search_files"],
            params=params,
            auth=auth,
            timeout=10
        )
        
        if response.status_code == 200:
            # Parse search API response
            try:
                data = response.json()
                return self._convert_search_results(data)
            except:
                pass
                
        return []
    
    def _search_via_webdav(self, terms: List[str]) -> List[Dict[str, Any]]:
        """Search by listing files and filtering"""
        all_files = self._list_all_files_recursive()
        
        # Filter files matching search terms
        matching_files = []
        for file_info in all_files:
            file_name = file_info.get("name", "").lower()
            if any(term.lower() in file_name for term in terms):
                file_info["relevance_reason"] = f"Filename matches: {terms}"
                matching_files.append(file_info)
                
        return matching_files[:20]  # Limit results
    
    def _list_all_files_recursive(self, path: str = "/", max_depth: int = 3) -> List[Dict[str, Any]]:
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
                subpath = item["path"]
                files.extend(self._list_all_files_recursive(subpath, max_depth - 1))
                
        return files
    
    def _convert_search_results(self, api_results: Any) -> List[Dict[str, Any]]:
        """Convert API search results to standard format"""
        # Implementation depends on Nextcloud search API response format
        # This is a placeholder that should be adapted based on actual API
        if isinstance(api_results, list):
            return api_results
        elif isinstance(api_results, dict) and "files" in api_results:
            return api_results["files"]
        return []