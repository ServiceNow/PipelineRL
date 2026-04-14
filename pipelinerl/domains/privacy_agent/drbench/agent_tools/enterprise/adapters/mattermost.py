"""Mattermost service adapter implementation"""
import logging
from typing import Any, Dict, List, Optional, Tuple
import requests

from ..base import BaseServiceAdapter, ServiceCapabilities, AuthMethod

logger = logging.getLogger(__name__)


class MattermostAdapter(BaseServiceAdapter):
    """Adapter for Mattermost API v4 integration"""
    
    def __init__(self, config: Dict[str, Any], session: Optional[requests.Session] = None):
        super().__init__("mattermost", config, session or requests.Session())
        self.auth_method = AuthMethod.TOKEN
        self.api_base = f"{self.base_url}/api/v4"
        
    def discover_capabilities(self) -> Dict[str, Any]:
        """Discover Mattermost capabilities and endpoints"""
        capabilities = []
        endpoints = {}
        working_credentials = None
        
        # Test basic connectivity
        if self._test_connectivity():
            capabilities.extend([
                ServiceCapabilities.SYSTEM_INFO,
                ServiceCapabilities.HEALTH_CHECK
            ])
            endpoints["ping"] = f"{self.api_base}/system/ping"
            
        # Try to authenticate and discover more capabilities
        token = self.config.get("token")
        credentials = self.config.get("credentials", {})
        username = credentials.get("username") or self.config.get("username", "admin")
        password = credentials.get("password") or self.config.get("password", "admin")
        
        # Try token first, then login
        if token:
            self.credentials["token"] = token
            if self._test_authenticated_access(token):
                working_credentials = {"token": token}
                self._add_authenticated_capabilities(capabilities, endpoints)
        elif username and password:
            # Try to login
            auth_token = self._login(username, password)
            if auth_token:
                working_credentials = {
                    "token": auth_token,
                    "username": username,
                    "password": password
                }
                self._add_authenticated_capabilities(capabilities, endpoints)
                
        self.capabilities = capabilities
        self.endpoints = endpoints
        self.credentials = working_credentials or {"username": username, "password": password}
        
        return {
            "capabilities": capabilities,
            "endpoints": endpoints,
            "auth_method": self.auth_method,
            "credentials": self.credentials
        }
    
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with Mattermost"""
        token = credentials.get("token")
        username = credentials.get("username")
        password = credentials.get("password")
        
        # Try token authentication first
        if token and self._test_authenticated_access(token):
            self.credentials["token"] = token
            return True
            
        # Try login with username/password
        if username and password:
            auth_token = self._login(username, password)
            if auth_token:
                self.credentials = {
                    "token": auth_token,
                    "username": username,
                    "password": password
                }
                return True
                
        return False
    
    def search(self, terms: List[str], context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for messages in Mattermost"""
        if not self.credentials.get("token"):
            return []
            
        search_query = " ".join(terms)
        search_data = {
            "terms": search_query,
            "is_or_search": True  # Use OR logic for better results
        }
        
        headers = self._get_auth_headers()
        
        try:
            response = self.session.post(
                f"{self.api_base}/posts/search",
                json=search_data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_search_results(data, terms)
                
        except Exception as e:
            logger.error(f"Mattermost search failed: {e}")
            
        return []
    
    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """List teams and channels (Mattermost doesn't have traditional file paths)"""
        if not self.credentials.get("token"):
            return []
            
        # In Mattermost context, "listing files" means listing teams/channels
        teams = self._list_teams()
        items = []
        
        for team in teams:
            items.append({
                "name": team.get("display_name", team.get("name", "")),
                "path": f"/teams/{team['id']}",
                "type": "team",
                "id": team["id"]
            })
            
        return items
    
    def download_file(self, file_path: str) -> Dict[str, Any]:
        """Download file attachment from Mattermost (if file_path is a file ID)"""
        # Mattermost uses file IDs rather than paths
        # This would need to be implemented based on actual file attachment handling
        return {
            "success": False,
            "error": "File download not implemented for Mattermost"
        }
    
    def parse_response(self, response: Any, operation: str) -> Any:
        """Parse Mattermost-specific response formats"""
        if isinstance(response, dict):
            # Mattermost typically returns JSON
            return response
        return response
    
    # Private helper methods
    
    def _test_connectivity(self) -> bool:
        """Test basic Mattermost connectivity"""
        try:
            response = self.session.get(f"{self.api_base}/system/ping", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _test_authenticated_access(self, token: str) -> bool:
        """Test if token provides authenticated access"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.api_base}/teams",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def _login(self, username: str, password: str) -> Optional[str]:
        """Login to Mattermost and get auth token"""
        login_methods = [
            {"login_id": username, "password": password},
            {"email": username, "password": password},
            {"username": username, "password": password}
        ]
        
        for login_data in login_methods:
            try:
                response = self.session.post(
                    f"{self.api_base}/users/login",
                    json=login_data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    # Extract token from headers
                    token = response.headers.get("Token") or response.headers.get("Authorization")
                    if token:
                        return token
                        
            except Exception as e:
                logger.debug(f"Login attempt failed: {e}")
                
        return None
    
    def _add_authenticated_capabilities(self, capabilities: List[str], endpoints: Dict[str, str]):
        """Add capabilities available with authentication"""
        capabilities.extend([
            ServiceCapabilities.TEAM_LISTING,
            ServiceCapabilities.CHANNEL_SEARCH,
            ServiceCapabilities.MESSAGE_SEARCH,
            ServiceCapabilities.USER_SEARCH
        ])
        
        endpoints.update({
            "list_teams": f"{self.api_base}/teams",
            "search_messages": f"{self.api_base}/posts/search",
            "list_channels": f"{self.api_base}/channels",
            "get_user": f"{self.api_base}/users/{{user_id}}"
        })
    
    def _list_teams(self) -> List[Dict[str, Any]]:
        """List all teams accessible to the user"""
        headers = self._get_auth_headers()
        
        try:
            response = self.session.get(
                f"{self.api_base}/teams",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.error(f"Failed to list teams: {e}")
            
        return []
    
    def _parse_search_results(self, data: Dict[str, Any], search_terms: List[str]) -> List[Dict[str, Any]]:
        """Parse Mattermost search response"""
        results = []
        
        posts = data.get("posts", {})
        order = data.get("order", [])
        
        for post_id in order[:20]:  # Limit to first 20 results
            post = posts.get(post_id, {})
            
            # Resolve user and channel info if possible
            user_id = post.get("user_id", "")
            channel_id = post.get("channel_id", "")
            
            result = {
                "post_id": post_id,
                "message": post.get("message", ""),
                "user_id": user_id,
                "channel_id": channel_id,
                "timestamp": post.get("create_at", 0),
                "type": "message",
                "relevance_reason": f"Message contains: {search_terms}"
            }
            
            # Try to resolve names
            if self.credentials.get("token"):
                user_name = self._resolve_user_id(user_id)
                team_name, channel_name = self._resolve_channel_id(channel_id)
                
                result.update({
                    "user_name": user_name,
                    "team_name": team_name,
                    "channel_name": channel_name
                })
                
            results.append(result)
            
        return results
    
    def _resolve_user_id(self, user_id: str) -> str:
        """Resolve user ID to username"""
        if not user_id or not self.credentials.get("token"):
            return user_id
            
        try:
            headers = self._get_auth_headers()
            response = self.session.get(
                f"{self.api_base}/users/{user_id}",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                user_data = response.json()
                return (
                    user_data.get("username") or
                    user_data.get("nickname") or
                    f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip() or
                    user_id
                )
        except:
            pass
            
        return user_id
    
    def _resolve_channel_id(self, channel_id: str) -> Tuple[str, str]:
        """Resolve channel ID to team and channel names"""
        if not channel_id or not self.credentials.get("token"):
            return "unknown", channel_id
            
        try:
            headers = self._get_auth_headers()
            response = self.session.get(
                f"{self.api_base}/channels/{channel_id}",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                channel_data = response.json()
                channel_name = (
                    channel_data.get("display_name") or
                    channel_data.get("name") or
                    channel_id
                )
                
                # Try to get team info
                team_id = channel_data.get("team_id")
                if team_id:
                    team_response = self.session.get(
                        f"{self.api_base}/teams/{team_id}",
                        headers=headers,
                        timeout=5
                    )
                    
                    if team_response.status_code == 200:
                        team_data = team_response.json()
                        team_name = (
                            team_data.get("display_name") or
                            team_data.get("name") or
                            "unknown"
                        )
                        return team_name, channel_name
                        
                return "unknown", channel_name
        except:
            pass
            
        return "unknown", channel_id