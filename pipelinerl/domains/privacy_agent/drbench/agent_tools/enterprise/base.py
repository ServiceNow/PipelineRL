"""Base classes and interfaces for enterprise service adapters"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseServiceAdapter(ABC):
    """Base class for all enterprise service adapters
    
    Service adapters should:
    1. Only be available if valid credentials are provided in configuration
    2. Use credentials exclusively from the config (no hardcoded defaults)
    3. Store discovered endpoints for documentation and debugging purposes
    
    Attributes:
        service_name: Name of the service (e.g., "email_imap", "nextcloud")
        config: Service configuration from environment, including credentials
        base_url: Optional web interface URL (not used for protocol-specific connections)
        endpoints: Dict of discovered endpoints for documentation purposes
        credentials: Credentials from configuration (should not contain defaults)
    """
    
    def __init__(self, service_name: str, config: Dict[str, Any], session=None):
        self.service_name = service_name
        self.config = config
        self.base_url = config.get("url", "")  # Optional web interface URL
        self.session = session
        self.capabilities = {}
        self.endpoints = {}  # For documentation/debugging, not for connection
        self.auth_method = "none"
        self.credentials = {}  # Should only contain validated credentials from config
        
    @abstractmethod
    def discover_capabilities(self) -> Dict[str, Any]:
        """
        Discover service capabilities and available endpoints
        
        This method should:
        1. Check if valid credentials exist in the configuration
        2. Return empty capabilities if no credentials are provided
        3. Test authentication with provided credentials
        4. Only enable capabilities if authentication succeeds
        
        Returns:
            Dict containing:
                - capabilities: List[str] of available operations (empty if no auth)
                - endpoints: Dict[str, str] for documentation purposes
                - auth_method: str indicating authentication type
                - credentials: Dict with username only (never return passwords)
                - error: Optional error message if service is unavailable
        """
        pass
    
    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with the service using provided credentials
        
        This method should:
        1. Test authentication with the provided credentials
        2. Store validated credentials internally if successful
        3. Not fall back to default credentials
        
        Args:
            credentials: Dict containing authentication details from config
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search(self, terms: List[str], context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for content within the service
        
        Args:
            terms: List of search terms
            context: Optional context for the search
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """
        List files/items at the given path
        
        Args:
            path: Path to list (default: root)
            
        Returns:
            List of file/item information
        """
        pass
    
    @abstractmethod
    def download_file(self, file_path: str) -> Dict[str, Any]:
        """
        Download a file from the service
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing file content and metadata
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: Any, operation: str) -> Any:
        """
        Parse service-specific response format
        
        Args:
            response: Raw response from the service
            operation: The operation that generated this response
            
        Returns:
            Parsed response data
        """
        pass
    
    def execute_action(self, action: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Execute a specific action on the service
        
        Args:
            action: Action definition with operation, parameters, etc.
            original_query: The original user query
            
        Returns:
            Execution result
        """
        operation = action.get("operation")
        parameters = action.get("parameters", {})
        
        try:
            if operation == "search":
                search_terms = parameters.get("search_terms", [])
                results = self.search(search_terms, {"query": original_query})
                return self._create_result(True, operation, results=results)
                
            elif operation == "list":
                path = parameters.get("path", "/")
                results = self.list_files(path)
                return self._create_result(True, operation, results=results)
                
            elif operation == "download":
                file_path = parameters.get("file_path")
                result = self.download_file(file_path)
                return self._create_result(True, operation, **result)
                
            elif operation == "discover":
                result = self.discover_capabilities()
                return self._create_result(True, operation, capabilities=result)
                
            else:
                return self._create_result(False, operation, 
                                         error=f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"{self.service_name}: {operation} failed - {str(e)}")
            return self._create_result(False, operation, error=str(e))
    
    def _create_result(self, success: bool, operation: str, **kwargs) -> Dict[str, Any]:
        """Helper to create standardized result"""
        result = {
            "service": self.service_name,
            "operation": operation,
            "success": success,
            "data_retrieved": success and any(
                kwargs.get(field) for field in ["results", "content", "files"]
            )
        }
        result.update(kwargs)
        return result
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on auth method"""
        headers = {}
        
        if self.auth_method == "token" and self.credentials.get("token"):
            headers["Authorization"] = f"Bearer {self.credentials['token']}"
        elif self.auth_method == "custom" and self.credentials.get("headers"):
            headers.update(self.credentials["headers"])
            
        return headers
    
    def _get_auth_tuple(self) -> Optional[Tuple[str, str]]:
        """Get basic auth tuple if applicable"""
        if self.auth_method == "basic":
            username = self.credentials.get("username")
            password = self.credentials.get("password")
            if username and password:
                return (username, password)
        return None


class ServiceCapabilities:
    """Enumeration of common service capabilities"""
    FILE_LISTING = "file_listing"
    FILE_DOWNLOAD = "file_download"
    FILE_SEARCH = "file_search"
    FILE_UPLOAD = "file_upload"
    MESSAGE_SEARCH = "message_search"
    TEAM_LISTING = "team_listing"
    CHANNEL_SEARCH = "channel_search"
    USER_SEARCH = "user_search"
    SHARING_MANAGEMENT = "sharing_management"
    SYSTEM_INFO = "system_info"
    HEALTH_CHECK = "health_check"


class AuthMethod:
    """Enumeration of authentication methods"""
    NONE = "none"
    BASIC = "basic"
    TOKEN = "token"
    OAUTH = "oauth"
    CUSTOM = "custom"