"""Service discovery with caching capabilities"""
import json
import os
import time
from typing import Any, Dict, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)


class DiscoveryCache:
    """Manages caching of service discovery results"""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl: int = 3600, use_file_cache: bool = False):
        """
        Initialize discovery cache
        
        Args:
            cache_dir: Directory to store cache files (only used if use_file_cache=True)
            ttl: Time to live in seconds (default: 1 hour)
            use_file_cache: Whether to use file-based cache (default: False, uses memory)
        """
        self.ttl = ttl
        self.use_file_cache = use_file_cache
        
        # In-memory cache
        self._memory_cache = {}
        
        # File-based cache setup
        if use_file_cache:
            self.cache_dir = cache_dir or ".cache/enterprise_discovery"
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.cache_dir = None
        
    def get_cache_key(self, service_name: str, config: Dict[str, Any]) -> str:
        """Generate a unique cache key for a service configuration"""
        # Create a deterministic key from service name and config
        config_str = json.dumps(config, sort_keys=True)
        hash_obj = hashlib.md5(f"{service_name}:{config_str}".encode())
        return hash_obj.hexdigest()
    
    def get(self, service_name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached discovery results if valid
        
        Args:
            service_name: Name of the service
            config: Service configuration
            
        Returns:
            Cached discovery results or None if cache miss/expired
        """
        cache_key = self.get_cache_key(service_name, config)
        
        if self.use_file_cache:
            return self._get_from_file(service_name, cache_key)
        else:
            return self._get_from_memory(service_name, cache_key)
    
    def _get_from_memory(self, service_name: str, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get from in-memory cache"""
        full_key = f"{service_name}_{cache_key}"
        
        if full_key not in self._memory_cache:
            logger.debug(f"Memory cache miss for {service_name}")
            return None
        
        cached_data = self._memory_cache[full_key]
        
        # Check if cache is expired
        if time.time() - cached_data.get("timestamp", 0) > self.ttl:
            logger.debug(f"Memory cache expired for {service_name}")
            del self._memory_cache[full_key]
            return None
        
        logger.info(f"Memory cache hit for {service_name}")
        return cached_data.get("discovery_results")
    
    def _get_from_file(self, service_name: str, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get from file-based cache"""
        cache_file = os.path.join(self.cache_dir, f"{service_name}_{cache_key}.json")
        
        if not os.path.exists(cache_file):
            logger.debug(f"File cache miss for {service_name}")
            return None
            
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                
            # Check if cache is expired
            if time.time() - cached_data.get("timestamp", 0) > self.ttl:
                logger.debug(f"File cache expired for {service_name}")
                os.remove(cache_file)
                return None
                
            logger.info(f"File cache hit for {service_name}")
            return cached_data.get("discovery_results")
            
        except Exception as e:
            logger.error(f"Error reading file cache for {service_name}: {e}")
            return None
    
    def set(self, service_name: str, config: Dict[str, Any], discovery_results: Dict[str, Any]):
        """
        Cache discovery results
        
        Args:
            service_name: Name of the service
            config: Service configuration
            discovery_results: Results to cache
        """
        cache_key = self.get_cache_key(service_name, config)
        
        cache_data = {
            "service_name": service_name,
            "timestamp": time.time(),
            "discovery_results": discovery_results
        }
        
        if self.use_file_cache:
            self._set_to_file(service_name, cache_key, cache_data)
        else:
            self._set_to_memory(service_name, cache_key, cache_data)
    
    def _set_to_memory(self, service_name: str, cache_key: str, cache_data: Dict[str, Any]):
        """Set to in-memory cache"""
        full_key = f"{service_name}_{cache_key}"
        self._memory_cache[full_key] = cache_data
        logger.info(f"Cached discovery results in memory for {service_name}")
    
    def _set_to_file(self, service_name: str, cache_key: str, cache_data: Dict[str, Any]):
        """Set to file-based cache"""
        cache_file = os.path.join(self.cache_dir, f"{service_name}_{cache_key}.json")
        
        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached discovery results to file for {service_name}")
        except Exception as e:
            logger.error(f"Error caching discovery results for {service_name}: {e}")
    
    def invalidate(self, service_name: str = None):
        """
        Invalidate cache entries
        
        Args:
            service_name: Specific service to invalidate (None for all)
        """
        if self.use_file_cache:
            self._invalidate_file_cache(service_name)
        else:
            self._invalidate_memory_cache(service_name)
    
    def _invalidate_memory_cache(self, service_name: str = None):
        """Invalidate in-memory cache"""
        if service_name:
            # Remove all cache entries for this service
            keys_to_remove = [key for key in self._memory_cache.keys() if key.startswith(f"{service_name}_")]
            for key in keys_to_remove:
                del self._memory_cache[key]
            if keys_to_remove:
                logger.info(f"Invalidated memory cache for {service_name}")
        else:
            # Clear entire cache
            self._memory_cache.clear()
            logger.info("Invalidated entire memory cache")
    
    def _invalidate_file_cache(self, service_name: str = None):
        """Invalidate file-based cache"""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
            
        if service_name:
            # Remove all cache files for this service
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(f"{service_name}_"):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                        logger.info(f"Invalidated file cache for {service_name}")
                    except:
                        pass
        else:
            # Clear entire cache
            for filename in os.listdir(self.cache_dir):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except:
                    pass
            logger.info("Invalidated entire file cache")


class ServiceDiscovery:
    """Handles service discovery with caching"""
    
    def __init__(self, cache: Optional[DiscoveryCache] = None):
        self.cache = cache or DiscoveryCache()
        
    def discover_service(self, adapter, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Discover service capabilities with caching
        
        Args:
            adapter: Service adapter instance
            force_refresh: Force fresh discovery (bypass cache)
            
        Returns:
            Discovery results
        """
        # Check cache first unless forced refresh
        if not force_refresh:
            cached_results = self.cache.get(adapter.service_name, adapter.config)
            if cached_results:
                # Apply cached results to adapter
                adapter.capabilities = cached_results.get("capabilities", [])
                adapter.endpoints = cached_results.get("endpoints", {})
                adapter.auth_method = cached_results.get("auth_method", "none")
                adapter.credentials = cached_results.get("credentials", {})
                return cached_results
        
        # Perform fresh discovery
        logger.info(f"Performing fresh discovery for {adapter.service_name}")
        discovery_results = adapter.discover_capabilities()
        
        # Cache the results
        self.cache.set(adapter.service_name, adapter.config, discovery_results)
        
        return discovery_results