import json
import os
import hashlib
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ResponseCache:
    def __init__(self, cache_dir="cache", ttl=86400):  # Default TTL: 24 hours
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_index_path = os.path.join(cache_dir, "cache_index.json")
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict:
        """Load the cache index or create a new one if it doesn't exist"""
        if os.path.exists(self.cache_index_path):
            try:
                with open(self.cache_index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache index: {str(e)}")
                return {"entries": {}}
        return {"entries": {}}
    
    def _save_cache_index(self):
        """Save the cache index"""
        try:
            with open(self.cache_index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {str(e)}")
    
    def _generate_key(self, query: str) -> str:
        """Generate a unique key for the query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict]:
        """Get a cached response for the query if it exists and is not expired"""
        key = self._generate_key(query)
        
        if key not in self.cache_index["entries"]:
            return None
        
        entry = self.cache_index["entries"][key]
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        # Check if cache entry is expired
        if time.time() - entry["timestamp"] > self.ttl:
            logger.info(f"Cache entry expired for query: {query[:50]}...")
            return None
        
        # Check if cache file exists
        if not os.path.exists(cache_file):
            logger.warning(f"Cache file missing for key {key}")
            return None
        
        # Load cached response
        try:
            with open(cache_file, 'r') as f:
                cached_response = json.load(f)
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_response
        except Exception as e:
            logger.error(f"Error loading cached response: {str(e)}")
            return None
    
    def set(self, query: str, response: Dict):
        """Cache a response for the query"""
        key = self._generate_key(query)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        # Create a serializable copy of the response
        try:
            serializable_response = self._make_serializable(response)
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_response, f, indent=2)
            
            # Update cache index
            self.cache_index["entries"][key] = {
                "query": query,
                "timestamp": time.time(),
                "file": f"{key}.json"
            }
            self._save_cache_index()
            logger.info(f"Cached response for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, dict):
            serializable = {}
            for key, value in obj.items():
                if key == 'source_documents':
                    # Convert Document objects to dictionaries
                    serializable[key] = [{
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    } for doc in value]
                else:
                    serializable[key] = self._make_serializable(value)
            return serializable
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def clear_expired(self):
        """Clear expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache_index["entries"].items():
            if current_time - entry["timestamp"] > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except Exception as e:
                    logger.error(f"Error removing cache file {cache_file}: {str(e)}")
            
            del self.cache_index["entries"][key]
        
        if expired_keys:
            self._save_cache_index()
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")