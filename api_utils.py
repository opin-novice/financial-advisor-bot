#!/usr/bin/env python3
"""
API utilities for handling rate limits and errors gracefully
"""

import time
import logging
from functools import wraps
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def handle_api_errors(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator to handle API rate limits gracefully
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Check if it's a rate limit error
                    if any(keyword in error_msg for keyword in ['rate_limit', 'too many requests', '429']):
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Rate limit hit, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(f"Max retries ({max_retries}) exceeded for rate limiting")
                            return {
                                "response": "I'm experiencing high demand right now. Please try again in a moment.", 
                                "sources": [],
                                "error": "rate_limit_exceeded"
                            }
                    
                    # Check if it's a payload too large error
                    elif '413' in error_msg or 'payload too large' in error_msg:
                        logger.error(f"Payload too large error: {e}")
                        return {
                            "response": "Your query is too complex. Please try asking a simpler or more specific question.", 
                            "sources": [],
                            "error": "payload_too_large"
                        }
                    
                    # For other errors, raise immediately
                    else:
                        logger.error(f"API error (non-retryable): {e}")
                        raise e
            
            # If we get here, we've exhausted retries
            logger.error(f"Function failed after {max_retries} retries. Last error: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator

class APIRateLimiter:
    """Simple rate limiter to prevent overwhelming APIs"""
    
    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests = max_requests_per_minute
        self.requests = []
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        # If we're at the limit, wait
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0]) + 1  # Wait until oldest request is >1min old
            logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
            # Clean up again after sleeping
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        # Record this request
        self.requests.append(now)

# Global rate limiter instance
rate_limiter = APIRateLimiter()

def with_rate_limiting(func):
    """Decorator to add rate limiting to API calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        rate_limiter.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper

def safe_api_call(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Safely call an API function with error handling
    
    Args:
        func: Function to call
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Dict containing response or error information
    """
    try:
        # Apply rate limiting
        rate_limiter.wait_if_needed()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # If result is already a dict with error handling, return as-is
        if isinstance(result, dict) and "error" in result:
            return result
        
        # Otherwise wrap in success response
        return {"success": True, "result": result}
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if any(keyword in error_msg for keyword in ['rate_limit', 'too many requests', '429']):
            return {
                "success": False,
                "error": "rate_limit",
                "message": "API rate limit exceeded. Please try again in a moment.",
                "response": "I'm experiencing high demand. Please try your query again in a few seconds."
            }
        elif '413' in error_msg or 'payload too large' in error_msg:
            return {
                "success": False,
                "error": "payload_too_large", 
                "message": "Query too complex",
                "response": "Your question is too complex. Please try asking something more specific."
            }
        else:
            logger.error(f"Unexpected API error: {e}")
            return {
                "success": False,
                "error": "unexpected",
                "message": str(e),
                "response": f"I encountered an error: {e}"
            }
