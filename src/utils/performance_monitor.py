import time
import logging
from typing import Dict, List
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.response_times = defaultdict(list)
        self.query_counts = defaultdict(int)
        self.cache_hit_rate = {"hits": 0, "misses": 0}
        
    def track_response_time(self, category: str, duration: float):
        """Track response time for a specific category"""
        self.response_times[category].append(duration)
        self.query_counts[category] += 1
        
    def track_cache_hit(self):
        """Track cache hit"""
        self.cache_hit_rate["hits"] += 1
        
    def track_cache_miss(self):
        """Track cache miss"""
        self.cache_hit_rate["misses"] += 1
        
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        stats = {
            "overall": self._calculate_overall_stats(),
            "by_category": self._calculate_category_stats(),
            "cache_performance": self._calculate_cache_stats()
        }
        return stats
    
    def _calculate_overall_stats(self) -> Dict:
        """Calculate overall performance statistics"""
        all_times = []
        for times in self.response_times.values():
            all_times.extend(times)
            
        if not all_times:
            return {"count": 0}
            
        return {
            "total_queries": len(all_times),
            "avg_response_time": statistics.mean(all_times),
            "median_response_time": statistics.median(all_times),
            "min_response_time": min(all_times),
            "max_response_time": max(all_times),
            "std_deviation": statistics.stdev(all_times) if len(all_times) > 1 else 0
        }
    
    def _calculate_category_stats(self) -> Dict:
        """Calculate performance statistics by category"""
        category_stats = {}
        
        for category, times in self.response_times.items():
            if times:
                category_stats[category] = {
                    "count": len(times),
                    "avg_response_time": statistics.mean(times),
                    "median_response_time": statistics.median(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times)
                }
        
        return category_stats
    
    def _calculate_cache_stats(self) -> Dict:
        """Calculate cache performance statistics"""
        total_requests = self.cache_hit_rate["hits"] + self.cache_hit_rate["misses"]
        
        if total_requests == 0:
            return {"hit_rate": 0, "total_requests": 0}
            
        hit_rate = (self.cache_hit_rate["hits"] / total_requests) * 100
        
        return {
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            "cache_hits": self.cache_hit_rate["hits"],
            "cache_misses": self.cache_hit_rate["misses"]
        }
    
    def log_performance_summary(self):
        """Log a summary of performance metrics"""
        stats = self.get_performance_stats()
        
        logger.info("=== Performance Summary ===")
        
        # Overall stats
        overall = stats["overall"]
        if overall.get("total_queries", 0) > 0:
            logger.info(f"Total Queries: {overall['total_queries']}")
            logger.info(f"Average Response Time: {overall['avg_response_time']:.2f}s")
            logger.info(f"Median Response Time: {overall['median_response_time']:.2f}s")
            logger.info(f"Response Time Range: {overall['min_response_time']:.2f}s - {overall['max_response_time']:.2f}s")
        
        # Cache performance
        cache = stats["cache_performance"]
        if cache["total_requests"] > 0:
            logger.info(f"Cache Hit Rate: {cache['hit_rate']}% ({cache['cache_hits']}/{cache['total_requests']})")
        
        # Category breakdown
        category_stats = stats["by_category"]
        if category_stats:
            logger.info("Performance by Category:")
            for category, cat_stats in category_stats.items():
                logger.info(f"  {category}: {cat_stats['count']} queries, avg {cat_stats['avg_response_time']:.2f}s")

class ResponseTimer:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: PerformanceMonitor, category: str):
        self.monitor = monitor
        self.category = category
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.track_response_time(self.category, duration)
