#!/usr/bin/env python3
"""
Performance Monitoring Script for M1 MacBook Air RAG System
Monitors system resources, query performance, and provides optimization recommendations
"""

import time
import psutil
import torch
import gc
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading

@dataclass
class PerformanceMetrics:
    timestamp: str
    memory_usage_gb: float
    memory_percentage: float
    cpu_percentage: float
    query_response_time: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    documents_processed: Optional[int] = None
    relevance_score: Optional[float] = None
    device_used: Optional[str] = None

class M1PerformanceMonitor:
    def __init__(self, log_file: str = "logs/performance_monitor.json"):
        self.log_file = log_file
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self.query_times = deque(maxlen=100)  # Keep last 100 query times
        self.cache_stats = {"hits": 0, "misses": 0}
        self.monitoring = False
        self.monitor_thread = None
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Load existing metrics if available
        self._load_existing_metrics()

    def _load_existing_metrics(self):
        """Load existing performance metrics from file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    for metric_data in data.get('metrics', [])[-100:]:  # Load last 100
                        metric = PerformanceMetrics(**metric_data)
                        self.metrics_history.append(metric)
                    
                    self.cache_stats = data.get('cache_stats', {"hits": 0, "misses": 0})
                print(f"📊 Loaded {len(self.metrics_history)} existing performance metrics")
        except Exception as e:
            print(f"⚠️  Could not load existing metrics: {e}")

    def start_monitoring(self, interval: int = 30):
        """Start continuous system monitoring"""
        if self.monitoring:
            print("⚠️  Monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._continuous_monitor, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print(f"🔍 Started continuous monitoring (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("⏹️  Stopped monitoring")

    def _continuous_monitor(self, interval: int):
        """Continuous monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(interval)
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(interval)

    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Determine device being used
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            memory_usage_gb=memory.used / (1024**3),
            memory_percentage=memory.percent,
            cpu_percentage=cpu_percent,
            device_used=device
        )

    def record_query_performance(self, response_time: float, relevance_score: float, 
                                docs_processed: int, cache_hit: bool = False):
        """Record performance metrics for a query"""
        # Update cache stats
        if cache_hit:
            self.cache_stats["hits"] += 1
        else:
            self.cache_stats["misses"] += 1
        
        # Record query time
        self.query_times.append(response_time)
        
        # Create metrics with query data
        base_metrics = self._collect_system_metrics()
        base_metrics.query_response_time = response_time
        base_metrics.relevance_score = relevance_score
        base_metrics.documents_processed = docs_processed
        base_metrics.cache_hit_rate = self._calculate_cache_hit_rate()
        
        self.metrics_history.append(base_metrics)
        
        # Auto-save periodically
        if len(self.metrics_history) % 10 == 0:
            self._save_metrics()

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total == 0:
            return 0.0
        return self.cache_stats["hits"] / total

    def get_performance_summary(self, hours: int = 1) -> Dict:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        # Calculate statistics
        memory_usage = [m.memory_usage_gb for m in recent_metrics]
        cpu_usage = [m.cpu_percentage for m in recent_metrics]
        query_times = [m.query_response_time for m in recent_metrics if m.query_response_time]
        relevance_scores = [m.relevance_score for m in recent_metrics if m.relevance_score]
        
        summary = {
            "time_period_hours": hours,
            "total_queries": len([m for m in recent_metrics if m.query_response_time]),
            "memory_stats": {
                "avg_gb": sum(memory_usage) / len(memory_usage),
                "max_gb": max(memory_usage),
                "min_gb": min(memory_usage)
            },
            "cpu_stats": {
                "avg_percent": sum(cpu_usage) / len(cpu_usage),
                "max_percent": max(cpu_usage),
                "min_percent": min(cpu_usage)
            },
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "device_used": recent_metrics[-1].device_used if recent_metrics else "unknown"
        }
        
        if query_times:
            summary["query_performance"] = {
                "avg_response_time": sum(query_times) / len(query_times),
                "max_response_time": max(query_times),
                "min_response_time": min(query_times)
            }
        
        if relevance_scores:
            summary["relevance_stats"] = {
                "avg_score": sum(relevance_scores) / len(relevance_scores),
                "max_score": max(relevance_scores),
                "min_score": min(relevance_scores)
            }
        
        return summary

    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on performance data"""
        recommendations = []
        
        if not self.metrics_history:
            return ["No performance data available for recommendations"]
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics
        
        # Memory recommendations
        avg_memory = sum(m.memory_percentage for m in recent_metrics) / len(recent_metrics)
        if avg_memory > 85:
            recommendations.append("🔴 HIGH MEMORY USAGE: Consider reducing MAX_DOCS_FOR_RETRIEVAL or CONTEXT_CHUNK_SIZE")
        elif avg_memory > 70:
            recommendations.append("🟡 MODERATE MEMORY USAGE: Monitor memory usage and close unnecessary applications")
        else:
            recommendations.append("🟢 MEMORY USAGE: Optimal")
        
        # CPU recommendations
        avg_cpu = sum(m.cpu_percentage for m in recent_metrics) / len(recent_metrics)
        if avg_cpu > 80:
            recommendations.append("🔴 HIGH CPU USAGE: Consider reducing batch sizes or thread count")
        elif avg_cpu > 60:
            recommendations.append("🟡 MODERATE CPU USAGE: Performance is acceptable")
        else:
            recommendations.append("🟢 CPU USAGE: Optimal")
        
        # Query performance recommendations
        query_times = [m.query_response_time for m in recent_metrics if m.query_response_time]
        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            if avg_query_time > 10:
                recommendations.append("🔴 SLOW QUERIES: Consider using smaller embedding models or reducing context size")
            elif avg_query_time > 5:
                recommendations.append("🟡 MODERATE QUERY SPEED: Consider enabling more aggressive caching")
            else:
                recommendations.append("🟢 QUERY SPEED: Optimal")
        
        # Cache recommendations
        cache_hit_rate = self._calculate_cache_hit_rate()
        if cache_hit_rate < 0.2:
            recommendations.append("🟡 LOW CACHE HIT RATE: Consider increasing cache size or TTL")
        elif cache_hit_rate > 0.5:
            recommendations.append("🟢 CACHE PERFORMANCE: Good hit rate")
        
        # Device recommendations
        if recent_metrics and recent_metrics[-1].device_used == "cpu":
            recommendations.append("🟡 DEVICE: Using CPU - ensure MPS is properly configured for M1 acceleration")
        elif recent_metrics and recent_metrics[-1].device_used == "mps":
            recommendations.append("🟢 DEVICE: Using MPS acceleration - optimal for M1")
        
        return recommendations

    def _save_metrics(self):
        """Save metrics to file"""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "cache_stats": self.cache_stats,
                "metrics": [asdict(m) for m in self.metrics_history]
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Failed to save metrics: {e}")

    def print_current_status(self):
        """Print current system status"""
        current_metrics = self._collect_system_metrics()
        
        print("\n" + "="*50)
        print("📊 M1 RAG System Performance Status")
        print("="*50)
        print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Memory: {current_metrics.memory_usage_gb:.1f}GB ({current_metrics.memory_percentage:.1f}%)")
        print(f"🖥️  CPU: {current_metrics.cpu_percentage:.1f}%")
        print(f"⚡ Device: {current_metrics.device_used}")
        print(f"📈 Cache Hit Rate: {self._calculate_cache_hit_rate():.1%}")
        
        if self.query_times:
            avg_query_time = sum(self.query_times) / len(self.query_times)
            print(f"⏱️  Avg Query Time: {avg_query_time:.2f}s")
        
        print("\n🎯 Optimization Recommendations:")
        for rec in self.get_optimization_recommendations():
            print(f"   {rec}")
        
        print("="*50)

    def export_performance_report(self, filename: str = None):
        """Export detailed performance report"""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_info": {
                "platform": psutil.platform,
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "torch_version": torch.__version__,
                "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            },
            "performance_summary_1h": self.get_performance_summary(1),
            "performance_summary_24h": self.get_performance_summary(24),
            "optimization_recommendations": self.get_optimization_recommendations(),
            "cache_statistics": self.cache_stats,
            "recent_metrics": [asdict(m) for m in list(self.metrics_history)[-50:]]  # Last 50 metrics
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Performance report exported to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Failed to export report: {e}")
            return None

    def cleanup_old_metrics(self, days: int = 7):
        """Clean up metrics older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        original_count = len(self.metrics_history)
        self.metrics_history = deque([
            m for m in self.metrics_history 
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ], maxlen=1000)
        
        cleaned_count = original_count - len(self.metrics_history)
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} old metrics (older than {days} days)")
            self._save_metrics()

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_monitoring()
        self._save_metrics()

# Global monitor instance
monitor = M1PerformanceMonitor()

def start_performance_monitoring():
    """Start performance monitoring"""
    monitor.start_monitoring()

def stop_performance_monitoring():
    """Stop performance monitoring"""
    monitor.stop_monitoring()

def record_query_metrics(response_time: float, relevance_score: float, 
                        docs_processed: int, cache_hit: bool = False):
    """Record query performance metrics"""
    monitor.record_query_performance(response_time, relevance_score, docs_processed, cache_hit)

def print_performance_status():
    """Print current performance status"""
    monitor.print_current_status()

def export_performance_report():
    """Export performance report"""
    return monitor.export_performance_report()

if __name__ == "__main__":
    # Interactive performance monitoring
    print("🚀 M1 RAG System Performance Monitor")
    print("Commands: status, start, stop, report, recommendations, export, quit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "status":
                monitor.print_current_status()
            elif command == "start":
                monitor.start_monitoring()
            elif command == "stop":
                monitor.stop_monitoring()
            elif command == "report":
                summary = monitor.get_performance_summary(1)
                print(json.dumps(summary, indent=2))
            elif command == "recommendations":
                recs = monitor.get_optimization_recommendations()
                for rec in recs:
                    print(f"  {rec}")
            elif command == "export":
                monitor.export_performance_report()
            elif command in ["quit", "exit", "q"]:
                monitor.stop_monitoring()
                break
            else:
                print("Unknown command. Available: status, start, stop, report, recommendations, export, quit")
                
        except KeyboardInterrupt:
            print("\n👋 Stopping monitor...")
            monitor.stop_monitoring()
            break
        except Exception as e:
            print(f"❌ Error: {e}")
