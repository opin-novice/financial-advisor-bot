"""
Configuration file for the Advanced RAG Feedback Loop system
"""
import os
from typing import Dict, Any

class RAGConfig:
    """Configuration class for RAG system with Advanced Feedback Loop"""
    
    def __init__(self):
        """Initialize configuration with default values and environment overrides"""
        
        # Basic RAG Configuration
        self.FAISS_INDEX_PATH = "faiss_index"
        self.EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
        self.GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.CACHE_TTL = 86400  # 24 hours
        
        # Retrieval Settings
        self.MAX_DOCS_FOR_RETRIEVAL = 12
        self.MAX_DOCS_FOR_CONTEXT = 5
        self.CONTEXT_CHUNK_SIZE = 1500
        
        # Cross-Encoder Re-ranking Configuration
        self.CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        self.RELEVANCE_THRESHOLD = 0.2
        
        # Advanced RAG Feedback Loop Configuration
        self.FEEDBACK_LOOP_CONFIG = {
            # Core Settings
            "enable_feedback_loop": self._get_bool_env("ENABLE_FEEDBACK_LOOP", True),
            "max_iterations": int(os.getenv("FEEDBACK_MAX_ITERATIONS", "3")),
            "relevance_threshold": float(os.getenv("FEEDBACK_RELEVANCE_THRESHOLD", "0.2")),
            "confidence_threshold": float(os.getenv("FEEDBACK_CONFIDENCE_THRESHOLD", "0.15")),
            "max_docs_retrieval": int(os.getenv("FEEDBACK_MAX_DOCS", "12")),
            
            # Refinement Strategies (in order of preference)
            "refinement_strategies": [
                "domain_expansion",
                "synonym_expansion", 
                "context_addition",
                "query_decomposition"
            ],
            
            # Performance Settings
            "enable_caching": self._get_bool_env("FEEDBACK_ENABLE_CACHING", True),
            "log_refinement_history": self._get_bool_env("FEEDBACK_LOG_HISTORY", True),
            
            # Quality Control
            "min_document_quality_score": float(os.getenv("FEEDBACK_MIN_DOC_QUALITY", "0.1")),
            "enable_answer_enhancement": self._get_bool_env("FEEDBACK_ENHANCE_ANSWERS", True),
            
            # Fallback Settings
            "fallback_to_traditional": self._get_bool_env("FEEDBACK_FALLBACK_TRADITIONAL", True),
            "traditional_on_failure": self._get_bool_env("FEEDBACK_TRADITIONAL_ON_FAILURE", True)
        }
        
        # Validation
        self._validate_config()
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _validate_config(self):
        """Validate configuration values"""
        # Only warn about missing API key, don't fail initialization
        if not self.GROQ_API_KEY:
            print("⚠️  GROQ_API_KEY environment variable not set - some features may not work")
        
        if self.FEEDBACK_LOOP_CONFIG["max_iterations"] < 1:
            raise ValueError("max_iterations must be at least 1")
        
        if not (0.0 <= self.FEEDBACK_LOOP_CONFIG["relevance_threshold"] <= 1.0):
            raise ValueError("relevance_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.FEEDBACK_LOOP_CONFIG["confidence_threshold"] <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
    
    def get_feedback_loop_config(self) -> Dict[str, Any]:
        """Get feedback loop configuration"""
        return self.FEEDBACK_LOOP_CONFIG.copy()
    
    def update_feedback_loop_config(self, updates: Dict[str, Any]):
        """Update feedback loop configuration"""
        self.FEEDBACK_LOOP_CONFIG.update(updates)
        self._validate_config()
    
    def disable_feedback_loop(self):
        """Disable the feedback loop (use traditional RAG)"""
        self.FEEDBACK_LOOP_CONFIG["enable_feedback_loop"] = False
    
    def enable_feedback_loop(self):
        """Enable the feedback loop"""
        self.FEEDBACK_LOOP_CONFIG["enable_feedback_loop"] = True
    
    def get_performance_mode_config(self, mode: str) -> Dict[str, Any]:
        """Get predefined configuration for different performance modes"""
        
        if mode == "fast":
            return {
                "max_iterations": 2,
                "relevance_threshold": 0.4,
                "confidence_threshold": 0.3,
                "refinement_strategies": ["domain_expansion", "synonym_expansion"]
            }
        
        elif mode == "balanced":
            return {
                "max_iterations": 3,
                "relevance_threshold": 0.2,
                "confidence_threshold": 0.15,
                "refinement_strategies": [
                    "domain_expansion", "synonym_expansion", "context_addition"
                ]
            }
        
        elif mode == "thorough":
            return {
                "max_iterations": 4,
                "relevance_threshold": 0.15,
                "confidence_threshold": 0.1,
                "refinement_strategies": [
                    "domain_expansion", "synonym_expansion", 
                    "context_addition", "query_decomposition"
                ]
            }
        
        else:
            raise ValueError(f"Unknown performance mode: {mode}")
    
    def set_performance_mode(self, mode: str):
        """Set configuration to a predefined performance mode"""
        mode_config = self.get_performance_mode_config(mode)
        self.update_feedback_loop_config(mode_config)
        print(f"[INFO] Feedback loop configured for '{mode}' mode")
    
    def print_config_summary(self):
        """Print a summary of current configuration"""
        print("\n" + "="*60)
        print("🔧 ADVANCED RAG FEEDBACK LOOP CONFIGURATION")
        print("="*60)
        
        print(f"📊 Status: {'✅ ENABLED' if self.FEEDBACK_LOOP_CONFIG['enable_feedback_loop'] else '❌ DISABLED'}")
        print(f"🔄 Max Iterations: {self.FEEDBACK_LOOP_CONFIG['max_iterations']}")
        print(f"🎯 Relevance Threshold: {self.FEEDBACK_LOOP_CONFIG['relevance_threshold']}")
        print(f"📈 Confidence Threshold: {self.FEEDBACK_LOOP_CONFIG['confidence_threshold']}")
        print(f"📚 Max Documents: {self.FEEDBACK_LOOP_CONFIG['max_docs_retrieval']}")
        
        print(f"\n🔧 Refinement Strategies:")
        for i, strategy in enumerate(self.FEEDBACK_LOOP_CONFIG['refinement_strategies'], 1):
            print(f"   {i}. {strategy.replace('_', ' ').title()}")
        
        print(f"\n⚙️  Advanced Features:")
        print(f"   - Caching: {'✅' if self.FEEDBACK_LOOP_CONFIG['enable_caching'] else '❌'}")
        print(f"   - History Logging: {'✅' if self.FEEDBACK_LOOP_CONFIG['log_refinement_history'] else '❌'}")
        print(f"   - Answer Enhancement: {'✅' if self.FEEDBACK_LOOP_CONFIG['enable_answer_enhancement'] else '❌'}")
        print(f"   - Traditional Fallback: {'✅' if self.FEEDBACK_LOOP_CONFIG['fallback_to_traditional'] else '❌'}")
        
        print("="*60 + "\n")

# Global configuration instance
config = RAGConfig()