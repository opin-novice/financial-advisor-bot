import os
import re
import logging
import time
from typing import Dict, List
from langchain.schema import Document

# Import RAG utilities and language detection
from rag_utils import RAGUtils
from advanced_rag_feedback import AdvancedRAGFeedbackLoop
from config import config
from language_utils import LanguageDetector, BilingualResponseFormatter
from document_retriever import DocumentRetriever
from document_ranker import DocumentRanker
from response_generator import ResponseGenerator

# --- Logging ---
logging.basicConfig(
    filename='logs/telegram_financial_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Cache ---
class ResponseCache:
    def __init__(self, ttl: int = config.CACHE_TTL):
        self.cache = {}
        self.ttl = ttl

    def get(self, key: str):
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value):
        self.cache[key] = (time.time(), value)

# --- Query Processing ---
class QueryProcessor:
    def process(self, query: str) -> str:
        return "general"

class FinancialAdvisorBotCore:
    """Core orchestrator for the financial advisor bot"""
    
    def __init__(self):
        self.cache = ResponseCache()
        self.processor = QueryProcessor()
        self.rag_utils = RAGUtils()  # Initialize RAG utilities
        self.feedback_loop = None  # Will be initialized after RAG setup
        
        # Initialize language detection components
        self.language_detector = LanguageDetector()
        self.response_formatter = BilingualResponseFormatter(self.language_detector)
        
        # Initialize document handling components
        self.retriever = DocumentRetriever()
        self.ranker = DocumentRanker()
        self.response_generator = ResponseGenerator(self.language_detector)
        
        # Enable hybrid retrieval
        self.retriever.enable_hybrid_retrieval()
        
        # Initialize Advanced RAG Feedback Loop
        print("[INFO] Initializing Advanced RAG Feedback Loop...")
        try:
            feedback_config = config.get_feedback_loop_config()
            # The feedback loop expects a vectorstore with a similarity_search method.
            # We will pass our retriever instance itself, as it now has that method.
            self.feedback_loop = AdvancedRAGFeedbackLoop(
                vectorstore=self.retriever,
                rag_utils=self.rag_utils,
                config=feedback_config
            )
            print("[INFO] ✅ Advanced RAG Feedback Loop initialized successfully.")
            
            if feedback_config.get("enable_feedback_loop", True):
                config.print_config_summary()
            else:
                print("[INFO] ⚠️  Advanced RAG Feedback Loop is DISABLED - using traditional RAG")
                
        except Exception as e:
            print(f"[WARNING] Failed to initialize Advanced RAG Feedback Loop: {e}")
            print("[INFO] Falling back to traditional RAG approach...")
            self.feedback_loop = None

    def process_query(self, query: str) -> Dict:
        """
        Enhanced process_query with translation-based multilingual support
        
        RESEARCH NOTE: Advanced RAG Feedback Loop has been DISABLED for research purposes.
        This ensures:
        1. Deterministic, reproducible results
        2. Mathematically justified methodology
        3. Compliance with academic research standards
        4. Clear, explainable components
        
        Using Traditional RAG approach for all queries.
        """
        # Detect language of the query
        detected_language, confidence = self.language_detector.detect_language(query)
        print(f"[INFO] 🌐 Language detected: {detected_language} (confidence: {confidence:.2f})")
        
        # Store original query and language for response formatting
        original_query = query
        original_language = detected_language
        
        # Get configuration for translation approach
        feedback_config = config.get_feedback_loop_config()
        use_translation = feedback_config.get("use_translation_approach", True)
        enable_query_translation = feedback_config.get("enable_query_translation", True)
        
        # If query is in Bangla and translation is enabled, translate to English for processing
        if detected_language == 'bengali' and use_translation and enable_query_translation:
            print(f"[INFO] 🔄 Translating Bangla query to English...")
            query = self.language_detector.translate_bangla_to_english(query)
            print(f"[INFO] ✅ Translated query: {query}")
        
        category = self.processor.process(query)
        logger.info(f"Processing query (category={category}, original_language={original_language}): {query}")
        print(f"[INFO] 🔍 Processing query: {query}")

        # Create cache key with original language and query
        cache_key = f"{original_language}:{original_query}"
        cached = self.cache.get(cache_key)
        if cached:
            print("[INFO] ✅ Cache hit - returning stored response.")
            # Add language context to cached response
            cached['detected_language'] = original_language
            cached['language_confidence'] = confidence
            return cached

        try:
            # Use Traditional RAG approach for research purposes (feedback loop DISABLED)
            # Research Note: Advanced RAG Feedback Loop has been disabled to ensure:
            # 1. Deterministic behavior for reproducible results
            # 2. Mathematically justified thresholds
            # 3. Clear, explainable methodology
            # 4. Compliance with academic research standards
            print("[INFO] 🔄 Using Traditional RAG approach (Feedback Loop DISABLED for Research)")
            result = self.response_generator.process_query_traditional(
                query, category, original_language, self.retriever, self.ranker, 
                use_remote_faiss=False)
            
            # Add language detection information to result
            result['detected_language'] = original_language
            result['language_confidence'] = confidence
            result['was_translated'] = (original_language == 'bengali' and use_translation and enable_query_translation)
            result['translated_query'] = query if result['was_translated'] else None
            
            # Cache the result with language-aware key
            self.cache.set(cache_key, result)
            
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"[ERROR] {e}")
            
            # Return error message in original language
            error_message = self.language_detector.translate_system_messages(
                f"Error: {e}", original_language
            )
            
            return {
                "response": error_message, 
                "sources": [], 
                "contexts": [],
                "detected_language": original_language,
                "language_confidence": confidence
            }
    
    def update_index_delta(self, documents: List[Document]) -> bool:
        """
        Update the FAISS index with delta changes
        
        Args:
            documents: List of Document objects to check for changes
            
        Returns:
            True if changes were applied, False if no changes detected
        """
        print("[INFO] 🔄 Updating index with delta changes...")
        return self.retriever.update_index_delta(documents)
    
    def get_index_statistics(self) -> Dict:
        """
        Get statistics about the current index state
        
        Returns:
            Dictionary with index statistics
        """
        return self.retriever.get_index_statistics()