import os
import logging
import time
from typing import Dict, List, Tuple, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from rag_utils import RAGUtils
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedRAGFeedbackLoop:
    """
    Advanced RAG Feedback Loop Implementation - DISABLED FOR RESEARCH PURPOSES
    
    This implements the core LegalRAG innovation:
    Query -> Retrieve -> [Check Relevance] -> (If Irrelevant: Refine Query & Repeat) -> Generate
    
    Instead of the traditional linear flow: Query -> Retrieve -> Generate
    
    NOTE: This feedback loop has been DISABLED for research purposes to ensure:
    1. Deterministic behavior for reproducible results
    2. Mathematically justified thresholds
    3. Clear, explainable methodology
    4. Compliance with academic research standards
    
    For research work, please use the Traditional RAG approach in _process_query_traditional method.
    """
    
    def __init__(self, vectorstore: FAISS, rag_utils: RAGUtils, config: Optional[Dict] = None):
        """
        Initialize the Advanced RAG Feedback Loop
        
        Args:
            vectorstore: FAISS vector store for document retrieval
            rag_utils: RAG utilities for relevance checking and query refinement
            config: Configuration parameters for the feedback loop
        """
        self.vectorstore = vectorstore
        self.rag_utils = rag_utils
        
        # Default configuration
        self.config = {
            "max_iterations": 3,           # Maximum refinement iterations
            "relevance_threshold": 0.3,    # Minimum relevance score to proceed
            "confidence_threshold": 0.2,   # Minimum confidence to proceed
            "max_docs_retrieval": 12,      # Documents to retrieve per iteration
            "enable_feedback_loop": True,  # Enable/disable feedback loop
            "refinement_strategies": [     # Different refinement strategies to try
                "domain_expansion",
                "synonym_expansion", 
                "context_addition",
                "query_decomposition"
            ]
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        logger.info(f"✅ Advanced RAG Feedback Loop initialized with config: {self.config}")
    
    def retrieve_with_feedback_loop(self, original_query: str, category: str = "general") -> Dict:
        """
        Main method implementing the Advanced RAG Feedback Loop
        
        Args:
            original_query: User's original query
            category: Query category for domain-specific processing
            
        Returns:
            Dict containing retrieved documents, final query used, and metadata
        """
        if not self.config["enable_feedback_loop"]:
            # Fallback to simple retrieval if feedback loop is disabled
            return self._simple_retrieval(original_query)
        
        logger.info(f"🔄 Starting Advanced RAG Feedback Loop for query: '{original_query}'")
        
        # Initialize tracking variables
        iteration = 0
        current_query = original_query
        best_result = None
        best_relevance_score = 0.0
        refinement_history = []
        
        while iteration < self.config["max_iterations"]:
            iteration += 1
            logger.info(f"🔍 Iteration {iteration}: Processing query: '{current_query}'")
            
            # Step 1: Retrieve documents
            retrieved_docs = self._retrieve_documents(current_query)
            
            if not retrieved_docs:
                logger.warning(f"❌ No documents retrieved for query: '{current_query}'")
                if iteration == 1:
                    # If first iteration fails, return empty result
                    return self._create_empty_result(original_query, "No documents found")
                else:
                    # Use best result from previous iterations
                    break
            
            # Step 2: Check relevance
            is_relevant, confidence_score = self.rag_utils.check_query_relevance(
                current_query, retrieved_docs
            )
            
            logger.info(f"📊 Relevance check - Relevant: {is_relevant}, Confidence: {confidence_score:.3f}")
            
            # Step 3: Evaluate if this is our best result so far
            if confidence_score > best_relevance_score:
                best_relevance_score = confidence_score
                best_result = {
                    "documents": retrieved_docs,
                    "query_used": current_query,
                    "relevance_score": confidence_score,
                    "is_relevant": is_relevant,
                    "iteration": iteration
                }
            
            # Step 4: Check if we should proceed or continue refining
            if (is_relevant and 
                confidence_score >= self.config["confidence_threshold"] and
                confidence_score >= self.config["relevance_threshold"]):
                
                logger.info(f"✅ Sufficient relevance achieved at iteration {iteration}")
                break
            
            # Step 5: Refine query for next iteration (if not last iteration)
            if iteration < self.config["max_iterations"]:
                refined_query = self._refine_query_strategically(
                    current_query, original_query, retrieved_docs, iteration, category
                )
                
                if refined_query == current_query:
                    logger.info("🔄 No further refinement possible, stopping iterations")
                    break
                
                refinement_history.append({
                    "iteration": iteration,
                    "original_query": current_query,
                    "refined_query": refined_query,
                    "relevance_score": confidence_score
                })
                
                current_query = refined_query
            
        # Return the best result found
        if best_result is None:
            return self._create_empty_result(original_query, "No relevant documents found after refinement")
        
        # Add metadata about the feedback loop process
        best_result.update({
            "original_query": original_query,
            "total_iterations": iteration,
            "refinement_history": refinement_history,
            "feedback_loop_used": True
        })
        
        logger.info(f"🎯 Feedback loop completed. Best result from iteration {best_result['iteration']} "
                   f"with relevance score: {best_result['relevance_score']:.3f}")
        
        return best_result
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve documents from vector store with cross-encoder reranking"""
        try:
            # Retrieve more candidates than needed for reranking
            candidate_count = self.config["max_docs_retrieval"] * 2
            initial_docs = self.vectorstore.similarity_search(
                query, 
                k=candidate_count
            )
            
            # If we have enough documents, apply cross-encoder reranking
            if len(initial_docs) > self.config["max_docs_retrieval"]:
                # Extract document contents for reranking
                doc_contents = [doc.page_content for doc in initial_docs]
                
                # Apply cross-encoder reranking
                reranker = CrossEncoderReranker()
                reranked = reranker.rerank(
                    query, 
                    doc_contents, 
                    top_k=self.config["max_docs_retrieval"]
                )
                
                # Reconstruct Document objects with reranked scores
                reranked_docs = []
                for content, score in reranked:
                    # Find original document and add score metadata
                    for doc in initial_docs:
                        if doc.page_content == content:
                            doc.metadata['rerank_score'] = float(score)
                            reranked_docs.append(doc)
                            break
                return reranked_docs
            else:
                return initial_docs
                
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _refine_query_strategically(self, current_query: str, original_query: str, 
                                  retrieved_docs: List[Document], iteration: int, 
                                  category: str) -> str:
        """
        Apply different refinement strategies based on iteration number
        """
        strategies = self.config["refinement_strategies"]
        
        if iteration > len(strategies):
            # If we've exhausted all strategies, use the basic refinement
            return self.rag_utils.refine_query(current_query)
        
        strategy = strategies[iteration - 1]
        logger.info(f"🔧 Applying refinement strategy: {strategy}")
        
        try:
            if strategy == "domain_expansion":
                return self._domain_expansion_refinement(current_query, category)
            elif strategy == "synonym_expansion":
                return self._synonym_expansion_refinement(current_query)
            elif strategy == "context_addition":
                return self._context_addition_refinement(current_query, retrieved_docs)
            elif strategy == "query_decomposition":
                return self._query_decomposition_refinement(current_query, original_query)
            else:
                # Fallback to basic refinement
                return self.rag_utils.refine_query(current_query)
                
        except Exception as e:
            logger.warning(f"Refinement strategy {strategy} failed: {e}")
            return self.rag_utils.refine_query(current_query)
    
    def _domain_expansion_refinement(self, query: str, category: str) -> str:
        """Add domain-specific terms to the query"""
        domain_terms = {
            "banking": ["bangladesh bank", "account opening", "deposit", "withdrawal"],
            "loans": ["credit", "interest rate", "emi", "collateral", "bangladesh"],
            "investment": ["savings certificate", "bond", "profit", "bangladesh"],
            "taxation": ["nbr", "income tax", "vat", "bangladesh tax"],
            "insurance": ["policy", "premium", "claim", "bangladesh insurance"],
            "business": ["startup", "registration", "license", "bangladesh business"]
        }
        
        terms = domain_terms.get(category, ["bangladesh", "financial services"])
        expanded_query = f"{query} {' '.join(terms[:2])}"  # Add top 2 terms
        return expanded_query
    
    def _synonym_expansion_refinement(self, query: str) -> str:
        """Expand query with financial synonyms"""
        synonyms = {
            "loan": "credit financing",
            "bank": "financial institution",
            "account": "banking service",
            "tax": "taxation revenue",
            "investment": "financial investment",
            "business": "enterprise"
        }
        
        words = query.lower().split()
        expanded = []
        
        for word in words:
            expanded.append(word)
            if word in synonyms:
                expanded.extend(synonyms[word].split())
        
        return " ".join(expanded)
    
    def _context_addition_refinement(self, query: str, retrieved_docs: List[Document]) -> str:
        """Add context from retrieved documents to refine the query"""
        if not retrieved_docs:
            return query
        
        # Extract key terms from the best retrieved document
        best_doc = retrieved_docs[0]
        doc_words = best_doc.page_content.lower().split()
        
        # Find relevant terms that aren't in the original query
        query_words = set(query.lower().split())
        relevant_terms = []
        
        financial_keywords = ["bangladesh", "bank", "account", "loan", "tax", "investment", 
                            "business", "service", "rate", "policy", "regulation"]
        
        for term in financial_keywords:
            if term in doc_words and term not in query_words:
                relevant_terms.append(term)
                if len(relevant_terms) >= 2:  # Limit to 2 additional terms
                    break
        
        if relevant_terms:
            return f"{query} {' '.join(relevant_terms)}"
        
        return query
    
    def _query_decomposition_refinement(self, current_query: str, original_query: str) -> str:
        """Break down complex queries into simpler components"""
        # If current query is too complex, try to simplify it
        if len(current_query.split()) > 8:
            # Extract key terms from original query
            original_words = original_query.lower().split()
            key_terms = []
            
            important_words = ["how", "what", "when", "where", "why", "bank", "loan", 
                             "tax", "account", "investment", "business", "open", "apply"]
            
            for word in original_words:
                if word in important_words or len(word) > 4:
                    key_terms.append(word)
                    if len(key_terms) >= 4:  # Limit to 4 key terms
                        break
            
            if key_terms:
                return " ".join(key_terms)
        
        return current_query
    
    def _simple_retrieval(self, query: str) -> Dict:
        """Fallback simple retrieval when feedback loop is disabled"""
        retrieved_docs = self._retrieve_documents(query)
        
        if not retrieved_docs:
            return self._create_empty_result(query, "No documents found")
        
        # Basic relevance check
        is_relevant, confidence_score = self.rag_utils.check_query_relevance(query, retrieved_docs)
        
        return {
            "documents": retrieved_docs,
            "query_used": query,
            "relevance_score": confidence_score,
            "is_relevant": is_relevant,
            "iteration": 1,
            "original_query": query,
            "total_iterations": 1,
            "refinement_history": [],
            "feedback_loop_used": False
        }
    
    def _create_empty_result(self, query: str, reason: str) -> Dict:
        """Create empty result structure"""
        return {
            "documents": [],
            "query_used": query,
            "relevance_score": 0.0,
            "is_relevant": False,
            "iteration": 1,
            "original_query": query,
            "total_iterations": 1,
            "refinement_history": [],
            "feedback_loop_used": True,
            "failure_reason": reason
        }
    
    def get_feedback_loop_stats(self) -> Dict:
        """Get statistics about feedback loop performance"""
        # This could be extended to track performance metrics
        return {
            "config": self.config,
            "status": "active" if self.config["enable_feedback_loop"] else "disabled"
        }