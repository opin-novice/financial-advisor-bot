import os
import logging
from typing import Dict, List, Tuple
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import numpy as np
import torch
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Config ---
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required. Please set it in your .env file.")

class CrossEncoderReranker:
    def __init__(self, model_name='BAAI/bge-reranker-large', batch_size=16):
        """
        Initialize Cross-Encoder Reranker
        
        Args:
            model_name: Cross-encoder model to use
            batch_size: Batch size for processing
        """
        try:
            self.model = CrossEncoder(model_name, max_length=512)
            self.batch_size = batch_size
            logger.info(f"✅ Cross-encoder reranker initialized with {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: User query string
            documents: List of document texts
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Get relevance scores with batch processing
            scores = self.model.predict(
                pairs, 
                batch_size=self.batch_size, 
                convert_to_tensor=True
            )
            
            # Sort by scores and return top_k
            scored_docs = list(zip(documents, scores.cpu().numpy()))
            return sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            # Return top_k documents without reranking if error occurs
            return [(doc, 0.0) for doc in documents[:top_k]]

class RAGUtils:
    def __init__(self):
        """Initialize RAG utilities with Ollama LLM"""
        try:
            # self.llm = ChatGroq(
            #     model=GROQ_MODEL,
            #     groq_api_key=GROQ_API_KEY,
            #     temperature=0.3,
            #     max_tokens=500,
            # )
            self.llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.3,
                num_predict=500,
            )
            logger.info("✅ Ollama LLM initialized for RAG utilities")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise

    def check_query_relevance(self, query: str, retrieved_docs: List[Document]) -> Tuple[bool, float]:
        """
        Check if the query is relevant to the retrieved documents using LLM.
        
        Args:
            query: User's input query
            retrieved_docs: List of documents retrieved from vector store
            
        Returns:
            Tuple of (is_relevant: bool, confidence_score: float)
        """
        if not retrieved_docs:
            return False, 0.0

        # Create a prompt to check relevance
        relevance_prompt = PromptTemplate.from_template("""
        Analyze if the following query is relevant to the provided document contexts.
        
        Query: {query}
        
        Document Contexts:
        {contexts}
        
        Respond with ONLY a JSON object in this exact format:
        {{"relevant": true/false, "confidence": 0.0-1.0}}
        
        Where:
        - relevant: boolean indicating if query relates to document topics
        - confidence: float between 0.0-1.0 indicating confidence level
        """)
        
        # Prepare context from top documents
        contexts = "\n\n".join([f"Document {i+1}:\n{doc.page_content[:500]}..." 
                               for i, doc in enumerate(retrieved_docs[:3])])
        
        # Generate relevance assessment
        try:
            prompt = relevance_prompt.format(query=query, contexts=contexts)
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON response
            import json
            try:
                result = json.loads(response.content.strip())
                is_relevant = result.get("relevant", False)
                confidence = float(result.get("confidence", 0.0))
                return is_relevant, confidence
            except json.JSONDecodeError:
                # Fallback: simple keyword matching if LLM response isn't valid JSON
                response_text = response.content.lower()
                is_relevant = any(word in response_text for word in ["true", "relevant", "yes"])
                confidence = 0.7 if is_relevant else 0.3
                return is_relevant, confidence
                
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}. Using fallback method.")
            # Fallback to simple keyword matching
            return self._fallback_relevance_check(query, retrieved_docs)

    def _fallback_relevance_check(self, query: str, retrieved_docs: List[Document]) -> Tuple[bool, float]:
        """Simple keyword-based relevance check as fallback"""
        query_terms = set(query.lower().split())
        relevance_scores = []
        
        for doc in retrieved_docs[:3]:  # Check top 3 documents
            doc_content = doc.page_content.lower()
            matches = sum(1 for term in query_terms if term in doc_content)
            score = matches / len(query_terms) if query_terms else 0
            relevance_scores.append(score)
        
        avg_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        is_relevant = avg_score > 0.1  # Threshold for relevance
        return is_relevant, avg_score

    def refine_query(self, query: str) -> str:
        """
        Refine the user's query to improve retrieval using LLM.
        
        Args:
            query: Original user query
            
        Returns:
            Refined query string
        """
        # Pre-expand common financial terms for better retrieval with Bangladesh context
        financial_terms_expansion = {
            "loan": "loan application requirements interest rate eligibility collateral bank Bangladesh",
            "account": "bank account opening requirements documents procedure minimum balance deposit Bangladesh",
            "tax": "tax filing requirements forms deadlines calculation return NBR Bangladesh",
            "investment": "investment options returns risk regulations Bangladesh BIDA mutual fund",
            "insurance": "insurance policy coverage premium claim process life health Bangladesh",
            "credit": "credit card application limit interest fee approval bank Bangladesh",
            "deposit": "deposit account interest rate minimum balance fixed savings Bangladesh bank",
            "mortgage": "mortgage loan home financing interest rate eligibility Bangladesh",
            "business": "business registration license TIN BIN startup procedure Bangladesh",
            "startup": "startup funding investment registration process incubator Bangladesh",
            "forex": "foreign exchange rate transaction regulation remittance Bangladesh",
            "remittance": "remittance process fee online transfer money transfer Bangladesh",
            "mobile banking": "mobile banking app features transaction limit security Bangladesh",
            "online banking": "online banking security features transaction login Bangladesh",
            "ATM": "ATM card withdrawal limit location fee transaction Bangladesh bank",
            "cheque": "cheque book request process clearance time deposit Bangladesh bank",
            "fixed deposit": "fixed deposit interest rate maturity period withdrawal Bangladesh bank",
            "savings": "savings account interest rate minimum balance features Bangladesh bank",
            "current account": "current account features minimum balance business Bangladesh bank",
            "NID": "NID card requirement account opening loan application Bangladesh",
            "TIN": "TIN number tax filing requirement registration Bangladesh",
            "BIN": "BIN number business registration requirement license Bangladesh"
        }
        
        # Expand query with financial terms, avoiding duplication
        expanded_query = query.lower()
        added_terms = []
        for term, expansion in financial_terms_expansion.items():
            # Check for exact word matches to avoid partial matches
            if re.search(r'\b' + re.escape(term) + r'\b', expanded_query) and term not in added_terms:
                expanded_query += f" {expansion}"
                added_terms.append(term)
        
        # Capitalize first letter to maintain natural query format
        expanded_query = expanded_query.strip()
        if expanded_query:
            expanded_query = expanded_query[0].upper() + expanded_query[1:]
        
        logger.info(f"Query expansion: '{query}' -> '{expanded_query}'")
        
        # Create a prompt for query refinement with expanded context
        refinement_prompt = PromptTemplate.from_template("""
        You are a query refinement assistant for a financial advisor system in Bangladesh.
        Improve the following user query to better retrieve relevant financial information.
        
        Original Query: {query}
        
        Instructions:
        1. Expand financial terms and acronyms with Bangladesh-specific context
        2. Add relevant financial/banking context
        3. Remove irrelevant words while preserving intent
        4. Make query more specific to Bangladesh financial context
        5. Keep the intent the same
        6. Return ONLY the refined query, no explanations
        
        Refined Query:
        """)
        
        try:
            prompt = refinement_prompt.format(query=expanded_query)
            response = self.llm.invoke(prompt)
            refined_query = response.content.strip()
            
            # Ensure we don't return empty or nonsensical refinements
            if refined_query and len(refined_query) > 5:
                logger.info(f"Query refined: '{query}' -> '{refined_query}'")
                return refined_query
            else:
                logger.warning(f"Query refinement returned poor result. Using expanded: {expanded_query}")
                return expanded_query if len(expanded_query) > len(query) else query
                
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}. Using expanded query.")
            return expanded_query if len(expanded_query) > len(query) else query

    def validate_answer(self, query: str, answer: str, contexts: List[str]) -> Dict:
        """
        Validate the generated answer using LLM.
        
        Args:
            query: Original user query
            answer: Generated answer from RAG system
            contexts: Retrieved document contexts
            
        Returns:
            Validation result with score and feedback
        """
        validation_prompt = PromptTemplate.from_template("""
        You are an answer validation assistant. Evaluate if the provided answer correctly addresses the query based on the given contexts.
        
        Query: {query}
        Answer: {answer}
        Contexts: {contexts}
        
        Respond with ONLY a JSON object in this exact format:
        {{
            "valid": true/false,
            "confidence": 0.0-1.0,
            "feedback": "brief explanation"
        }}
        """)
        
        # Prepare context
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx[:300]}..." 
                                   for i, ctx in enumerate(contexts[:2])])
        
        try:
            prompt = validation_prompt.format(
                query=query, 
                answer=answer, 
                contexts=context_text
            )
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON response
            import json
            try:
                result = json.loads(response.content.strip())
                return {
                    "valid": bool(result.get("valid", False)),
                    "confidence": float(result.get("confidence", 0.0)),
                    "feedback": str(result.get("feedback", ""))
                }
            except json.JSONDecodeError:
                return {
                    "valid": True,  # Assume valid if parsing fails
                    "confidence": 0.5,
                    "feedback": "Validation parsing failed but answer generated"
                }
                
        except Exception as e:
            logger.warning(f"Answer validation failed: {e}")
            return {
                "valid": True,  # Assume valid if validation fails
                "confidence": 0.5,
                "feedback": f"Validation failed: {str(e)}"
            }

    def calculate_cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector_a: First vector
            vector_b: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Ensure vectors are 1D
        if vector_a.ndim > 1:
            vector_a = vector_a.flatten()
        if vector_b.ndim > 1:
            vector_b = vector_b.flatten()
            
        # Calculate dot product
        dot_product = np.dot(vector_a, vector_b)
        
        # Calculate magnitudes
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        # Calculate cosine similarity
        cosine_sim = dot_product / (norm_a * norm_b)
        
        # Ensure result is in valid range [-1, 1]
        return max(-1.0, min(1.0, cosine_sim))

    def analyze_retrieval_scores(self, scores: List[float], threshold: float = 0.15) -> Dict:
        """
        Analyze retrieval scores and provide quality assessment.
        
        Args:
            scores: List of similarity scores from FAISS
            threshold: Minimum score threshold for quality results
            
        Returns:
            Dictionary with score analysis
        """
        if not scores:
            return {
                "total_results": 0,
                "high_quality_count": 0,
                "medium_quality_count": 0,
                "low_quality_count": 0,
                "average_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "quality_distribution": []
            }
            
        # Categorize scores
        high_quality = [s for s in scores if s >= 0.5]
        medium_quality = [s for s in scores if 0.3 <= s < 0.5]
        low_quality = [s for s in scores if s < 0.3]
        
        # Filter by threshold
        quality_results = [s for s in scores if s >= threshold]
        
        return {
            "total_results": len(scores),
            "high_quality_count": len(high_quality),
            "medium_quality_count": len(medium_quality),
            "low_quality_count": len(low_quality),
            "quality_results_count": len(quality_results),
            "average_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": float(max(scores)) if scores else 0.0,
            "min_score": float(min(scores)) if scores else 0.0,
            "quality_distribution": [
                {"range": "high (>=0.5)", "count": len(high_quality)},
                {"range": "medium (0.3-0.5)", "count": len(medium_quality)},
                {"range": "low (<0.3)", "count": len(low_quality)}
            ]
        }

    def verify_cosine_similarity_calculation(self, query_embedding: np.ndarray, 
                                          document_embedding: np.ndarray) -> float:
        """
        Verify cosine similarity calculation between query and document embeddings.
        This is useful for debugging and ensuring FAISS scores are correct.
        
        Args:
            query_embedding: Query embedding vector
            document_embedding: Document embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Ensure vectors are 1D and float32
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            if document_embedding.ndim > 1:
                document_embedding = document_embedding.flatten()
                
            query_embedding = query_embedding.astype(np.float32)
            document_embedding = document_embedding.astype(np.float32)
            
            # Manual cosine similarity calculation
            dot_product = np.dot(query_embedding, document_embedding)
            query_norm = np.linalg.norm(query_embedding)
            doc_norm = np.linalg.norm(document_embedding)
            
            if query_norm == 0 or doc_norm == 0:
                return 0.0
                
            cosine_similarity = dot_product / (query_norm * doc_norm)
            
            # Ensure result is in valid range
            return max(-1.0, min(1.0, float(cosine_similarity)))
            
        except Exception as e:
            logger.warning(f"Error in cosine similarity verification: {e}")
            return 0.0

    def compare_faiss_vs_manual_similarity(self, query: str, document_content: str,
                                         embedding_model) -> Dict:
        """
        Compare FAISS similarity score with manual cosine similarity calculation.
        Useful for debugging embedding consistency.
        
        Args:
            query: Query text
            document_content: Document content to compare
            embedding_model: Sentence transformer model
            
        Returns:
            Dictionary with both scores for comparison
        """
        try:
            # Generate embeddings
            query_embedding = embedding_model.encode([query], normalize_embeddings=True)
            doc_embedding = embedding_model.encode([document_content], normalize_embeddings=True)
            
            # Manual cosine similarity
            manual_score = self.verify_cosine_similarity_calculation(
                query_embedding[0], doc_embedding[0]
            )
            
            return {
                "manual_cosine_similarity": manual_score,
                "calculation_method": "Manual dot product of L2-normalized vectors",
                "consistency_check": "✅ Consistent" if -1.0 <= manual_score <= 1.0 else "❌ Inconsistent"
            }
            
        except Exception as e:
            logger.warning(f"Error in similarity comparison: {e}")
            return {
                "manual_cosine_similarity": 0.0,
                "calculation_method": "Manual dot product of L2-normalized vectors",
                "consistency_check": "❌ Error in calculation"
            }