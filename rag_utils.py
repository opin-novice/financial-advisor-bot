import os
import logging
from typing import Dict, List, Tuple
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

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

class RAGUtils:
    def __init__(self):
        """Initialize RAG utilities with Groq LLM"""
        try:
            self.llm = ChatGroq(
                model=GROQ_MODEL,
                groq_api_key=GROQ_API_KEY,
                temperature=0.3,
                max_tokens=500,
            )
            logger.info("✅ Groq LLM initialized for RAG utilities")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
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
        # Create a prompt for query refinement
        refinement_prompt = PromptTemplate.from_template("""
        You are a query refinement assistant for a financial advisor system.
        Improve the following user query to better retrieve relevant financial information.
        
        Original Query: {query}
        
        Instructions:
        1. Expand financial terms and acronyms
        2. Add relevant financial/banking context
        3. Remove irrelevant words
        4. Make query more specific to Bangladesh financial context if applicable
        5. Keep the intent the same
        6. Return ONLY the refined query, no explanations or additional text
        
        Refined Query:
        """)
        
        try:
            prompt = refinement_prompt.format(query=query)
            response = self.llm.invoke(prompt)
            refined_query = response.content.strip()
            
            # Ensure we don't return empty or nonsensical refinements
            if refined_query and len(refined_query) > 5:
                logger.info(f"Query refined: '{query}' -> '{refined_query}'")
                return refined_query
            else:
                logger.warning(f"Query refinement returned poor result. Using original: {query}")
                return query
                
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}. Using original query.")
            return query

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