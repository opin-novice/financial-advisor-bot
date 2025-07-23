import os
import logging
from typing import Dict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from src.utils.query_processor import QueryProcessor
from src.utils.pdf_processor import PDFProcessor
from src.utils.response_cache import ResponseCache

# Setup logging
logging.basicConfig(
    filename='logs/financial_advisor_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CACHE_TTL = 86400  # 24 hours in seconds

class FinancialAdvisorBot:
    LEGAL_DISCLAIMER = "This information is for educational purposes only and not financial advice. Consult a professional for personalized guidance."
    BLOCKED_PHRASES = ["illegal activity", "harmful content", "unethical advice"]

    def __init__(self):
        self.query_processor = QueryProcessor()
        self.pdf_processor = PDFProcessor()
        self.response_cache = ResponseCache(cache_dir="cache", ttl=CACHE_TTL)
        self.setup_qa_system()

    def setup_qa_system(self):
        logger.info("Initializing QA system...")
        try:
            # Load FAISS index
            # Change this line in setup_qa_system method
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}  # Changed from "cuda" to "cpu" for M1
            )
            self.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

            # Set up LLM and retriever
            self.retriever = self.vectorstore.as_retriever()
            self.llm = OllamaLLM(model="mistral:7b-instruct-v0.2-q4_K_M", temperature=0)  # Changed from "llama3" to "mistral"

            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            logger.info("QA system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QA system: {str(e)}")
            raise

    def content_filter(self, text: str) -> bool:
        """Checks if the given text contains any blocked phrases."""
        text_lower = text.lower()
        for phrase in self.BLOCKED_PHRASES:
            if phrase in text_lower:
                return True
        return False

    def format_response(self, response_text: str, source_documents: list, category: str, complexity: str) -> Dict:
        """Formats the final response with source attribution and legal disclaimer."""
        formatted_sources = []
        if source_documents:
            for doc in source_documents:
                source_name = doc['metadata'].get('source', 'Unknown Source')
                page_number = doc['metadata'].get('page', 'N/A')
                formatted_sources.append(f"Source: {source_name} (Page: {page_number})")

        sources_str = "\n\n" + "\n".join(formatted_sources) if formatted_sources else ""

        return {
            "response": f"{response_text}{sources_str}\n\n{self.LEGAL_DISCLAIMER}",
            "source_documents": source_documents,
            "category": category,
            "complexity": complexity
        }

    def process_query(self, query: str) -> Dict:
        try:
            # Process and enhance query
            query_info = self.query_processor.process_query(query)
            logger.info(f"Query category detected: {query_info['category']}")

            # Check cache for non-followup queries
            if not query_info['is_followup']:
                cached_response = self.response_cache.get(query)
                if cached_response:
                    logger.info(f"Returning cached response for query: {query[:50]}...")
                    return cached_response

            # Keyword-based pre-filtering: filter retriever by category metadata if not general
            if query_info['category'] != 'general':
                # Configure retriever with metadata filter
                filtered_retriever = self.vectorstore.as_retriever(search_kwargs={"filter": {"category": query_info['category']}})
                results_with_scores = filtered_retriever.invoke(query) # Changed from .get_relevant_documents(query) to .invoke(query)
                # LangChain's invoke returns Document objects directly, not with scores.
                # We need to simulate the score or adjust downstream logic if scores are critical.
                # For now, we'll assign a dummy score or assume relevance based on retrieval.
                # If scores are needed, we might need to explore custom retriever implementations or direct FAISS search with filters.
                # For simplicity, let's assume invoke is sufficient for now.
                # For now, we'll convert to the expected format, assuming a perfect score for retrieved documents.
                formatted_results = []
                for doc in results_with_scores:
                    formatted_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': 1.0 # Assign a dummy score as invoke doesn't return scores
                    })
                results_with_scores = formatted_results

            else:
                # Use full vectorstore for general queries
                results_with_scores = self.vectorstore.similarity_search_with_score(query, k=5)

            # Format search results
            search_results = []
            # Adjust this loop to handle both formats (with and without explicit scores from invoke)
            for i, item in enumerate(results_with_scores):
                if isinstance(item, tuple): # Original format (doc, score) from similarity_search_with_score
                    doc, score = item
                    search_results.append({
                        'rank': i + 1,
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': score
                    })
                else: # New format from invoke (dict with content, metadata, score)
                    search_results.append({
                        'rank': i + 1,
                        'content': item['content'],
                        'metadata': item['metadata'],
                        'score': item.get('score', 1.0) # Use provided score or default to 1.0
                    })

            # Summarize top results content (simple concatenation here; replace with actual summarization if available)
            summarized_content = "\n---\n".join([res['content'][:500] for res in search_results])

            # Generate LLM response
            if search_results:
                response_text = self.llm.invoke(summarized_content)
            else:
                response_text = "No relevant documents found."
                
            # Content filtering
            if self.content_filter(response_text):
                logger.warning(f"Blocked response due to sensitive content: {response_text[:50]}...")
                return {"response": "I cannot provide information on this topic due to content policy restrictions.", "source_documents": [], "category": query_info['category'], "complexity": "low"}

            # Prepare source documents for output
            source_documents = []
            for res in search_results:
                source_documents.append({
                    'content': res['content'],
                    'metadata': res['metadata']
                })

            # Determine response complexity (placeholder)
            complexity = "medium" # This would be determined by LLM or other logic

            # Format the final response
            final_response = self.format_response(
                response_text,
                source_documents,
                query_info['category'],
                complexity
            )
            
            # Add is_followup field to the response
            final_response['is_followup'] = query_info['is_followup']
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {"error": str(e), "category": "error", "source_documents": []}

def main():
    bot = FinancialAdvisorBot()
    print("Financial Advisor Bot initialized. Ask your questions!")

    while True:
        query = input("\nAsk me anything (type 'exit' to quit):\n> ")
        if query.lower() == "exit":
            break

        try:
            response = bot.process_query(query)

            if 'error' in response:
                print(f"\n❌ Error: {response['error']}")
                continue

            # Print the answer
            print("\n🔍 Answer:\n")
            print(response['response'])

            # Print category and followup status
            print(f"\n📊 Category: {response['category']}")
            if response.get('is_followup', False):
                print("ℹ️ This was detected as a follow-up question")

            # Print the source documents
            print("\n📚 Sources:")
            for i, doc in enumerate(response['source_documents'], 1):
                print(f"\n--- Source {i} ---")
                print(doc['content'])
                if 'page' in doc['metadata']:
                    print(f"[Page: {doc['metadata']['page']}]")
                print("-" * 40)

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
