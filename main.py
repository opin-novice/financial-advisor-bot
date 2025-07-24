import os
import logging
import time
from typing import Dict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from src.utils.query_processor import QueryProcessor
from src.utils.response_cache import ResponseCache
from src.utils.performance_monitor import PerformanceMonitor, ResponseTimer

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
LLM_MODEL = "mistral:7b-instruct-v0.2-q4_K_M"

# Speed optimization settings
MAX_DOCS_FOR_RETRIEVAL = 8  # Reduced from 10 for faster processing
MAX_DOCS_FOR_CONTEXT = 3   # Reduced for faster LLM processing
CONTEXT_CHUNK_SIZE = 1500  # Limit context size per document

# Enhanced prompt template for better RAG responses
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
Be concise and helpful.

Context: {context}

Question: {question}
Helpful Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)

class FinancialAdvisorBot:
    LEGAL_DISCLAIMER = "This information is for educational purposes only and not financial advice. Consult a professional for personalized guidance."

    def __init__(self):
        self.query_processor = QueryProcessor()
        self.response_cache = ResponseCache(cache_dir="cache", ttl=CACHE_TTL)
        self.performance_monitor = PerformanceMonitor()
        self.setup_qa_system()

    def setup_qa_system(self):
        logger.info("Initializing QA system...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )
            self.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            self.llm = OllamaLLM(model=LLM_MODEL, temperature=0)
            self.qa_chain = load_qa_chain(self.llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
            logger.info("QA system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QA system: {str(e)}")
            raise

    def process_query(self, query: str, chat_history: list = None) -> Dict:
        # Detect query category first for performance tracking
        query_info = self.query_processor.process_query(query)
        category = query_info['category']
        
        with ResponseTimer(self.performance_monitor, category):
            try:
                # Check cache first for speed
                cached_response = self.response_cache.get(query)
                if cached_response:
                    self.performance_monitor.track_cache_hit()
                    logger.info(f"Returning cached response for query: {query[:50]}...")
                    return cached_response
                
                self.performance_monitor.track_cache_miss()
                logger.info(f"Query category detected: {category}")

                # Optimized document retrieval with reduced count
                retrieved_docs = self.vectorstore.similarity_search(query, k=MAX_DOCS_FOR_RETRIEVAL)

                # Enhanced document filtering and ranking for better relevance
                filtered_docs = self._rank_and_filter_documents(retrieved_docs, query, category)

                if not filtered_docs:
                    return {
                        "response": "I apologize, but I couldn't find relevant information in my knowledge base for your specific question. Please consider consulting with a financial professional for personalized guidance.", 
                        "sources": [],
                        "category": category,
                        "source_documents": [],
                        "disclaimer": self.LEGAL_DISCLAIMER
                    }

                # Use only top documents for faster processing
                top_docs = filtered_docs[:MAX_DOCS_FOR_CONTEXT]
                
                # Truncate document content for faster processing
                processed_docs = self._prepare_documents_for_context(top_docs)

                result = self.qa_chain.invoke({"input_documents": processed_docs, "question": query})
                response_text = result.get('output_text', "I apologize, but I couldn't generate a complete answer from the available information.")

                final_response = self.format_enhanced_response(response_text, top_docs, category)
                
                # Cache the response for future use
                self.response_cache.set(query, final_response)

                return final_response

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                return {
                    "error": f"I encountered an error while processing your question: {str(e)}", 
                    "sources": [],
                    "category": category,
                    "source_documents": [],
                    "disclaimer": self.LEGAL_DISCLAIMER
                }

    

    def _rank_and_filter_documents(self, docs, query, category):
        """Enhanced document ranking with multiple factors"""
        query_terms = set(query.lower().split())
        
        def calculate_relevance_score(doc):
            content = doc.page_content.lower()
            metadata = doc.metadata
            
            # Base score from keyword matching
            keyword_score = sum(1 for term in query_terms if term in content)
            
            # Category bonus
            category_bonus = 2 if metadata.get('category') == category else 0
            
            # Content length penalty (prefer more focused content)
            length_penalty = len(content) / 5000  # Normalize by typical doc length
            
            return keyword_score + category_bonus - length_penalty
        
        # Sort by relevance score
        scored_docs = [(doc, calculate_relevance_score(doc)) for doc in docs]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs if score > 0]
    
    def _prepare_documents_for_context(self, docs):
        """Prepare documents for LLM context with size optimization"""
        processed_docs = []
        for doc in docs:
            # Truncate very long documents to speed up processing
            content = doc.page_content
            if len(content) > CONTEXT_CHUNK_SIZE:
                content = content[:CONTEXT_CHUNK_SIZE] + "...[content truncated]"
            
            # Create new document with truncated content
            from langchain.schema import Document
            processed_doc = Document(
                page_content=content,
                metadata=doc.metadata
            )
            processed_docs.append(processed_doc)
        
        return processed_docs

    def format_enhanced_response(self, response_text: str, source_documents: list, category: str) -> Dict:
        """Enhanced response formatting for better readability"""
        
        # Clean up the response text
        cleaned_response = self._clean_and_enhance_response(response_text)
        
        # Prepare enhanced sources information
        sources_info = self._format_sources(source_documents)
        
        # Create category-specific response structure
        formatted_response = self._apply_category_formatting(cleaned_response, category)
        
        return {
            "response": formatted_response,
            "sources": sources_info,
            "category": category,
            "source_documents": source_documents,
            "disclaimer": self.LEGAL_DISCLAIMER
        }
    
    def _clean_and_enhance_response(self, response_text: str) -> str:
        """Clean and enhance the raw response text"""
        # Remove any unwanted patterns or clean up formatting
        cleaned = response_text.strip()
        
        # Ensure proper paragraph spacing
        lines = cleaned.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Add proper formatting for numbered lists
                if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                    formatted_lines.append(f"\n{line}")
                # Add proper formatting for bullet points
                elif line.startswith(('•', '-', '*')):
                    formatted_lines.append(f"\n{line}")
                else:
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines).strip()
    
    def _format_sources(self, source_documents: list) -> list:
        """Format source information in a structured way"""
        sources = []
        seen_sources = set()
        
        for doc in source_documents:
            source_name = doc.metadata.get('source', 'Unknown Source')
            page_number = doc.metadata.get('page', 'N/A')
            category = doc.metadata.get('category', 'General')
            
            source_key = f"{source_name}_{page_number}"
            if source_key not in seen_sources:
                sources.append({
                    "name": source_name,
                    "page": page_number,
                    "category": category.title()
                })
                seen_sources.add(source_key)
        
        return sources
    
    def _apply_category_formatting(self, response: str, category: str) -> str:
        """Apply category-specific formatting to responses"""
        category_headers = {
            'taxation': '📊 Tax Information',
            'banking': '🏦 Banking Guidance', 
            'loans': '💰 Loan Information',
            'investment': '📈 Investment Guidance',
            'sme': '🏢 SME & Business Information',
            'regulations': '📜 Regulatory Information',
            'general': '💼 Financial Guidance'
        }
        
        header = category_headers.get(category, '💼 Financial Information')
        
        # Add header and ensure proper structure
        formatted = f"{header}\n\n{response}"
        
        return formatted

def main():
    bot = FinancialAdvisorBot()
    print("\n" + "="*60)
    print("🤖 Financial Advisor Bot - Enhanced RAG System")
    print("="*60)
    print("Ready to help with your financial questions!")
    print("Type 'exit' to quit or 'stats' to see performance statistics.\n")

    query_count = 0
    
    while True:
        query = input("Ask me anything (type 'exit' to quit):\n> ")
        
        if query.lower() == "exit":
            print("\n👋 Thank you for using Financial Advisor Bot!")
            if query_count > 0:
                bot.performance_monitor.log_performance_summary()
            break
            
        if query.lower() == "stats":
            stats = bot.performance_monitor.get_performance_stats()
            if stats['overall'].get('total_queries', 0) > 0:
                print("\n📈 Performance Statistics:")
                print(f"Total queries: {stats['overall']['total_queries']}")
                print(f"Average response time: {stats['overall']['avg_response_time']:.2f}s")
                print(f"Cache hit rate: {stats['cache_performance']['hit_rate']}%")
                print("Category breakdown:")
                for cat, cat_stats in stats['by_category'].items():
                    print(f"  {cat}: {cat_stats['count']} queries, avg {cat_stats['avg_response_time']:.2f}s")
            else:
                print("\n📊 No queries processed yet.")
            continue

        if not query.strip():
            print("\n⚠️  Please enter a valid question.")
            continue

        try:
            start_time = time.time()
            print("\n🔍 Processing your question...")
            response = bot.process_query(query)
            processing_time = time.time() - start_time
            query_count += 1

            if 'error' in response:
                print(f"\n❌ Error: {response['error']}")
                continue

            # Display the enhanced response
            print("\n" + "="*50)
            print(response['response'])
            
            # Display sources in a formatted way
            if response.get('sources'):
                print("\n" + "-"*40)
                print("📚 Sources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['name']} (Page: {source['page']}) - {source['category']}")
            
            # Display disclaimer and processing time
            print("\n" + "-"*40)
            print(f"⚠️  {response['disclaimer']}")
            print(f"⏱️  Response generated in {processing_time:.2f}s")
            print("="*50)

        except Exception as e:
            print(f"\n❌ Unexpected Error: {str(e)}")
            print("Please try again or contact support if the issue persists.")

if __name__ == "__main__":
    main()