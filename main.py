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
            self.llm = OllamaLLM(model="llama3")  # or any other model available

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
                # Create a filtered retriever for the category
                filtered_docs = [doc for doc in self.vectorstore.docstore._dict.values() 
                                 if doc.metadata.get('category') == query_info['category']]
                if filtered_docs:
                    # Create a temporary FAISS index with filtered docs
                    # Note: This is a simplified approach; in production, maintain separate indices or use metadata filtering
                    temp_vectorstore = FAISS.from_documents(filtered_docs, self.vectorstore.embedding_function)
                    results_with_scores = temp_vectorstore.similarity_search_with_score(query, k=5)
                else:
                    results_with_scores = []
            else:
                # Use full vectorstore for general queries
                results_with_scores = self.vectorstore.similarity_search_with_score(query, k=5)

            # Format search results
            search_results = []
            for i, (doc, score) in enumerate(results_with_scores):
                search_results.append({
                    'rank': i + 1,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                })

            # Summarize top results content (simple concatenation here; replace with actual summarization if available)
            summarized_content = "\n---\n".join([res['content'][:500] for res in search_results])

            # Add at the top of the file
            LEGAL_DISCLAIMER = "\n\n---\n*Disclaimer: This is not personalized financial advice; please consult a professional.*"
            
            BLOCKED_PHRASES = ["invest all money", "tax evasion", "guaranteed returns"]
            
            def content_filter(text):
                for phrase in BLOCKED_PHRASES:
                    if phrase in text.lower():
                        return False
                return True
            
            def format_response(text, category, complexity_level, sources):
                if complexity_level == 'simple':
                    formatted = f"{text}\n\n(Source: {', '.join(sources)})"
                else:
                    formatted = f"Detailed Response:\n{text}\n\nSources:\n" + '\n'.join([f"- {src}" for src in sources])
                return formatted + LEGAL_DISCLAIMER
            
            # In the process_query method, replace the response generation part with:
            
            # Generate response using LLM on summarized content
            # Generate LLM response
            if search_results:
                response_text = self.llm.invoke(summarized_content)
            else:
                response_text = "No relevant documents found."
            
            # Content filtering
            if not content_filter(response_text):
                response_text = "Sorry, the response contains content that is not allowed."
            
            # Prepare sources list
            sources = [res['metadata'].get('file_name', 'unknown source') for res in search_results]
            
            # Determine complexity
            complexity_level = 'simple' if len(query.split()) < 10 else 'detailed'
            
            # Format response
            response_text = format_response(response_text, query_info['category'], complexity_level, sources)
            
            # Log response quality metrics
            logger.info(f"Response generated for query: {query[:50]}... Category: {query_info['category']} Complexity: {complexity_level}")
            
            # Continue with caching and returning response
            
            enhanced_response = {
                'result': response_text,
                'category': query_info['category'],
                'is_followup': query_info['is_followup'],
                'source_documents': [
                    {
                        'content': res['content'][:500],
                        'metadata': res['metadata'],
                        'score': res['score']
                    } for res in search_results
                ]
            }

            # Cache the response for non-followup queries
            if not query_info['is_followup']:
                self.response_cache.set(query, enhanced_response)

            logger.info(f"Successfully processed query in category: {query_info['category']}")
            return enhanced_response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {'error': str(e)}

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
            print(response['result'])

            # Print category and followup status
            print(f"\n📊 Category: {response['category']}")
            if response['is_followup']:
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
