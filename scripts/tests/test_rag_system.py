import sys
import os
import logging
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import FinancialAdvisorBot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_end_to_end_queries():
    """Test end-to-end query processing with the RAG system"""
    bot = FinancialAdvisorBot()
    
    test_queries = [
        "What are the best investment strategies for beginners?",
        "How do I calculate my tax liability?",
        "What are the current regulations for small business loans?",
        "How can I improve my credit score?"
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        start_time = time.time()
        response = bot.process_query(query)
        processing_time = time.time() - start_time
        
        logger.info(f"Processing time: {processing_time:.2f} seconds")

        if 'error' in response:
            logger.error(f"Error processing query '{query}': {response['error']}")
            continue

        logger.info(f"Response category: {response['category']}")
        logger.info(f"Number of source documents: {len(response['source_documents'])}")
        assert 'category' in response
        assert 'source_documents' in response
        assert len(response['source_documents']) > 0
        logger.info("-" * 50)

if __name__ == "__main__":
    logger.info("Starting RAG system tests")
    test_end_to_end_queries()
    logger.info("RAG system tests completed")