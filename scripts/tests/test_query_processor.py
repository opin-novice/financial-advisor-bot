import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.query_processor import QueryProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_query_categorization():
    """Test query categorization functionality"""
    query_processor = QueryProcessor()
    
    test_queries = [
        "What are the current interest rates for home loans?",
        "How do I open a savings account?",
        "What stocks should I invest in for long-term growth?",
        "How can I reduce my tax liability?",
        "What is the process for filing income tax returns?"
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        result = query_processor.process_query(query)
        logger.info(f"Detected category: {result['category']}")
        logger.info(f"Is followup: {result['is_followup']}")
        logger.info("-" * 50)

def test_followup_detection():
    """Test followup query detection"""
    query_processor = QueryProcessor()
    
    # Process an initial query
    initial_query = "What are the best investment options for retirement?"
    logger.info(f"Initial query: {initial_query}")
    result = query_processor.process_query(initial_query)
    logger.info(f"Detected category: {result['category']}")
    logger.info(f"Is followup: {result['is_followup']}")
    
    # Test followup queries
    followup_queries = [
        "What about for short-term goals?",
        "How much should I invest in these options?",
        "Are there tax benefits for these investments?"
    ]
    
    for query in followup_queries:
        logger.info(f"Testing followup query: {query}")
        result = query_processor.process_query(query)
        logger.info(f"Detected category: {result['category']}")
        logger.info(f"Is followup: {result['is_followup']}")
        logger.info("-" * 50)

if __name__ == "__main__":
    logger.info("Starting query processor tests")
    test_query_categorization()
    test_followup_detection()
    logger.info("Query processor tests completed")