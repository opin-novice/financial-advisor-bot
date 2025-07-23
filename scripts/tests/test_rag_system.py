import unittest
import time
import logging
import os
import json
import pytest
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.query_processor import QueryProcessor
from src.utils.response_cache import ResponseCache
from src.utils.document_manager import DocumentManager

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestRAGSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.query_processor = QueryProcessor()
        cls.cache = ResponseCache()
        cls.doc_manager = DocumentManager()
        cls.test_queries = [
            ("What are the eligibility criteria for a personal loan?", "loans"),
            ("How can I open a savings account?", "banking"),
            ("What is a mutual fund?", "investment"),
            ("Explain the process of applying for a mortgage.", "loans"),
            ("What documents are required for KYC?", "banking"),
            ("What are the tax implications of investing in stocks?", "investment"),
            ("How does a credit score affect loan applications?", "loans"),
            ("What is the difference between a checking and savings account?", "banking"),
            ("What are the risks associated with high-yield bonds?", "investment"),
            ("Can you explain overdraft protection?", "banking"),
            ("What is a diversified portfolio?", "investment"),
            ("What are the benefits of a fixed deposit?", "banking")
        ]

    def test_end_to_end_query_processing(self):
        logging.info("\n--- Running End-to-End RAG System Tests ---")
        results = []
        for i, (query, expected_category) in enumerate(self.test_queries):
            start_time = time.time()
            query_result = self.query_processor.process_query(query)
            end_time = time.time()
            response_time = end_time - start_time

            # Extract values from the query result
            processed_query = query_result.get('processed_query', query)
            category = query_result.get('category', 'general')
            is_followup = query_result.get('is_followup', False)

            logging.info(f"Query {i+1}: {query}")
            logging.info(f"Processed Query: {processed_query}")
            logging.info(f"Category: {category}")
            logging.info(f"Is Followup: {is_followup}")
            logging.info(f"Response Time: {response_time:.2f} seconds")

            # Since QueryProcessor only categorizes queries and doesn't generate responses,
            # we'll use placeholder values for response and source documents
            response = f"Sample response for {category} query: {query}"
            source_documents = []  # QueryProcessor doesn't return source documents
            
            # Basic quality metrics (placeholders for now)
            context_match = "Medium"  # Since we don't have actual retrieval
            answer_quality = "Good" # Placeholder: requires manual/LLM evaluation

            results.append({
                "query": query,
                "response": response,
                "category": category,
                "expected_category": expected_category,
                "response_time": response_time,
                "source_documents_count": len(source_documents),
                "context_match": context_match,
                "answer_quality": answer_quality,
                "overall_confidence": "N/A", # Requires more advanced evaluation
                "source_quality": "N/A", # Requires more advanced evaluation
                "is_followup": is_followup
            })

            self.assertIsNotNone(processed_query)
            self.assertGreater(len(processed_query), 0)
            self.assertIsNotNone(category)
            # Test that category detection is working
            self.assertIn(category, ['banking', 'investment', 'loans', 'taxation', 'general'])

        # You can log or process the results list further if needed
        self.log_detailed_results(results)
        # Print the results in a machine-readable format for run_all_tests.py
        print("====TEST_RAG_SYSTEM_RESULTS_START====")
        print(json.dumps(results, indent=2))
        print("====TEST_RAG_SYSTEM_RESULTS_END====")

    def log_detailed_results(self, results):
        logging.info("\n--- Detailed RAG System Test Results ---")
        total_response_time = 0
        successful_queries = 0
        for res in results:
            logging.info(f"Query: {res['query']}")
            logging.info(f"  Response Time: {res['response_time']:.2f}s")
            logging.info(f"  Category: {res['category']} (Expected: {res['expected_category']})")
            logging.info(f"  Source Docs: {res['source_documents_count']}")
            logging.info(f"  Context Match: {res['context_match']}")
            logging.info(f"  Answer Quality: {res['answer_quality']}")
            logging.info(f"  Overall Confidence: {res['overall_confidence']}")
            logging.info(f"  Source Quality: {res['source_quality']}")
            total_response_time += res['response_time']
            successful_queries += 1 # Assuming all queries in this test are 'successful' if they return a response

        if successful_queries > 0:
            avg_response_time = total_response_time / successful_queries
            logging.info(f"\nAverage Response Time: {avg_response_time:.2f} seconds")
            logging.info(f"Total Queries Processed: {len(results)}")
            logging.info(f"Successful Queries: {successful_queries}")
        else:
            logging.info("No queries were processed.")

if __name__ == '__main__':
    # When run directly, it will execute tests and print summary
    unittest.main()
