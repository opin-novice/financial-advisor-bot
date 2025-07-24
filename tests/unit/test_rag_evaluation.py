import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from main import FinancialAdvisorBot

class TestRAGEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class with a FinancialAdvisorBot instance."""
        try:
            cls.bot = FinancialAdvisorBot()
        except Exception as e:
            raise unittest.SkipTest(f"Skipping RAG evaluation tests: {e}")

    def evaluate_response(self, query, bot_response, expected_keywords, expected_sources):
        """
        Evaluates a single bot response against expected outcomes.
        """
        response_text = bot_response.get('response', '').lower()
        
        # Keyword Match Score
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in response_text]
        keyword_score = len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0

        # Source Match Score
        retrieved_sources = [os.path.basename(doc.metadata.get('source', '')) for doc in bot_response.get('source_documents', [])]
        matched_sources = [src for src in expected_sources if src in retrieved_sources]
        source_score = len(matched_sources) / len(expected_sources) if expected_sources else 1.0

        # Coherence Score (simple heuristic)
        is_coherent = all(msg not in response_text for msg in ["error", "cannot", "unable"]) and len(response_text) > 20
        coherence_score = 1.0 if is_coherent else 0.0

        return {
            "query": query,
            "keyword_score": keyword_score,
            "source_score": source_score,
            "coherence_score": coherence_score,
            "retrieved_sources": retrieved_sources
        }

    def test_rag_quality(self):
        """
        Tests the overall quality of the RAG system using a predefined set of queries and expected outcomes.
        """
        test_data = [
            {
                "query": "How do I open a bank account?",
                "expected_keywords": ["application form", "identification", "proof of address"],
                "expected_sources": ["bd-personal-account-opening-form.pdf", "personal_ac_opening_form.pdf"]
            },
            {
                "query": "What are the benefits of a 5-Year Bangladesh Sanchayapatra?",
                "expected_keywords": ["investment", "interest rate", "government security"],
                "expected_sources": ["5-Year Bangladesh Savings Certificate.pdf"]
            },
            {
                "query": "What documents are required for a home loan?",
                "expected_keywords": ["application", "income", "property"],
                "expected_sources": ["home_loan_application.pdf", "Comprehensive Loan Documentation Guide for Bangladesh.pdf"]
            }
        ]

        results = []
        for item in test_data:
            query = item["query"]
            bot_response = self.bot.process_query(query)
            evaluation = self.evaluate_response(query, bot_response, item["expected_keywords"], item["expected_sources"])
            results.append(evaluation)

        # Print the evaluation report
        print("\n--- RAG Evaluation Report ---")
        total_keyword_score = 0
        total_source_score = 0
        total_coherence_score = 0

        for res in results:
            print(f"\nQuery: {res['query']}")
            print(f"  - Keyword Match Score: {res['keyword_score']:.2f}")
            print(f"  - Source Match Score: {res['source_score']:.2f}")
            print(f"  - Coherence Score: {res['coherence_score']:.2f}")
            print(f"  - Retrieved Sources: {res['retrieved_sources']}")
            total_keyword_score += res['keyword_score']
            total_source_score += res['source_score']
            total_coherence_score += res['coherence_score']

        avg_keyword_score = total_keyword_score / len(results)
        avg_source_score = total_source_score / len(results)
        avg_coherence_score = total_coherence_score / len(results)

        print("\n--- Average Scores ---")
        print(f"Average Keyword Match Score: {avg_keyword_score:.2f}")
        print(f"Average Source Match Score: {avg_source_score:.2f}")
        print(f"Average Coherence Score: {avg_coherence_score:.2f}")

        # Assert that the average scores are above a certain threshold
        self.assertGreater(avg_keyword_score, 0.6)
        self.assertGreater(avg_source_score, 0.6)
        self.assertGreater(avg_coherence_score, 0.6)

if __name__ == '__main__':
    unittest.main()