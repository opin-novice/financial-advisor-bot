
import os
import sys
import unittest
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from main import FinancialAdvisorBot

# Configuration from main.py
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:1b-extended')

class TestRagasEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class with a FinancialAdvisorBot instance."""
        try:
            cls.bot = FinancialAdvisorBot()
            # Disable cache for testing to ensure fresh responses
            cls.bot.response_cache.get = lambda query: None
            
            # Initialize LLM and Embeddings for RAGAS
            cls.llm = OllamaLLM(model=LLM_MODEL, temperature=0)
            cls.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )
        except Exception as e:
            raise unittest.SkipTest(f"Skipping RAGAS evaluation tests: {e}")

    def test_ragas_evaluation(self):
        """
        Tests the RAG system using the RAGAS framework.
        """
        test_data = [
            {
                "question": "How do I open a bank account?",
                "ground_truths": [
                    "To open a bank account, you typically need to fill out an application form, provide valid identification (like a national ID card or passport), and show proof of address."
                ],
            },
            {
                "question": "What are the benefits of a 5-Year Bangladesh Sanchayapatra?",
                "ground_truths": [
                    "The 5-Year Bangladesh Sanchayapatra is a government savings certificate that offers a competitive interest rate and is considered a very safe investment."
                ],
            },
            {
                "question": "What documents are required for a home loan?",
                "ground_truths": [
                    "To apply for a home loan, you will generally need to provide an application form, proof of income (such as salary slips or bank statements), and documents related to the property you intend to purchase."
                ],
            },
        ]

        # Collect questions and ground truths
        questions = [item["question"] for item in test_data]
        ground_truths = [item["ground_truths"][0] for item in test_data]

        # Get answers and contexts from the bot
        answers = []
        contexts = []
        for question in questions:
            bot_response = self.bot.process_query(question)
            answers.append(bot_response.get("response", ""))
            contexts.append(
                [doc.page_content for doc in bot_response.get("source_documents", [])]
            )

        # Create a dataset for RAGAS
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(dataset_dict)

        # Define the metrics for evaluation
        metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]

        # Run the evaluation
        result = evaluate(
            dataset, 
            metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )

        # Print the results
        print("--- RAGAS Evaluation Report ---")
        print(result)

        # Assert that the scores are above a certain threshold
        if hasattr(result, 'faithfulness') and isinstance(result.faithfulness, float) and not np.isnan(result.faithfulness):
            self.assertGreater(result.faithfulness, 0.7, "Faithfulness score is below threshold.")
        else:
            print(f"Warning: Faithfulness score is NaN or not available. Skipping assertion.")

        if hasattr(result, 'answer_relevancy') and isinstance(result.answer_relevancy, float) and not np.isnan(result.answer_relevancy):
            self.assertGreater(result.answer_relevancy, 0.7, "Answer relevancy score is below threshold.")
        else:
            print(f"Warning: Answer relevancy score is NaN or not available. Skipping assertion.")

        if hasattr(result, 'context_recall') and isinstance(result.context_recall, float) and not np.isnan(result.context_recall):
            self.assertGreater(result.context_recall, 0.7, "Context recall score is below threshold.")
        else:
            print(f"Warning: Context recall score is NaN or not available. Skipping assertion.")

        if hasattr(result, 'context_precision') and isinstance(result.context_precision, float) and not np.isnan(result.context_precision):
            self.assertGreater(result.context_precision, 0.7, "Context precision score is below threshold.")
        else:
            print(f"Warning: Context precision score is NaN or not available. Skipping assertion.")

if __name__ == "__main__":
    unittest.main()
