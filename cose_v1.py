import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports (adjust as needed)
from main2 import RAGSystem  # <-- your Vanilla/Advanced RAG system

# -------------------------------------------------
# Logging Setup
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------------------------------
# Data Classes
# -------------------------------------------------
class EvaluationMetrics:
    def __init__(self, question_id: int, question: str, ground_truth: str):
        self.question_id = question_id
        self.question = question
        self.ground_truth = ground_truth
        self.retrieved_passages: List[str] = []
        self.llm_response: str = ""
        self.similarity_passages: Dict = {}
        self.similarity_response: Dict = {}
        self.processing_time: float = 0.0
        self.metadata: Dict = {}
        self.error: str = ""

# -------------------------------------------------
# Cosine Similarity Evaluator
# -------------------------------------------------
class CosineSimilarityEvaluator:
    def __init__(self, ground_truth_file: str):
        logging.info("🚀 Initializing Cosine Similarity Evaluator...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        # Load embedding model
        model_name = "BAAI/bge-m3"
        logging.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        logging.info("✅ Embedding model loaded successfully")

        # Load QA dataset (ground truth)
        self.ground_truth_df = pd.read_csv(ground_truth_file)
        logging.info(f"✅ Ground truth file loaded with {len(self.ground_truth_df)} questions")

        # Initialize RAG system
        self._init_rag_system()

    def _init_rag_system(self):
        logging.info("Initializing RAG system...")
        self.rag_system = RAGSystem()
        if not self.rag_system.llm:
            raise Exception("LLM not loaded in RAG system")
        logging.info("✅ RAG system initialized successfully")

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    def calculate_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0])

    def evaluate_single_question(self, question_data: Dict, question_id: int) -> EvaluationMetrics:
        question = question_data["question"]
        ground_truth_answer = question_data.get("answer", "")
        metrics = EvaluationMetrics(question_id, question, ground_truth_answer)

        try:
            start_time = datetime.now()
            result = self.rag_system.process_query(question)

            # Save metadata
            metrics.metadata = {
                "sources": result.get("sources", []),
                "num_docs": result.get("num_docs", 0),
                "processing_time": result.get("processing_time", 0.0),
            }
            metrics.processing_time = (datetime.now() - start_time).total_seconds()

            # Retrieved passages
            passages = result.get("passages", [])
            metrics.retrieved_passages = passages

            # Final LLM response
            llm_response = result.get("response", "")
            metrics.llm_response = llm_response

            # ---- Passage-level similarity ----
            if passages:
                passage_embs = self.compute_embeddings(passages)
                gt_emb = self.compute_embeddings([ground_truth_answer])
                sims = [self.calculate_cosine_similarity(p_emb, gt_emb[0]) for p_emb in passage_embs]
                metrics.similarity_passages = {
                    "max": float(np.max(sims)),
                    "min": float(np.min(sims)),
                    "avg": float(np.mean(sims)),
                }

            # ---- Response-level similarity ----
            if llm_response.strip():
                resp_emb = self.compute_embeddings([llm_response])
                gt_emb = self.compute_embeddings([ground_truth_answer])
                sim = self.calculate_cosine_similarity(resp_emb[0], gt_emb[0])
                metrics.similarity_response = {"cosine": sim}

            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics

    def evaluate(self, output_file: str = "cosine_results.csv") -> pd.DataFrame:
        results = []

        for idx, row in self.ground_truth_df.iterrows():
            metrics = self.evaluate_single_question(row, idx)
            results.append({
                "question_id": metrics.question_id,
                "question": metrics.question,
                "ground_truth": metrics.ground_truth,
                "llm_response": metrics.llm_response,
                "retrieved_passages": metrics.retrieved_passages,
                "similarity_passages": metrics.similarity_passages,
                "similarity_response": metrics.similarity_response,
                "processing_time": metrics.processing_time,
                "error": metrics.error,
            })

        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logging.info(f"✅ Evaluation complete. Results saved to {output_file}")
        return df

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python cosev2.py <ground_truth_file.csv>")
        sys.exit(1)

    ground_truth_file = sys.argv[1]
    evaluator = CosineSimilarityEvaluator(ground_truth_file)
    evaluator.evaluate()

if __name__ == "__main__":
    sys.exit(main())

