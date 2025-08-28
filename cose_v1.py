#!/usr/bin/env python3
"""
Cosine Similarity Evaluation for Vanilla/Advanced RAG Systems
Measures similarity between LLM-generated answers and ground truth
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from typing import Dict, List
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import your RAG system here (Vanilla or Advanced)
from main import VanillaRAGSystem

# Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_GROUND_TRUTH_FILE = "dataqa/ENGqapair.json"
MAX_QUESTIONS = None  # Set to None for all questions, or a number to limit

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cosine_evaluation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Container for evaluation metrics"""
    def __init__(self, question_id: int, question: str, ground_truth: str):
        self.question_id = question_id
        self.question = question
        self.ground_truth = ground_truth
        self.generated_answer = ""
        self.similarity_scores = []
        self.max_similarity = 0.0
        self.avg_similarity = 0.0
        self.min_similarity = 0.0
        self.best_match_idx = -1
        self.metadata = {}
        self.processing_time = 0.0
        self.error = None


class CosineSimilarityEvaluator:
    """Evaluates cosine similarity between generated answers and ground truth answers"""

    def __init__(self, ground_truth_file: str = DEFAULT_GROUND_TRUTH_FILE):
        logger.info("🚀 Initializing Cosine Similarity Evaluator...")
        self.ground_truth_file = ground_truth_file
        self._init_embedding_model()
        self._init_rag_system()
        logger.info("✅ Cosine Similarity Evaluator initialized successfully")

    def _init_embedding_model(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _init_rag_system(self):
        """Initialize the RAG system"""
        try:
            logger.info("Initializing RAG system...")
            self.rag_system = VanillaRAGSystem()
            if not hasattr(self.rag_system, 'llm') or self.rag_system.llm is None:
                raise Exception("LLM not loaded in RAG system")
            logger.info("✅ RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise

    def load_ground_truth(self, file_path: str) -> List[Dict]:
        """Load ground truth data from JSON file"""
        try:
            logger.info(f"Loading ground truth data from: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Ground truth file not found: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded {len(data)} questions from ground truth data")
            return data
        except Exception as e:
            logger.error(f"Failed to load ground truth data: {e}")
            raise

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts"""
        valid_texts = [t.strip() for t in texts if t.strip()]
        if not valid_texts:
            return np.array([])
        try:
            embeddings = self.embedding_model.encode(valid_texts, normalize_embeddings=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return np.array([])

    def evaluate_single_question(self, question_data: Dict, question_id: int) -> EvaluationMetrics:
        """Evaluate a single question based on LLM-generated answer"""
        question = question_data['question']
        ground_truth_answer = question_data.get('answer', question_data.get('predicted_answer', ''))

        metrics = EvaluationMetrics(question_id, question, ground_truth_answer)

        try:
            start_time = datetime.now()
            # Process query with RAG system
            result = self.rag_system.process_query(question)
            generated_answer = result.get("response", "").strip()
            metrics.generated_answer = generated_answer
            metrics.metadata = result
            metrics.processing_time = (datetime.now() - start_time).total_seconds()

            if not generated_answer:
                metrics.error = "No generated answer"
                return metrics

            # Compute embeddings
            gen_emb = self.compute_embeddings([generated_answer])
            gt_emb = self.compute_embeddings([ground_truth_answer])

            if gen_emb.size == 0 or gt_emb.size == 0:
                metrics.error = "Invalid embeddings"
                return metrics

            # Cosine similarity
            similarity = float(cosine_similarity(gen_emb, gt_emb).flatten()[0])
            metrics.similarity_scores = [similarity]
            metrics.max_similarity = similarity
            metrics.avg_similarity = similarity
            metrics.min_similarity = similarity
            metrics.best_match_idx = 0

            return metrics

        except Exception as e:
            metrics.error = str(e)
            logger.error(f"Failed to evaluate question {question_id + 1}: {e}")
            return metrics

    def run_evaluation(self) -> str:
        """Run the full evaluation"""
        try:
            ground_truth_data = self.load_ground_truth(self.ground_truth_file)
            if MAX_QUESTIONS:
                ground_truth_data = ground_truth_data[:MAX_QUESTIONS]

            results = []
            successful = 0

            for i, q_data in enumerate(ground_truth_data):
                metrics = self.evaluate_single_question(q_data, i)
                results.append({
                    'question_id': metrics.question_id,
                    'question': metrics.question,
                    'ground_truth': metrics.ground_truth,
                    'generated_answer': metrics.generated_answer,
                    'similarity_scores': metrics.similarity_scores,
                    'max_similarity': metrics.max_similarity,
                    'avg_similarity': metrics.avg_similarity,
                    'min_similarity': metrics.min_similarity,
                    'metadata': metrics.metadata,
                    'processing_time': metrics.processing_time,
                    'error': metrics.error
                })
                if metrics.error is None:
                    successful += 1

            summary = self._generate_summary(results)
            output_file = self._save_results(results, summary)
            logger.info(f"✅ Evaluation completed: {successful}/{len(results)} successful")
            return output_file

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def _generate_summary(self, results: List[Dict]) -> Dict:
        valid = [r for r in results if r.get('error') is None]
        if not valid:
            return {'total_questions': len(results), 'successful': 0, 'failed': len(results)}

        max_sims = [r['max_similarity'] for r in valid]
        summary = {
            'total_questions': len(results),
            'successful': len(valid),
            'failed': len(results) - len(valid),
            'max_similarity_stats': {
                'mean': float(np.mean(max_sims)),
                'median': float(np.median(max_sims)),
                'min': float(np.min(max_sims)),
                'max': float(np.max(max_sims)),
            },
            'avg_retrieval_time': float(np.mean([r['processing_time'] for r in valid]))
        }
        return summary

    def _save_results(self, results: List[Dict], summary: Dict) -> str:
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/cosine_similarity_evaluation_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'detailed_results': results}, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to: {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system by cosine similarity")
    parser.add_argument('file', nargs='?', help='Ground truth JSON file')
    parser.add_argument('--max-questions', '-m', type=int, help='Maximum number of questions to evaluate')
    args = parser.parse_args()

    global MAX_QUESTIONS
    if args.max_questions:
        MAX_QUESTIONS = args.max_questions

    ground_truth_file = args.file if args.file else DEFAULT_GROUND_TRUTH_FILE
    evaluator = CosineSimilarityEvaluator(ground_truth_file)
    results_file = evaluator.run_evaluation()
    print(f"🎉 Evaluation completed! Results saved to: {results_file}")


if __name__ == "__main__":
    sys.exit(main())
