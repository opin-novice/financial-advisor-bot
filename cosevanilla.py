#!/usr/bin/env python3
"""
Cosine Similarity Evaluation for Vanilla RAG System (main.py)
Evaluates the quality of retrieved passages against ground truth answers
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import the vanilla RAG system from main.py
from main import VanillaRAGSystem

# Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_GROUND_TRUTH_FILE = "dataqa/ENGqapair.json"
MAX_QUESTIONS = None  # Set to None for all questions, or a number to limit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cosevanilla_evaluation.log', encoding='utf-8'),
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
        self.retrieved_passages = []
        self.max_similarity = 0.0
        self.avg_similarity = 0.0
        self.min_similarity = 0.0
        self.best_match_idx = -1
        self.similarity_scores = []
        self.metadata = {}
        self.processing_time = 0.0
        self.error = None

class CosineSimilarityEvaluator:
    """Evaluates cosine similarity between retrieved passages and ground truth answers"""
    
    def __init__(self, ground_truth_file: str = DEFAULT_GROUND_TRUTH_FILE):
        """Initialize the evaluator"""
        logger.info("🚀 Initializing Vanilla Cosine Similarity Evaluator...")
        logger.info(f"📄 Ground truth file: {ground_truth_file}")
        
        self.ground_truth_file = ground_truth_file
        self._init_embedding_model()
        self._init_rag_system()
        
        logger.info("✅ Vanilla Cosine Similarity Evaluator initialized successfully")
    
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
        """Initialize the vanilla RAG system for document retrieval"""
        try:
            logger.info("Initializing vanilla RAG system...")
            self.rag_system = VanillaRAGSystem()
            
            # Verify system is ready
            if not hasattr(self.rag_system, 'vectorstore') or self.rag_system.vectorstore is None:
                raise Exception("FAISS index not loaded")
            if not hasattr(self.rag_system, 'llm') or self.rag_system.llm is None:
                raise Exception("LLM not loaded in RAG system")
            
            logger.info("✅ Vanilla RAG system initialized successfully")
            logger.info(f"   - Documents: {self.rag_system.vectorstore.index.ntotal:,}")
            logger.info(f"   - FAISS Index: ✅")
            logger.info(f"   - LLM: ✅")
            
        except Exception as e:
            logger.error(f"Failed to initialize vanilla RAG system: {e}")
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
    
    def retrieve_passages(self, question: str) -> Tuple[List[str], Dict]:
        """Retrieve passages for a question using the vanilla RAG system"""
        try:
            # Process query using vanilla RAG system
            result = self.rag_system.process_query(question)
            
            # Extract retrieved passages from contexts
            retrieved_passages = result.get('contexts', [])
            
            # If no contexts found, try to get from source documents directly
            if not retrieved_passages:
                # Try to access the retriever directly to get documents
                try:
                    # Get documents directly from the retriever
                    docs = self.rag_system.retriever.get_relevant_documents(question)
                    retrieved_passages = [doc.page_content for doc in docs[:6]]  # Limit to 6 docs
                    logger.info(f"Retrieved {len(retrieved_passages)} passages directly from retriever")
                except Exception as retriever_error:
                    logger.warning(f"Could not get documents directly: {retriever_error}")
            
            # Get metadata
            metadata = {
                'detected_language': result.get('detected_language', 'unknown'),
                'language_confidence': result.get('language_confidence', 0.0),
                'num_docs': result.get('num_docs', 0),
                'processing_time': result.get('processing_time', 0.0),
                'sources': result.get('sources', []),
                'response': result.get('response', '')[:200]  # Include first 200 chars of response
            }
            
            logger.info(f"Retrieved {len(retrieved_passages)} passages for question")
            return retrieved_passages, metadata
            
        except Exception as e:
            logger.error(f"Failed to retrieve passages for question: {question[:50]}... Error: {e}")
            return [], {}
    
    def compute_embeddings(self, passages: List[str]) -> np.ndarray:
        """Compute embeddings for a list of passages"""
        if not passages:
            return np.array([])
        
        try:
            # Filter out empty or very short passages
            valid_passages = [p.strip() for p in passages if p.strip() and len(p.strip()) > 10]
            
            if not valid_passages:
                logger.warning("No valid passages found for embedding")
                return np.array([])
            
            # Compute embeddings
            embeddings = self.embedding_model.encode(
                valid_passages,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return np.array([])
    
    def calculate_cosine_similarity(self, retrieved_embeddings: np.ndarray, 
                                  ground_truth_embeddings: np.ndarray) -> Tuple[List[float], Dict]:
        """Calculate cosine similarity between retrieved and ground truth embeddings"""
        if retrieved_embeddings.size == 0 or ground_truth_embeddings.size == 0:
            return [], {
                'max_similarity': 0.0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'best_match_idx': -1
            }
        
        try:
            # Calculate cosine similarity matrix
            # Shape: (retrieved_count, ground_truth_count)
            similarity_matrix = cosine_similarity(retrieved_embeddings, ground_truth_embeddings)
            
            # For each retrieved passage, find the best matching ground truth passage
            max_similarities = np.max(similarity_matrix, axis=1).tolist()
            best_match_indices = np.argmax(similarity_matrix, axis=1).tolist()
            
            # Overall statistics
            all_similarities = similarity_matrix.flatten()
            stats = {
                'max_similarity': float(np.max(all_similarities)),
                'avg_similarity': float(np.mean(all_similarities)),
                'min_similarity': float(np.min(all_similarities)),
                'best_match_idx': int(np.argmax(all_similarities) % ground_truth_embeddings.shape[0]),
                'similarity_matrix': similarity_matrix.tolist(),
                'max_similarities_per_retrieved': max_similarities,
                'best_match_indices': best_match_indices
            }
            
            return max_similarities, stats
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return [], {
                'max_similarity': 0.0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'best_match_idx': -1
            }
    
    def evaluate_single_question(self, question_data: Dict, question_id: int) -> EvaluationMetrics:
        """Evaluate a single question"""
        question = question_data['question']
        # Try 'answer' first, then 'predicted_answer' for backward compatibility
        ground_truth_answer = question_data.get('answer', question_data.get('predicted_answer', ''))
        
        logger.info(f"📝 Evaluating question {question_id + 1}: {question[:50]}...")
        
        metrics = EvaluationMetrics(question_id, question, ground_truth_answer)
        
        try:
            # Step 1: Retrieve passages using vanilla RAG system
            start_time = datetime.now()
            retrieved_passages, metadata = self.retrieve_passages(question)
            metrics.retrieved_passages = retrieved_passages
            metrics.metadata = metadata
            metrics.processing_time = (datetime.now() - start_time).total_seconds()
            
            if not retrieved_passages:
                logger.warning(f"No passages retrieved for question {question_id + 1}")
                metrics.error = "No passages retrieved"
                return metrics
            
            # Step 2: Compute embeddings for retrieved passages
            retrieved_embeddings = self.compute_embeddings(retrieved_passages)
            
            if retrieved_embeddings.size == 0:
                logger.warning(f"No valid embeddings for retrieved passages in question {question_id + 1}")
                metrics.error = "No valid embeddings for retrieved passages"
                return metrics
            
            # Step 3: Compute embeddings for ground truth answer
            ground_truth_embeddings = self.compute_embeddings([ground_truth_answer])
            
            if ground_truth_embeddings.size == 0:
                logger.warning(f"No valid embeddings for ground truth in question {question_id + 1}")
                metrics.error = "No valid embeddings for ground truth"
                return metrics
            
            # Step 4: Calculate cosine similarity
            similarity_scores, similarity_stats = self.calculate_cosine_similarity(
                retrieved_embeddings, ground_truth_embeddings
            )
            
            # Step 5: Update metrics
            metrics.similarity_scores = similarity_scores
            metrics.max_similarity = similarity_stats['max_similarity']
            metrics.avg_similarity = similarity_stats['avg_similarity']
            metrics.min_similarity = similarity_stats['min_similarity']
            metrics.best_match_idx = similarity_stats['best_match_idx']
            
            logger.info(f"✅ Question {question_id + 1} evaluated:")
            logger.info(f"   - Max similarity: {metrics.max_similarity:.4f}")
            logger.info(f"   - Avg similarity: {metrics.avg_similarity:.4f}")
            logger.info(f"   - Retrieved passages: {len(retrieved_passages)}")
            logger.info(f"   - Ground truth answer: ✅")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate question {question_id + 1}: {e}")
            metrics.error = str(e)
            return metrics
    
    def run_evaluation(self) -> str:
        """Run the complete evaluation"""
        logger.info("🚀 Starting Vanilla Cosine Similarity Evaluation...")
        
        try:
            # Load ground truth data
            ground_truth_data = self.load_ground_truth(self.ground_truth_file)
            
            # Limit questions if specified
            if MAX_QUESTIONS:
                ground_truth_data = ground_truth_data[:MAX_QUESTIONS]
                logger.info(f"⚠️ Limiting evaluation to first {MAX_QUESTIONS} questions")
            
            logger.info(f"🔍 Starting evaluation of {len(ground_truth_data)} questions...")
            
            # Evaluate each question
            results = []
            successful_evaluations = 0
            
            for i, question_data in enumerate(ground_truth_data):
                try:
                    metrics = self.evaluate_single_question(question_data, i)
                    results.append({
                        'question_id': metrics.question_id,
                        'question': metrics.question,
                        'ground_truth': metrics.ground_truth,
                        'retrieved_passages': metrics.retrieved_passages,
                        'max_similarity': metrics.max_similarity,
                        'avg_similarity': metrics.avg_similarity,
                        'min_similarity': metrics.min_similarity,
                        'similarity_scores': metrics.similarity_scores,
                        'best_match_idx': metrics.best_match_idx,
                        'metadata': metrics.metadata,
                        'processing_time': metrics.processing_time,
                        'error': metrics.error
                    })
                    
                    if metrics.error is None:
                        successful_evaluations += 1
                        
                except Exception as e:
                    logger.error(f"Failed to evaluate question {i + 1}: {e}")
                    results.append({
                        'question_id': i,
                        'question': question_data.get('question', ''),
                        'ground_truth': question_data.get('answer', ''),
                        'error': str(e)
                    })
            
            # Generate summary statistics
            summary = self._generate_summary(results)
            
            # Save results
            output_file = self._save_results(results, summary)
            
            logger.info(f"✅ Evaluation completed! {successful_evaluations} questions evaluated successfully")
            return output_file
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from evaluation results"""
        # Filter out failed evaluations
        valid_results = [r for r in results if r.get('error') is None]
        
        if not valid_results:
            return {
                'total_questions': len(results),
                'successful_evaluations': 0,
                'failed_evaluations': len(results),
                'error': "No successful evaluations"
            }
        
        # Extract similarity scores
        max_similarities = [r['max_similarity'] for r in valid_results]
        avg_similarities = [r['avg_similarity'] for r in valid_results]
        
        # Calculate statistics
        summary = {
            'total_questions': len(results),
            'successful_evaluations': len(valid_results),
            'failed_evaluations': len(results) - len(valid_results),
            
            # Max similarity statistics
            'max_similarity_stats': {
                'mean': float(np.mean(max_similarities)),
                'median': float(np.median(max_similarities)),
                'std_dev': float(np.std(max_similarities)),
                'min': float(np.min(max_similarities)),
                'max': float(np.max(max_similarities)),
                'q25': float(np.percentile(max_similarities, 25)),
                'q75': float(np.percentile(max_similarities, 75))
            },
            
            # Average similarity statistics
            'avg_similarity_stats': {
                'mean': float(np.mean(avg_similarities)),
                'median': float(np.median(avg_similarities)),
                'std_dev': float(np.std(avg_similarities)),
                'min': float(np.min(avg_similarities)),
                'max': float(np.max(avg_similarities)),
                'q25': float(np.percentile(avg_similarities, 25)),
                'q75': float(np.percentile(avg_similarities, 75))
            },
            
            # Similarity distribution
            'similarity_distribution': {
                'high_similarity': len([s for s in max_similarities if s >= 0.8]),
                'medium_similarity': len([s for s in max_similarities if 0.5 <= s < 0.8]),
                'low_similarity': len([s for s in max_similarities if s < 0.5])
            },
            
            # Passage statistics
            'passage_stats': {
                'avg_retrieved_passages': float(np.mean([len(r['retrieved_passages']) for r in valid_results])),
                'total_retrieved': sum(len(r['retrieved_passages']) for r in valid_results),
                'questions_with_answers': len(valid_results)
            },
            
            # Language distribution
            'language_distribution': self._analyze_language_distribution(valid_results),
            
            # Processing time statistics
            'processing_time_stats': {
                'mean': float(np.mean([r['processing_time'] for r in valid_results])),
                'median': float(np.median([r['processing_time'] for r in valid_results])),
                'total': float(np.sum([r['processing_time'] for r in valid_results]))
            }
        }
        
        return summary
    
    def _analyze_language_distribution(self, results: List[Dict]) -> Dict:
        """Analyze language distribution from metadata"""
        language_counts = {}
        total = len(results)
        
        for result in results:
            language = result.get('metadata', {}).get('detected_language', 'unknown')
            language_counts[language] = language_counts.get(language, 0) + 1
        
        # Convert to percentages
        language_distribution = {}
        for language, count in language_counts.items():
            language_distribution[language] = {
                'count': count,
                'percentage': round((count / total) * 100, 1)
            }
        
        return language_distribution
    
    def _save_results(self, results: List[Dict], summary: Dict) -> str:
        """Save evaluation results to JSON file"""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs('logs', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/vanilla_cosine_similarity_evaluation_{timestamp}.json"
            
            # Prepare output data
            output_data = {
                'evaluation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'embedding_model': EMBEDDING_MODEL,
                    'ground_truth_file': self.ground_truth_file,
                    'max_questions': MAX_QUESTIONS,
                    'evaluator_type': 'vanilla_rag'
                },
                'summary': summary,
                'detailed_results': results
            }
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saving results to: {filename}")
            logger.info(f"✅ Results saved successfully to: {filename}")
            
            # Print summary
            self._print_summary(summary)
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def _print_summary(self, summary: Dict):
        """Print evaluation summary"""
        print("\n" + "=" * 80)
        print("🔍 VANILLA COSINE SIMILARITY EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Questions Evaluated: {summary['total_questions']}")
        print(f"   Successful Evaluations: {summary['successful_evaluations']}")
        print(f"   Failed Evaluations: {summary['failed_evaluations']}")
        
        if 'similarity_distribution' in summary:
            dist = summary['similarity_distribution']
            total = summary['successful_evaluations']
            print(f"   High Similarity (≥0.8): {dist['high_similarity']} ({dist['high_similarity']/total*100:.1f}%)")
            print(f"   Medium Similarity (0.5-0.8): {dist['medium_similarity']} ({dist['medium_similarity']/total*100:.1f}%)")
            print(f"   Low Similarity (<0.5): {dist['low_similarity']} ({dist['low_similarity']/total*100:.1f}%)")
        
        if 'max_similarity_stats' in summary:
            stats = summary['max_similarity_stats']
            print(f"\n📈 Max Cosine Similarity Statistics:")
            print(f"   Mean: {stats['mean']:.4f}")
            print(f"   Median: {stats['median']:.4f}")
            print(f"   Std Dev: {stats['std_dev']:.4f}")
            print(f"   Min: {stats['min']:.4f}")
            print(f"   Max: {stats['max']:.4f}")
            print(f"   25th Percentile: {stats['q25']:.4f}")
            print(f"   75th Percentile: {stats['q75']:.4f}")
        
        if 'avg_similarity_stats' in summary:
            stats = summary['avg_similarity_stats']
            print(f"\n📈 Average Cosine Similarity Statistics:")
            print(f"   Mean: {stats['mean']:.4f}")
            print(f"   Median: {stats['median']:.4f}")
            print(f"   Std Dev: {stats['std_dev']:.4f}")
            print(f"   Range: {stats['min']:.4f} - {stats['max']:.4f}")
        
        if 'passage_stats' in summary:
            stats = summary['passage_stats']
            print(f"\n📄 Passage Statistics:")
            print(f"   Avg Retrieved Passages: {stats['avg_retrieved_passages']:.1f}")
            print(f"   Ground Truth: Answer (1 per question)")
            print(f"   Total Retrieved: {stats['total_retrieved']}")
            print(f"   Questions with Answers: {stats['questions_with_answers']}")
        
        if 'language_distribution' in summary:
            print(f"\n🌐 Language Distribution:")
            for lang, info in summary['language_distribution'].items():
                print(f"   {lang.capitalize()}: {info['count']} questions ({info['percentage']}%)")
        
        print("\n" + "=" * 80)

def list_available_files():
    """List available JSON files in the dataqa folder"""
    dataqa_dir = "dataqa"
    if not os.path.exists(dataqa_dir):
        print(f"❌ DataQA directory not found: {dataqa_dir}")
        return []
    
    json_files = [f for f in os.listdir(dataqa_dir) if f.endswith('.json')]
    if not json_files:
        print(f"❌ No JSON files found in {dataqa_dir}")
        return []
    
    print(f"\nAvailable files in {dataqa_dir}:")
    for i, file in enumerate(json_files, 1):
        print(f"  {i}. {file}")
    
    return json_files

def interactive_file_selection():
    """Interactive file selection"""
    files = list_available_files()
    if not files:
        return None
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(files)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(files):
                return f"dataqa/{files[choice_idx]}"
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Vanilla Cosine Similarity Evaluation for RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cosevanilla.py                    # Use default file
  python cosevanilla.py --interactive      # Interactive file selection
  python cosevanilla.py ENGqapair.json     # Use specific file
  python cosevanilla.py --list             # List available files
  python cosevanilla.py --max-questions 10 # Limit to 10 questions
        """
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='Ground truth JSON file (will look in dataqa folder)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive file selection'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available JSON files'
    )
    
    parser.add_argument(
        '--max-questions', '-m',
        type=int,
        help='Maximum number of questions to evaluate'
    )
    
    args = parser.parse_args()
    
    print("🔍 Vanilla Cosine Similarity Evaluation for RAG System")
    print("=" * 60)
    
    # Handle list files option
    if args.list:
        list_available_files()
        return 0
    
    # Handle interactive mode
    ground_truth_file = None
    if args.interactive:
        ground_truth_file = interactive_file_selection()
        if not ground_truth_file:
            print("❌ File selection cancelled")
            return 1
    elif args.file:
        # User specified a file - only look in dataqa folder
        specified_file = args.file
        
        # Always look in dataqa folder
        if not specified_file.startswith("dataqa/"):
            specified_file = f"dataqa/{specified_file}"
        
        # Add .json extension if not present
        if not specified_file.endswith('.json'):
            specified_file += '.json'
        
        if os.path.exists(specified_file):
            ground_truth_file = specified_file
        else:
            print(f"❌ Error: File '{specified_file}' not found in dataqa folder!")
            print("\nAvailable files in dataqa folder:")
            list_available_files()
            return 1
    else:
        # No file specified, show available files and ask user
        print("No file specified. Available files in dataqa folder:")
        available_files = list_available_files()
        if not available_files:
            return 1
        
        print(f"\nChoose an option:")
        print(f"1. Use default file: {DEFAULT_GROUND_TRUTH_FILE}")
        print(f"2. Run interactive mode: python cosevanilla.py --interactive")
        print(f"3. Specify a file: python cosevanilla.py <filename>")
        
        # Use default file (must be in dataqa folder)
        if os.path.exists(DEFAULT_GROUND_TRUTH_FILE):
            ground_truth_file = DEFAULT_GROUND_TRUTH_FILE
        else:
            print(f"❌ Default file {DEFAULT_GROUND_TRUTH_FILE} not found in dataqa folder!")
            return 1
    
    # Set max questions if specified
    global MAX_QUESTIONS
    if args.max_questions:
        MAX_QUESTIONS = args.max_questions
    
    print(f"Ground Truth File: {ground_truth_file}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    if MAX_QUESTIONS:
        print(f"Max Questions: {MAX_QUESTIONS}")
    print("=" * 60)
    
    try:
        # Initialize and run evaluator
        evaluator = CosineSimilarityEvaluator(ground_truth_file)
        results_file = evaluator.run_evaluation()
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
