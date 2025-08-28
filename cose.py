#!/usr/bin/env python3
"""
🔍 Cosine Similarity Evaluation Script for RAG System
====================================================

This script evaluates the RAG system by comparing the cosine similarity between:
- Retrieved passage embeddings from the RAG system
- Ground truth answer embedding from the evaluation dataset

Features:
- Uses the same embedding model as main2.py (BAAI/bge-m3)
- Loads ground truth data from JSON files in dataqa folder or other locations
- Interactive file selection from available JSON files
- Retrieves passages using the RRF Fusion RAG system
- Computes cosine similarity between retrieved passages and answer
- Provides detailed evaluation metrics and saves results to JSON
- Enhanced error handling and memory management

Author: RAG Evaluation System
Version: 2.1 - Enhanced with better error handling and memory management

Usage:
# Interactive mode (recommended)
python cose.py --interactive

# List available files
python cose.py --list

# Direct file selection
python cose.py ENGqapair.json
python cose.py BNqapair.json
python cose.py dataqa/ENGqapair.json

# Limit number of questions for testing
python cose.py ENGqapair.json --max-questions 10

# Get help
python cose.py --help
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
import gc
import sys

# Core ML/NLP Libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import faiss
except ImportError as e:
    print(f"❌ Required library not found: {e}")
    print("Please install required packages: pip install sentence-transformers scikit-learn faiss-cpu")
    sys.exit(1)

# Import RAG system components from main2.py
try:
    from main2 import CoreRAGSystem, config
except ImportError as e:
    print(f"❌ Failed to import RAG system from main2.py: {e}")
    print("Please ensure main2.py is in the same directory and all dependencies are installed")
    sys.exit(1)

# =============================================================================
# 📊 CONFIGURATION
# =============================================================================

# Evaluation Configuration
DEFAULT_GROUND_TRUTH_FILE = "dataqa/ENGqapair.json"  # Default file if none specified
EMBEDDING_MODEL = "BAAI/bge-m3"  # Same as main2.py
MAX_QUESTIONS = None  # Set to None to evaluate all questions, or a number to limit

# Memory Management
MAX_MEMORY_USAGE_MB = 8000  # 8GB limit for VRAM
BATCH_SIZE = 32  # Process embeddings in batches to manage memory

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 📊 EVALUATION METRICS
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    question_id: int
    question: str
    max_cosine_similarity: float
    avg_cosine_similarity: float
    min_cosine_similarity: float
    retrieved_passages_count: int
    ground_truth_passages_count: int
    similarity_scores: List[float]
    best_match_idx: int
    retrieved_passages: List[str]
    ground_truth_passages: List[str]
    detected_language: str
    retrieval_method: str
    error: Optional[str] = None
    processing_time: float = 0.0

# =============================================================================
# 🔍 COSINE SIMILARITY EVALUATOR
# =============================================================================

class CosineSimilarityEvaluator:
    """Evaluates RAG system using cosine similarity between retrieved passages and ground truth answer"""
    
    def __init__(self, ground_truth_file: str = None):
        """Initialize the evaluator
        
        Args:
            ground_truth_file: Path to the ground truth JSON file (default: uses DEFAULT_GROUND_TRUTH_FILE)
        """
        logger.info("🚀 Initializing Cosine Similarity Evaluator...")
        
        # Set ground truth file
        self.ground_truth_file = ground_truth_file or DEFAULT_GROUND_TRUTH_FILE
        logger.info(f"📄 Ground truth file: {self.ground_truth_file}")
        
        # Initialize components
        self.embedding_model = None
        self.rag_system = None
        
        # Initialize components with error handling
        self._init_embedding_model()
        self._init_rag_system()
        
        logger.info("✅ Cosine Similarity Evaluator initialized successfully")
    
    def _init_embedding_model(self):
        """Initialize the embedding model with memory management"""
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            
            # Check available memory before loading
            self._check_memory_availability()
            
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
            
            # Test the model with a small sample
            test_embeddings = self.embedding_model.encode(["test"], normalize_embeddings=True)
            logger.info(f"✅ Embedding model loaded successfully (test embedding shape: {test_embeddings.shape})")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _init_rag_system(self):
        """Initialize the RAG system for document retrieval with error handling"""
        try:
            logger.info("Initializing RAG system...")
            self.rag_system = CoreRAGSystem()
            
            # Verify system is ready with detailed checks
            info = self.rag_system.get_system_info()
            
            if not info['faiss_index_loaded']:
                raise Exception("FAISS index not loaded - check if faiss_index directory exists")
            if not info['embedding_model_loaded']:
                raise Exception("Embedding model not loaded in RAG system")
            
            logger.info("✅ RAG system initialized successfully")
            logger.info(f"   - Documents: {info['total_vectors']:,}")
            logger.info(f"   - FAISS Index: {'✅' if info['faiss_index_loaded'] else '❌'}")
            logger.info(f"   - Embedding Model: {'✅' if info['embedding_model_loaded'] else '❌'}")
            logger.info(f"   - Cross-Encoder: {'✅' if info['cross_encoder_loaded'] else '❌'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            logger.error("Please ensure:")
            logger.error("1. FAISS index exists in faiss_index/ directory")
            logger.error("2. All required models are downloaded")
            logger.error("3. main2.py is properly configured")
            raise
    
    def _check_memory_availability(self):
        """Check if there's enough memory available"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            if available_mb < MAX_MEMORY_USAGE_MB:
                logger.warning(f"⚠️ Low memory available: {available_mb:.1f}MB (recommended: {MAX_MEMORY_USAGE_MB}MB)")
                logger.warning("Consider reducing batch size or closing other applications")
            else:
                logger.info(f"✅ Memory available: {available_mb:.1f}MB")
                
        except ImportError:
            logger.warning("psutil not available - cannot check memory usage")
    
    def load_ground_truth(self, file_path: str) -> List[Dict]:
        """Load ground truth data from JSON file with validation"""
        try:
            logger.info(f"Loading ground truth data from: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Ground truth file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, list):
                raise ValueError("Ground truth file should contain a list of questions")
            
            if len(data) == 0:
                raise ValueError("Ground truth file is empty")
            
            # Validate first item structure
            first_item = data[0]
            if not isinstance(first_item, dict):
                raise ValueError("Each item should be a dictionary")
            
            if 'question' not in first_item:
                raise ValueError("Each item should have a 'question' field")
            
            if 'answer' not in first_item and 'predicted_answer' not in first_item:
                raise ValueError("Each item should have either 'answer' or 'predicted_answer' field")
            
            logger.info(f"✅ Loaded {len(data)} questions from ground truth data")
            logger.info(f"   - Sample question: {first_item['question'][:50]}...")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load ground truth data: {e}")
            raise
    
    def retrieve_passages(self, question: str) -> Tuple[List[str], Dict]:
        """Retrieve passages for a question using the RAG system with error handling"""
        try:
            # Process query using RAG system
            result = self.rag_system.process_query_sync(question)
            
            # Extract retrieved passages from contexts
            retrieved_passages = result.get('contexts', [])
            
            # Validate retrieved passages
            if not retrieved_passages:
                logger.warning(f"No passages retrieved for question: {question[:50]}...")
            
            # Get metadata
            metadata = {
                'detected_language': result.get('detected_language', 'unknown'),
                'retrieval_method': result.get('retrieval_method', 'unknown'),
                'documents_found': result.get('documents_found', 0),
                'documents_used': result.get('documents_used', 0),
                'cross_encoder_used': result.get('cross_encoder_used', False)
            }
            
            return retrieved_passages, metadata
            
        except Exception as e:
            logger.error(f"Failed to retrieve passages for question: {question[:50]}... Error: {e}")
            return [], {}
    
    def compute_embeddings(self, passages: List[str]) -> np.ndarray:
        """Compute embeddings for a list of passages with memory management"""
        if not passages:
            return np.array([])
        
        try:
            # Filter out empty or very short passages
            valid_passages = [p.strip() for p in passages if p.strip() and len(p.strip()) > 10]
            
            if not valid_passages:
                logger.warning("No valid passages found for embedding")
                return np.array([])
            
            # Process in batches to manage memory
            all_embeddings = []
            for i in range(0, len(valid_passages), BATCH_SIZE):
                batch = valid_passages[i:i + BATCH_SIZE]
                
                # Compute embeddings for batch
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                all_embeddings.append(batch_embeddings)
                
                # Force garbage collection after each batch
                gc.collect()
            
            # Combine all batches
            if all_embeddings:
                embeddings = np.vstack(all_embeddings)
                logger.debug(f"Computed embeddings for {len(valid_passages)} passages (shape: {embeddings.shape})")
                return embeddings
            else:
                return np.array([])
            
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
        """Evaluate a single question with comprehensive error handling"""
        question = question_data['question']
        # Try 'answer' first, then 'predicted_answer' for backward compatibility
        ground_truth_answer = question_data.get('answer', question_data.get('predicted_answer', ''))
        
        logger.info(f"📝 Evaluating question {question_id + 1}: {question[:50]}...")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve passages using RAG system
            retrieved_passages, metadata = self.retrieve_passages(question)
            
            # Step 2: Compute embeddings for retrieved passages
            retrieved_embeddings = self.compute_embeddings(retrieved_passages)
            
            # Step 3: Compute embedding for ground truth answer
            ground_truth_embeddings = self.compute_embeddings([ground_truth_answer]) if ground_truth_answer else np.array([])
            
            # Step 4: Calculate cosine similarity
            max_similarities, similarity_stats = self.calculate_cosine_similarity(
                retrieved_embeddings, ground_truth_embeddings
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create evaluation metrics
            metrics = EvaluationMetrics(
                question_id=question_id,
                question=question,
                max_cosine_similarity=similarity_stats.get('max_similarity', 0.0),
                avg_cosine_similarity=similarity_stats.get('avg_similarity', 0.0),
                min_cosine_similarity=similarity_stats.get('min_similarity', 0.0),
                retrieved_passages_count=len(retrieved_passages),
                ground_truth_passages_count=1 if ground_truth_answer else 0,
                similarity_scores=max_similarities,
                best_match_idx=similarity_stats.get('best_match_idx', -1),
                retrieved_passages=retrieved_passages,
                ground_truth_passages=[ground_truth_answer] if ground_truth_answer else [],
                detected_language=metadata.get('detected_language', 'unknown'),
                retrieval_method=metadata.get('retrieval_method', 'unknown'),
                processing_time=processing_time
            )
            
            logger.info(f"✅ Question {question_id + 1} evaluated:")
            logger.info(f"   - Max similarity: {metrics.max_cosine_similarity:.4f}")
            logger.info(f"   - Avg similarity: {metrics.avg_cosine_similarity:.4f}")
            logger.info(f"   - Retrieved passages: {metrics.retrieved_passages_count}")
            logger.info(f"   - Ground truth answer: {'✅' if ground_truth_answer else '❌'}")
            logger.info(f"   - Processing time: {processing_time:.2f}s")
            
            return metrics
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Failed to evaluate question {question_id + 1}: {e}")
            
            # Return error metrics
            return EvaluationMetrics(
                question_id=question_id,
                question=question,
                max_cosine_similarity=0.0,
                avg_cosine_similarity=0.0,
                min_cosine_similarity=0.0,
                retrieved_passages_count=0,
                ground_truth_passages_count=0,
                similarity_scores=[],
                best_match_idx=-1,
                retrieved_passages=[],
                ground_truth_passages=[],
                detected_language='unknown',
                retrieval_method='error',
                error=str(e),
                processing_time=processing_time
            )
    
    def evaluate_all_questions(self, ground_truth_data: List[Dict]) -> List[EvaluationMetrics]:
        """Evaluate all questions in the ground truth data with progress tracking"""
        logger.info(f"🔍 Starting evaluation of {len(ground_truth_data)} questions...")
        
        all_metrics = []
        
        # Limit questions if specified
        questions_to_evaluate = ground_truth_data
        if MAX_QUESTIONS and MAX_QUESTIONS < len(ground_truth_data):
            questions_to_evaluate = ground_truth_data[:MAX_QUESTIONS]
            logger.info(f"⚠️ Limiting evaluation to first {MAX_QUESTIONS} questions")
        
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, question_data in enumerate(questions_to_evaluate):
            try:
                metrics = self.evaluate_single_question(question_data, i)
                all_metrics.append(metrics)
                
                if metrics.error is None:
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
                
                # Progress update every 10 questions
                if (i + 1) % 10 == 0:
                    avg_max_sim = np.mean([m.max_cosine_similarity for m in all_metrics if m.error is None])
                    logger.info(f"📊 Progress: {i + 1}/{len(questions_to_evaluate)} questions")
                    logger.info(f"   - Successful: {successful_evaluations}, Failed: {failed_evaluations}")
                    logger.info(f"   - Avg Max Similarity: {avg_max_sim:.4f}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to evaluate question {i + 1}: {e}")
                failed_evaluations += 1
                continue
        
        logger.info(f"✅ Evaluation completed!")
        logger.info(f"   - Total questions: {len(questions_to_evaluate)}")
        logger.info(f"   - Successful evaluations: {successful_evaluations}")
        logger.info(f"   - Failed evaluations: {failed_evaluations}")
        
        return all_metrics
    
    def calculate_overall_statistics(self, all_metrics: List[EvaluationMetrics]) -> Dict:
        """Calculate overall evaluation statistics with error handling"""
        if not all_metrics:
            return {}
        
        # Filter out failed evaluations
        successful_metrics = [m for m in all_metrics if m.error is None]
        failed_metrics = [m for m in all_metrics if m.error is not None]
        
        if not successful_metrics:
            return {
                'total_questions': len(all_metrics),
                'successful_evaluations': 0,
                'failed_evaluations': len(failed_metrics),
                'error': "No successful evaluations"
            }
        
        # Extract key metrics from successful evaluations
        max_similarities = [m.max_cosine_similarity for m in successful_metrics]
        avg_similarities = [m.avg_cosine_similarity for m in successful_metrics]
        retrieved_counts = [m.retrieved_passages_count for m in successful_metrics]
        ground_truth_counts = [m.ground_truth_passages_count for m in successful_metrics]
        processing_times = [m.processing_time for m in successful_metrics]
        
        # Calculate statistics
        stats = {
            'total_questions': len(all_metrics),
            'successful_evaluations': len(successful_metrics),
            'failed_evaluations': len(failed_metrics),
            'success_rate': len(successful_metrics) / len(all_metrics),
            'max_cosine_similarity': {
                'mean': float(np.mean(max_similarities)),
                'std': float(np.std(max_similarities)),
                'min': float(np.min(max_similarities)),
                'max': float(np.max(max_similarities)),
                'median': float(np.median(max_similarities)),
                'percentile_25': float(np.percentile(max_similarities, 25)),
                'percentile_75': float(np.percentile(max_similarities, 75))
            },
            'avg_cosine_similarity': {
                'mean': float(np.mean(avg_similarities)),
                'std': float(np.std(avg_similarities)),
                'min': float(np.min(avg_similarities)),
                'max': float(np.max(avg_similarities)),
                'median': float(np.median(avg_similarities))
            },
            'retrieved_passages': {
                'mean': float(np.mean(retrieved_counts)),
                'std': float(np.std(retrieved_counts)),
                'min': int(np.min(retrieved_counts)),
                'max': int(np.max(retrieved_counts)),
                'total': int(np.sum(retrieved_counts))
            },
            'ground_truth_passages': {
                'mean': float(np.mean(ground_truth_counts)),
                'std': float(np.std(ground_truth_counts)),
                'min': int(np.min(ground_truth_counts)),
                'max': int(np.max(ground_truth_counts)),
                'total': int(np.sum(ground_truth_counts))
            },
            'processing_time': {
                'mean': float(np.mean(processing_times)),
                'median': float(np.median(processing_times)),
                'min': float(np.min(processing_times)),
                'max': float(np.max(processing_times)),
                'total': float(np.sum(processing_times))
            },
            'high_similarity_questions': len([m for m in successful_metrics if m.max_cosine_similarity >= 0.8]),
            'medium_similarity_questions': len([m for m in successful_metrics if 0.5 <= m.max_cosine_similarity < 0.8]),
            'low_similarity_questions': len([m for m in successful_metrics if m.max_cosine_similarity < 0.5]),
            'language_distribution': {
                lang: len([m for m in successful_metrics if m.detected_language == lang])
                for lang in set(m.detected_language for m in successful_metrics)
            },
            'retrieval_method_distribution': {
                method: len([m for m in successful_metrics if m.retrieval_method == method])
                for method in set(m.retrieval_method for m in successful_metrics)
            }
        }
        
        # Add error summary if there are failed evaluations
        if failed_metrics:
            error_types = {}
            for metric in failed_metrics:
                error_type = metric.error.split(':')[0] if ':' in metric.error else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1
            stats['error_summary'] = error_types
        
        return stats
    
    def save_results(self, all_metrics: List[EvaluationMetrics], overall_stats: Dict, 
                    output_file: str):
        """Save evaluation results to JSON file with enhanced error information"""
        try:
            logger.info(f"💾 Saving results to: {output_file}")
            
            # Convert metrics to dictionary format
            results_data = {
                'evaluation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'ground_truth_file': self.ground_truth_file,
                    'embedding_model': EMBEDDING_MODEL,
                    'rag_system_config': {
                        'faiss_index_path': getattr(config, 'FAISS_INDEX_PATH', 'faiss_index'),
                        'embedding_model': getattr(config, 'EMBEDDING_MODEL', EMBEDDING_MODEL),
                        'cross_encoder_model': getattr(config, 'CROSS_ENCODER_MODEL', 'BAAI/bge-reranker-v2-m3'),
                        'max_docs_retrieval': getattr(config, 'MAX_DOCS_FOR_RETRIEVAL', 30),
                        'max_docs_context': getattr(config, 'MAX_DOCS_FOR_CONTEXT', 6),
                        'rrf_weights': getattr(config, 'RRF_WEIGHTS', {'faiss': 0.3, 'bm25': 0.25, 'colbert': 0.25, 'dirichlet': 0.2})
                    },
                    'evaluation_settings': {
                        'max_questions': MAX_QUESTIONS,
                        'total_evaluated': len(all_metrics),
                        'batch_size': BATCH_SIZE,
                        'memory_limit_mb': MAX_MEMORY_USAGE_MB
                    }
                },
                'overall_statistics': overall_stats,
                'detailed_results': []
            }
            
            # Add detailed results
            for metrics in all_metrics:
                result_item = {
                    'question_id': metrics.question_id,
                    'question': metrics.question,
                    'cosine_similarity_metrics': {
                        'max_similarity': metrics.max_cosine_similarity,
                        'avg_similarity': metrics.avg_cosine_similarity,
                        'min_similarity': metrics.min_cosine_similarity,
                        'similarity_scores': metrics.similarity_scores
                    },
                    'passage_counts': {
                        'retrieved': metrics.retrieved_passages_count,
                        'ground_truth': metrics.ground_truth_passages_count
                    },
                    'best_match_idx': metrics.best_match_idx,
                    'metadata': {
                        'detected_language': metrics.detected_language,
                        'retrieval_method': metrics.retrieval_method,
                        'processing_time': metrics.processing_time
                    },
                    'passages': {
                        'retrieved': metrics.retrieved_passages,
                        'ground_truth': metrics.ground_truth_passages
                    }
                }
                
                # Add error information if present
                if metrics.error:
                    result_item['error'] = metrics.error
                
                results_data['detailed_results'].append(result_item)
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Results saved successfully to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def print_summary(self, overall_stats: Dict):
        """Print evaluation summary with enhanced statistics"""
        print("\n" + "="*80)
        print("🔍 COSINE SIMILARITY EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Questions Evaluated: {overall_stats['total_questions']}")
        print(f"   Successful Evaluations: {overall_stats['successful_evaluations']}")
        print(f"   Failed Evaluations: {overall_stats['failed_evaluations']}")
        print(f"   Success Rate: {overall_stats['success_rate']*100:.1f}%")
        
        if overall_stats['successful_evaluations'] > 0:
            print(f"\n📊 Similarity Distribution (Successful Evaluations):")
            total_successful = overall_stats['successful_evaluations']
            print(f"   High Similarity (≥0.8): {overall_stats['high_similarity_questions']} ({overall_stats['high_similarity_questions']/total_successful*100:.1f}%)")
            print(f"   Medium Similarity (0.5-0.8): {overall_stats['medium_similarity_questions']} ({overall_stats['medium_similarity_questions']/total_successful*100:.1f}%)")
            print(f"   Low Similarity (<0.5): {overall_stats['low_similarity_questions']} ({overall_stats['low_similarity_questions']/total_successful*100:.1f}%)")
        
        if 'error_summary' in overall_stats:
            print(f"\n❌ Error Summary:")
            for error_type, count in overall_stats['error_summary'].items():
                print(f"   {error_type}: {count} occurrences")
        
        if overall_stats['successful_evaluations'] > 0:
            print(f"\n📈 Max Cosine Similarity Statistics:")
            max_stats = overall_stats['max_cosine_similarity']
            print(f"   Mean: {max_stats['mean']:.4f}")
            print(f"   Median: {max_stats['median']:.4f}")
            print(f"   Std Dev: {max_stats['std']:.4f}")
            print(f"   Min: {max_stats['min']:.4f}")
            print(f"   Max: {max_stats['max']:.4f}")
            print(f"   25th Percentile: {max_stats['percentile_25']:.4f}")
            print(f"   75th Percentile: {max_stats['percentile_75']:.4f}")
            
            print(f"\n📈 Average Cosine Similarity Statistics:")
            avg_stats = overall_stats['avg_cosine_similarity']
            print(f"   Mean: {avg_stats['mean']:.4f}")
            print(f"   Median: {avg_stats['median']:.4f}")
            print(f"   Std Dev: {avg_stats['std']:.4f}")
            print(f"   Range: {avg_stats['min']:.4f} - {avg_stats['max']:.4f}")
            
            print(f"\n⏱️ Processing Time Statistics:")
            time_stats = overall_stats['processing_time']
            print(f"   Mean: {time_stats['mean']:.2f}s")
            print(f"   Median: {time_stats['median']:.2f}s")
            print(f"   Min: {time_stats['min']:.2f}s")
            print(f"   Max: {time_stats['max']:.2f}s")
            print(f"   Total: {time_stats['total']:.2f}s")
            
            print(f"\n📄 Passage Statistics:")
            ret_stats = overall_stats['retrieved_passages']
            gt_stats = overall_stats['ground_truth_passages']
            print(f"   Avg Retrieved Passages: {ret_stats['mean']:.1f}")
            print(f"   Ground Truth: Answer (1 per question)")
            print(f"   Total Retrieved: {ret_stats['total']}")
            print(f"   Questions with Answers: {gt_stats['total']}")
            
            print(f"\n🌐 Language Distribution:")
            for lang, count in overall_stats['language_distribution'].items():
                print(f"   {lang.title()}: {count} questions ({count/total_successful*100:.1f}%)")
            
            if 'retrieval_method_distribution' in overall_stats:
                print(f"\n🔍 Retrieval Method Distribution:")
                for method, count in overall_stats['retrieval_method_distribution'].items():
                    print(f"   {method}: {count} questions ({count/total_successful*100:.1f}%)")
        
        print("\n" + "="*80)
    
    def run_evaluation(self) -> str:
        """Run the complete evaluation process"""
        try:
            logger.info("🚀 Starting Cosine Similarity Evaluation...")
            
            # Generate output filename based on input file
            input_filename = os.path.splitext(os.path.basename(self.ground_truth_file))[0]
            output_file = f"{input_filename}_cosine_similarity_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Step 1: Load ground truth data
            ground_truth_data = self.load_ground_truth(self.ground_truth_file)
            
            # Step 2: Evaluate all questions
            all_metrics = self.evaluate_all_questions(ground_truth_data)
            
            if not all_metrics:
                raise Exception("No questions were successfully evaluated")
            
            # Step 3: Calculate overall statistics
            overall_stats = self.calculate_overall_statistics(all_metrics)
            
            # Step 4: Save results
            self.save_results(all_metrics, overall_stats, output_file)
            
            # Step 5: Print summary
            self.print_summary(overall_stats)
            
            logger.info("✅ Evaluation completed successfully!")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise

# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

def list_available_files():
    """List available JSON files in the dataqa folder only"""
    all_files = []
    
    # Check dataqa folder only
    dataqa_folder = "dataqa"
    if not os.path.exists(dataqa_folder):
        print(f"❌ Error: {dataqa_folder} folder not found!")
        return []
    
    dataqa_files = [f for f in os.listdir(dataqa_folder) if f.endswith('.json')]
    for file in dataqa_files:
        all_files.append(os.path.join(dataqa_folder, file))
    
    if not all_files:
        print(f"❌ No JSON files found in {dataqa_folder} folder!")
        return []
    
    print(f"📁 Available JSON files in {dataqa_folder} folder:")
    for i, file_path in enumerate(all_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if it has question-answer structure
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    if 'question' in data[0] and ('answer' in data[0] or 'predicted_answer' in data[0]):
                        answer_field = 'answer' if 'answer' in data[0] else 'predicted_answer'
                        print(f"   {i}. {file_path} ({len(data)} questions, using '{answer_field}' field)")
                    else:
                        print(f"   {i}. {file_path} ({len(data)} items, structure unknown)")
                else:
                    print(f"   {i}. {file_path} (structure unknown)")
        except Exception as e:
            print(f"   {i}. {file_path} (Error reading: {e})")
    
    return all_files


def interactive_file_selection():
    """Interactive mode for file selection from dataqa folder"""
    print("🔍 Cosine Similarity Evaluation - Interactive Mode")
    print("=" * 60)
    
    # List available files from dataqa folder
    available_files = list_available_files()
    if not available_files:
        return None
    
    while True:
        try:
            print(f"\n📥 Select evaluation file from dataqa folder (1-{len(available_files)}):")
            choice = input("Enter your choice: ").strip()
            
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(available_files):
                    selected_file = available_files[index]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(available_files)}")
            else:
                print("❌ Please enter a valid number.")
        except ValueError:
            print("❌ Invalid input. Please try again.")
    
    # Verify the file has the right structure
    try:
        with open(selected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            print("❌ Error: File should contain a list of questions")
            return None
        
        sample = data[0]
        if not isinstance(sample, dict) or 'question' not in sample:
            print("❌ Error: File should contain question-answer pairs with 'question' field")
            return None
        
        if 'answer' not in sample and 'predicted_answer' not in sample:
            print("❌ Error: File should contain 'answer' or 'predicted_answer' field")
            return None
        
        answer_field = 'answer' if 'answer' in sample else 'predicted_answer'
        print(f"✅ Selected file: {selected_file}")
        print(f"   - Questions: {len(data)}")
        print(f"   - Answer field: '{answer_field}'")
        
        return selected_file
        
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return None

def main():
    """Main function to run the cosine similarity evaluation"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="🔍 Cosine Similarity Evaluation for RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cose.py --interactive
  python cose.py ENGqapair.json
  python cose.py BNqapair.json
  python cose.py --list
  python cose.py ENGqapair.json --max-questions 10
        """
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='JSON file to evaluate (optional)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode to select files'
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
    
    print("🔍 Cosine Similarity Evaluation for RAG System")
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
        print(f"2. Run interactive mode: python cose.py --interactive")
        print(f"3. Specify a file: python cose.py <filename>")
        
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
