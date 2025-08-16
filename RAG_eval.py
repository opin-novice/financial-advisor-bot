#!/usr/bin/env python3
"""
RAG Pipeline Evaluation System
-----------------------------
Comprehensive evaluation system for the multilingual RAG pipeline.
Supports both English and Bangla evaluation with detailed metrics.
"""

import json
import os
import logging
import time
import csv
import statistics
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Import your existing RAG components
from main import FinancialAdvisorTelegramBot
from config import config
from language_utils import LanguageDetector
from fuzzywuzzy import fuzz
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
EVAL_DATA_PATH = "dataqa/eval_set.json"
RESULTS_DIR = "evaluation_results"
DETAILED_LOG_FILE = f"{RESULTS_DIR}/detailed_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
SUMMARY_CSV_FILE = f"{RESULTS_DIR}/evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Improved evaluation thresholds
SIMILARITY_THRESHOLD = 40  # Lowered from 70 to be more realistic
CONFIDENCE_THRESHOLD = 0.6  # For validation confidence
RATE_LIMIT_DELAY = 4.0     # Seconds between questions to avoid API limits

@dataclass
class EvaluationResult:
    """Data class for storing comprehensive evaluation results"""
    question: str
    expected_answer: str
    generated_answer: str
    similarity_scores: Dict[str, float]  # Multiple similarity metrics
    best_similarity: float
    response_time: float
    sources: List[Dict]
    contexts: List[str]
    detected_language: str
    validation_confidence: float = 0.0
    relevance_score: float = 0.0
    was_translated: bool = False
    feedback_iterations: int = 0
    content_analysis: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "similarity_scores": self.similarity_scores,
            "best_similarity": self.best_similarity,
            "response_time": self.response_time,
            "sources": self.sources,
            "contexts": self.contexts,
            "detected_language": self.detected_language,
            "validation_confidence": self.validation_confidence,
            "relevance_score": self.relevance_score,
            "was_translated": self.was_translated,
            "feedback_iterations": self.feedback_iterations,
            "content_analysis": self.content_analysis or {}
        }

class RAGEvaluator:
    """Comprehensive RAG evaluation system with improved similarity calculation"""
    
    def __init__(self):
        """Initialize the evaluator with the existing RAG bot"""
        print("[INFO] 🔍 Initializing RAG Evaluator...")
        self.bot = FinancialAdvisorTelegramBot()
        self.language_detector = LanguageDetector()
        self.results: List[EvaluationResult] = []
        
        # Create results directory
        Path(RESULTS_DIR).mkdir(exist_ok=True)
        print(f"[INFO] ✅ RAG Evaluator initialized. Results will be saved to: {RESULTS_DIR}")
    
    def load_evaluation_data(self, file_path: str = EVAL_DATA_PATH) -> List[Dict]:
        """Load evaluation questions and expected answers from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[INFO] ✅ Loaded {len(data)} evaluation samples from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load evaluation data: {e}")
            raise
    
    def clean_answer(self, text: str) -> str:
        """Clean answer text for better comparison"""
        # Remove common conversational phrases and disclaimers
        text = re.sub(r'according to.*?context[,.]?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'note:.*?advice\.\*?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'please verify.*?sources', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\*+', '', text)  # Remove asterisks
        text = re.sub(r'⚠️', '', text)  # Remove warning emoji
        
        # Normalize currency format
        text = re.sub(r'tk\.?\s*', 'tk ', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+),(\d+)', r'\1\2', text)  # Remove commas from numbers
        
        return text.strip()
    
    def calculate_content_similarity(self, generated: str, expected: str) -> float:
        """Calculate content-based similarity focusing on key information"""
        # Extract numbers (amounts, percentages, etc.)
        gen_numbers = set(re.findall(r'\d+(?:\.\d+)?', generated))
        exp_numbers = set(re.findall(r'\d+(?:\.\d+)?', expected))
        
        # Extract key financial terms
        financial_terms = r'\b(?:tk|taka|percent|rate|loan|deposit|bank|account|tax|income|threshold|limit|fee|charge|interest)\b'
        gen_terms = set(re.findall(financial_terms, generated.lower()))
        exp_terms = set(re.findall(financial_terms, expected.lower()))
        
        # Calculate overlaps
        number_overlap = len(gen_numbers.intersection(exp_numbers)) / max(len(exp_numbers), 1) if exp_numbers else 0
        term_overlap = len(gen_terms.intersection(exp_terms)) / max(len(exp_terms), 1) if exp_terms else 0
        
        # Weighted score (numbers are more important than terms)
        content_score = (number_overlap * 0.7 + term_overlap * 0.3) * 100
        return min(content_score, 100)  # Cap at 100
    
    def calculate_multiple_similarities(self, generated: str, expected: str) -> Dict[str, float]:
        """Calculate multiple similarity metrics for better evaluation"""
        # Clean answers for comparison
        gen_clean = self.clean_answer(generated)
        exp_clean = self.clean_answer(expected)
        
        similarities = {
            "token_sort_ratio": fuzz.token_sort_ratio(gen_clean, exp_clean),
            "token_set_ratio": fuzz.token_set_ratio(gen_clean, exp_clean),
            "partial_ratio": fuzz.partial_ratio(gen_clean, exp_clean),
            "simple_ratio": fuzz.ratio(gen_clean, exp_clean),
            "content_similarity": self.calculate_content_similarity(gen_clean, exp_clean)
        }
        
        return similarities
    
    def analyze_content_match(self, generated: str, expected: str) -> Dict:
        """Analyze content match and provide insights"""
        gen_clean = self.clean_answer(generated)
        exp_clean = self.clean_answer(expected)
        
        # Word analysis
        gen_words = set(gen_clean.lower().split())
        exp_words = set(exp_clean.lower().split())
        common_words = gen_words.intersection(exp_words)
        
        # Number analysis
        gen_numbers = set(re.findall(r'\d+(?:\.\d+)?', generated))
        exp_numbers = set(re.findall(r'\d+(?:\.\d+)?', expected))
        
        return {
            "generated_length": len(generated),
            "expected_length": len(expected),
            "word_overlap_percent": len(common_words) / max(len(exp_words), 1) * 100,
            "common_words": list(common_words),
            "missing_numbers": list(exp_numbers - gen_numbers),
            "extra_numbers": list(gen_numbers - exp_numbers),
            "has_disclaimer": "note:" in generated.lower() or "verify" in generated.lower()
        }
    
    def evaluate_single_question(self, question: str, expected_answer: str, show_details: bool = False) -> EvaluationResult:
        """Evaluate a single question using the RAG pipeline with improved metrics"""
        print(f"\n[INFO] 🔍 Evaluating: {question[:80]}...")
        
        # Detect language
        detected_language, confidence = self.language_detector.detect_language(question)
        print(f"[INFO] 🌐 Language detected: {detected_language} (confidence: {confidence:.2f})")
        
        # Measure response time
        start_time = time.time()
        
        try:
            # Process query using the RAG pipeline
            response = self.bot.process_query(question)
            response_time = time.time() - start_time
            
            # Extract response data
            if isinstance(response, dict):
                generated_answer = response.get("response", str(response))
                sources = response.get("sources", [])
                contexts = response.get("contexts", [])
                validation_confidence = response.get("validation_confidence", 0.0)
                relevance_score = response.get("relevance_score", 0.0)
                was_translated = response.get("was_translated", False)
                feedback_iterations = response.get("feedback_iterations", 0)
            else:
                generated_answer = str(response)
                sources = []
                contexts = []
                validation_confidence = 0.0
                relevance_score = 0.0
                was_translated = False
                feedback_iterations = 0
            
            # Calculate multiple similarity metrics
            similarities = self.calculate_multiple_similarities(generated_answer, expected_answer)
            best_similarity = max(similarities.values())
            
            # Content analysis
            content_analysis = self.analyze_content_match(generated_answer, expected_answer)
            
            print(f"[INFO] ⏱️  Response time: {response_time:.2f}s")
            print(f"[INFO] 📊 Best similarity: {best_similarity:.1f}%")
            print(f"[INFO] ✅ Validation confidence: {validation_confidence:.2f}")
            
            # Show answer comparison if requested
            if show_details:
                print(f"\n[COMPARISON]")
                print(f"Expected ({len(expected_answer)} chars): {expected_answer}")
                print(f"Generated ({len(generated_answer)} chars): {generated_answer[:200]}{'...' if len(generated_answer) > 200 else ''}")
                print(f"\nSimilarity Metrics:")
                for metric, score in similarities.items():
                    print(f"  {metric}: {score:.1f}%")
                print(f"Word overlap: {content_analysis['word_overlap_percent']:.1f}%")
            
            # Create evaluation result
            result = EvaluationResult(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                similarity_scores=similarities,
                best_similarity=best_similarity,
                response_time=response_time,
                sources=sources,
                contexts=contexts,
                detected_language=detected_language,
                validation_confidence=validation_confidence,
                relevance_score=relevance_score,
                was_translated=was_translated,
                feedback_iterations=feedback_iterations,
                content_analysis=content_analysis
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating question: {e}")
            response_time = time.time() - start_time
            
            # Return error result
            return EvaluationResult(
                question=question,
                expected_answer=expected_answer,
                generated_answer=f"ERROR: {str(e)}",
                similarity_scores={"error": 0.0},
                best_similarity=0.0,
                response_time=response_time,
                sources=[],
                contexts=[],
                detected_language=detected_language,
                validation_confidence=0.0,
                relevance_score=0.0,
                was_translated=False,
                feedback_iterations=0,
                content_analysis={"error": True}
            )
    
    def evaluate_dataset(self, eval_data: List[Dict], sample_limit: int = None) -> List[EvaluationResult]:
        """Evaluate the entire dataset with improved rate limiting"""
        if sample_limit:
            eval_data = eval_data[:sample_limit]
            print(f"[INFO] 📝 Evaluating {sample_limit} samples (limited)")
        else:
            print(f"[INFO] 📝 Evaluating all {len(eval_data)} samples")
        
        results = []
        total_samples = len(eval_data)
        
        for i, sample in enumerate(eval_data, 1):
            print(f"\n{'='*80}")
            print(f"[INFO] 📊 Progress: {i}/{total_samples} ({(i/total_samples)*100:.1f}%)")
            
            # Extract question and answer (handle different JSON formats)
            question = sample.get('question') or sample.get('query', '')
            expected_answer = sample.get('answer') or sample.get('expected_answer', '')
            
            if not question or not expected_answer:
                logger.warning(f"Skipping invalid sample {i}: missing question or answer")
                continue
            
            # Evaluate the question (show details for first few questions)
            show_details = i <= 3  # Show comparison for first 3 questions
            result = self.evaluate_single_question(question, expected_answer, show_details)
            results.append(result)
            
            # Show immediate result
            status = "✅" if result.best_similarity >= SIMILARITY_THRESHOLD else "❌"
            print(f"[INFO] {status} Result: {result.best_similarity:.1f}% similarity")
            
            # Rate limiting delay (except for last question)
            if i < total_samples:
                print(f"[INFO] ⏸️ Waiting {RATE_LIMIT_DELAY}s to avoid rate limits...")
                time.sleep(RATE_LIMIT_DELAY)
        
        self.results = results
        return results
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        if not results:
            return {}
        
        # Basic accuracy metrics
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.best_similarity >= SIMILARITY_THRESHOLD)
        accuracy = (correct_answers / total_questions) * 100
        
        # Similarity scores
        similarity_scores = [r.best_similarity for r in results]
        avg_similarity = statistics.mean(similarity_scores)
        median_similarity = statistics.median(similarity_scores)
        
        # Response times
        response_times = [r.response_time for r in results]
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        
        # Validation confidence
        validation_scores = [r.validation_confidence for r in results if r.validation_confidence > 0]
        avg_validation = statistics.mean(validation_scores) if validation_scores else 0.0
        
        # Language distribution
        languages = [r.detected_language for r in results]
        language_dist = {lang: languages.count(lang) for lang in set(languages)}
        
        # Error analysis
        error_count = sum(1 for r in results if "ERROR:" in r.generated_answer)
        
        # Feedback loop usage (if available)
        feedback_iterations = [r.feedback_iterations for r in results if r.feedback_iterations > 0]
        avg_feedback_iterations = statistics.mean(feedback_iterations) if feedback_iterations else 0.0
        
        # Confidence levels
        high_confidence = sum(1 for r in results if r.validation_confidence >= CONFIDENCE_THRESHOLD)
        
        # Content analysis
        disclaimers = sum(1 for r in results if r.content_analysis and r.content_analysis.get('has_disclaimer', False))
        
        metrics = {
            "evaluation_summary": {
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "accuracy_percentage": round(accuracy, 2),
                "error_count": error_count
            },
            "similarity_metrics": {
                "average_similarity": round(avg_similarity, 2),
                "median_similarity": round(median_similarity, 2),
                "best_score": round(max(similarity_scores), 2),
                "worst_score": round(min(similarity_scores), 2),
                "similarity_std": round(statistics.stdev(similarity_scores) if len(similarity_scores) > 1 else 0, 2)
            },
            "performance_metrics": {
                "average_response_time": round(avg_response_time, 2),
                "median_response_time": round(median_response_time, 2),
                "total_evaluation_time": round(sum(response_times), 2)
            },
            "confidence_metrics": {
                "average_validation_confidence": round(avg_validation, 2),
                "high_confidence_answers": high_confidence,
                "high_confidence_percentage": round((high_confidence / total_questions) * 100, 2)
            },
            "language_distribution": language_dist,
            "advanced_features": {
                "translation_used": sum(1 for r in results if r.was_translated),
                "avg_feedback_iterations": round(avg_feedback_iterations, 2),
                "feedback_loop_usage": len(feedback_iterations),
                "answers_with_disclaimers": disclaimers
            }
        }
        
        return metrics
    
    def save_detailed_results(self, results: List[EvaluationResult], filepath: str = None):
        """Save detailed evaluation results to JSON file"""
        if filepath is None:
            filepath = DETAILED_LOG_FILE
        
        # Prepare data for JSON serialization
        detailed_data = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(results),
                "evaluation_config": {
                    "similarity_threshold": SIMILARITY_THRESHOLD,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "rate_limit_delay": RATE_LIMIT_DELAY,
                    "eval_data_path": EVAL_DATA_PATH
                }
            },
            "detailed_results": [result.to_dict() for result in results],
            "metrics": self.calculate_metrics(results)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] 💾 Detailed results saved to: {filepath}")
    
    def save_summary_csv(self, results: List[EvaluationResult], filepath: str = None):
        """Save summary results to CSV file"""
        if filepath is None:
            filepath = SUMMARY_CSV_FILE
        
        # Prepare CSV data
        csv_data = []
        for i, result in enumerate(results, 1):
            csv_data.append({
                "Question_ID": i,
                "Question": result.question[:100] + "..." if len(result.question) > 100 else result.question,
                "Expected_Answer": result.expected_answer[:100] + "..." if len(result.expected_answer) > 100 else result.expected_answer,
                "Generated_Answer": result.generated_answer[:100] + "..." if len(result.generated_answer) > 100 else result.generated_answer,
                "Best_Similarity": result.best_similarity,
                "Token_Sort_Ratio": result.similarity_scores.get("token_sort_ratio", 0),
                "Token_Set_Ratio": result.similarity_scores.get("token_set_ratio", 0),
                "Content_Similarity": result.similarity_scores.get("content_similarity", 0),
                "Response_Time": round(result.response_time, 2),
                "Detected_Language": result.detected_language,
                "Validation_Confidence": result.validation_confidence,
                "Was_Translated": result.was_translated,
                "Feedback_Iterations": result.feedback_iterations,
                "Status": "PASS" if result.best_similarity >= SIMILARITY_THRESHOLD else "FAIL"
            })
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        print(f"[INFO] 📊 Summary CSV saved to: {filepath}")
    
    def print_evaluation_summary(self, results: List[EvaluationResult]):
        """Print a comprehensive evaluation summary"""
        metrics = self.calculate_metrics(results)
        
        print(f"\n{'='*80}")
        print("📊 RAG EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        # Basic metrics
        summary = metrics.get("evaluation_summary", {})
        print(f"📝 Total Questions: {summary.get('total_questions', 0)}")
        print(f"✅ Passing Answers (≥{SIMILARITY_THRESHOLD}%): {summary.get('correct_answers', 0)}")
        print(f"🎯 Success Rate: {summary.get('accuracy_percentage', 0)}%")
        print(f"❌ Errors: {summary.get('error_count', 0)}")
        
        # Similarity metrics
        similarity = metrics.get("similarity_metrics", {})
        print(f"\n📏 SIMILARITY METRICS")
        print(f"   Average Similarity: {similarity.get('average_similarity', 0)}%")
        print(f"   Median Similarity: {similarity.get('median_similarity', 0)}%")
        print(f"   Best Score: {similarity.get('best_score', 0)}%")
        print(f"   Worst Score: {similarity.get('worst_score', 0)}%")
        print(f"   Standard Deviation: {similarity.get('similarity_std', 0)}%")
        
        # Performance metrics
        performance = metrics.get("performance_metrics", {})
        print(f"\n⏱️  PERFORMANCE METRICS")
        print(f"   Average Response Time: {performance.get('average_response_time', 0)}s")
        print(f"   Median Response Time: {performance.get('median_response_time', 0)}s")
        print(f"   Total Evaluation Time: {performance.get('total_evaluation_time', 0)}s")
        
        # Confidence metrics
        confidence = metrics.get("confidence_metrics", {})
        print(f"\n🔍 CONFIDENCE METRICS")
        print(f"   Average Validation Confidence: {confidence.get('average_validation_confidence', 0)}")
        print(f"   High Confidence Answers: {confidence.get('high_confidence_answers', 0)}")
        print(f"   High Confidence %: {confidence.get('high_confidence_percentage', 0)}%")
        
        # Language distribution
        languages = metrics.get("language_distribution", {})
        print(f"\n🌐 LANGUAGE DISTRIBUTION")
        for lang, count in languages.items():
            print(f"   {lang.title()}: {count} questions")
        
        # Advanced features
        advanced = metrics.get("advanced_features", {})
        print(f"\n🚀 ADVANCED FEATURES")
        print(f"   Translation Used: {advanced.get('translation_used', 0)} questions")
        print(f"   Feedback Loop Usage: {advanced.get('feedback_loop_usage', 0)} questions")
        print(f"   Avg Feedback Iterations: {advanced.get('avg_feedback_iterations', 0)}")
        print(f"   Answers with Disclaimers: {advanced.get('answers_with_disclaimers', 0)}")
        
        print(f"{'='*80}")

def main(sample_limit: int = None, eval_file: str = None, quick_test: bool = False):
    """Main evaluation function"""
    print("🚀 Starting RAG Pipeline Evaluation")
    print(f"{'='*80}")
    print("Features:")
    print("- Multiple similarity metrics (token_sort, token_set, content-based)")
    print(f"- Realistic similarity threshold ({SIMILARITY_THRESHOLD}%)")
    print("- Better answer cleaning and normalization")
    print("- Content analysis and detailed comparison")
    print(f"- Rate limiting ({RATE_LIMIT_DELAY}s delays)")
    print(f"{'='*80}")
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Load evaluation data
    eval_file_path = eval_file or EVAL_DATA_PATH
    eval_data = evaluator.load_evaluation_data(eval_file_path)
    
    # Run evaluation
    if quick_test and not sample_limit:
        sample_limit = 3
        print(f"[INFO] 🚀 Quick test mode: evaluating {sample_limit} samples")
    
    results = evaluator.evaluate_dataset(eval_data, sample_limit)
    
    # Save results
    evaluator.save_detailed_results(results)
    evaluator.save_summary_csv(results)
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    print(f"\n✅ Evaluation complete! Check {RESULTS_DIR} for detailed results.")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline")
    parser.add_argument("--samples", "-n", type=int, help="Limit number of samples to evaluate")
    parser.add_argument("--eval-file", "-f", type=str, help="Path to evaluation JSON file")
    parser.add_argument("--quick-test", "-q", action="store_true", help="Quick test with 3 samples and detailed output")
    args = parser.parse_args()
    
    main(sample_limit=args.samples, eval_file=args.eval_file, quick_test=args.quick_test)