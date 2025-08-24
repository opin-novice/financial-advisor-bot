#!/usr/bin/env python3
"""
BERTScore Evaluation for RAG Pipeline
Measures embedding quality and retrieval performance
"""
import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from bert_score import score
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class BERTScoreEvaluator:
    """Evaluates RAG pipeline performance using BERTScore and related metrics"""
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """
        Initialize BERTScore evaluator
        
        Args:
            model_name: BERT model for evaluation
        """
        self.model_name = model_name
        self.bert_scorer = None
        self.cross_encoder = None
        self.sentence_transformer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize components
        self._init_bert_scorer()
        self._init_sentence_transformer()
        
    def _init_bert_scorer(self):
        """Initialize BERTScore scorer"""
        try:
            print(f"[INFO] Initializing BERTScore scorer with model: {self.model_name}")
            # BERTScore will be initialized on first use
            print("[INFO] [OK] BERTScore scorer ready")
        except Exception as e:
            print(f"[ERROR] Failed to initialize BERTScore scorer: {e}")
            self.bert_scorer = None
    
    def _init_sentence_transformer(self):
        """Initialize sentence transformer for embedding-based metrics"""
        try:
            print("[INFO] Loading sentence transformer for embedding metrics...")
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            print("[INFO] [OK] Sentence transformer loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    def evaluate_answer_quality(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        Evaluate answer quality using multiple metrics
        
        Args:
            predicted: Predicted answer from RAG system
            reference: Reference/gold standard answer
            
        Returns:
            Dictionary of evaluation scores
        """
        if not predicted or not reference:
            return {
                "bert_score_precision": 0.0,
                "bert_score_recall": 0.0,
                "bert_score_f1": 0.0,
                "cosine_similarity": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "bleu": 0.0
            }
        
        # BERTScore evaluation
        try:
            P, R, F1 = score([predicted], [reference], lang='en', verbose=False)
            bert_precision = P.mean().item()
            bert_recall = R.mean().item()
            bert_f1 = F1.mean().item()
        except Exception as e:
            print(f"[WARNING] BERTScore evaluation failed: {e}")
            bert_precision = bert_recall = bert_f1 = 0.0
        
        # Cosine similarity using sentence transformers
        try:
            if self.sentence_transformer:
                pred_embedding = self.sentence_transformer.encode([predicted])
                ref_embedding = self.sentence_transformer.encode([reference])
                cosine_sim = cosine_similarity(pred_embedding, ref_embedding)[0][0]
            else:
                cosine_sim = 0.0
        except Exception as e:
            print(f"[WARNING] Cosine similarity evaluation failed: {e}")
            cosine_sim = 0.0
        
        # ROUGE scores
        try:
            rouge_scores = self.rouge_scorer.score(reference, predicted)
            rouge1 = rouge_scores['rouge1'].fmeasure
            rouge2 = rouge_scores['rouge2'].fmeasure
            rougeL = rouge_scores['rougeL'].fmeasure
        except Exception as e:
            print(f"[WARNING] ROUGE evaluation failed: {e}")
            rouge1 = rouge2 = rougeL = 0.0
        
        # BLEU score
        try:
            pred_tokens = word_tokenize(predicted.lower())
            ref_tokens = [word_tokenize(reference.lower())]
            # Use smoothing function to avoid zero scores
            smoothie = SmoothingFunction().method1
            bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        except Exception as e:
            print(f"[WARNING] BLEU evaluation failed: {e}")
            bleu = 0.0
        
        return {
            "bert_score_precision": bert_precision,
            "bert_score_recall": bert_recall,
            "bert_score_f1": bert_f1,
            "cosine_similarity": float(cosine_sim),
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "bleu": bleu
        }
    
    def evaluate_retrieval_quality(self, retrieved_docs: List[str], query: str) -> Dict[str, float]:
        """
        Evaluate retrieval quality using embedding-based metrics
        
        Args:
            retrieved_docs: List of retrieved document contents
            query: User query
            
        Returns:
            Dictionary of retrieval quality scores
        """
        if not retrieved_docs or not query:
            return {
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "std_similarity": 0.0
            }
        
        try:
            if not self.sentence_transformer:
                return {
                    "avg_similarity": 0.0,
                    "max_similarity": 0.0,
                    "min_similarity": 0.0,
                    "std_similarity": 0.0
                }
            
            # Encode query and documents
            query_embedding = self.sentence_transformer.encode([query])
            doc_embeddings = self.sentence_transformer.encode(retrieved_docs)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            return {
                "avg_similarity": float(np.mean(similarities)),
                "max_similarity": float(np.max(similarities)),
                "min_similarity": float(np.min(similarities)),
                "std_similarity": float(np.std(similarities))
            }
            
        except Exception as e:
            print(f"[ERROR] Retrieval quality evaluation failed: {e}")
            return {
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "std_similarity": 0.0
            }
    
    def evaluate_end_to_end(self, queries: List[str], predictions: List[str], 
                          references: List[str], retrieved_docs: List[List[str]]) -> Dict[str, float]:
        """
        End-to-end evaluation combining answer quality and retrieval metrics
        
        Args:
            queries: List of user queries
            predictions: List of predicted answers
            references: List of reference answers
            retrieved_docs: List of lists of retrieved documents for each query
            
        Returns:
            Comprehensive evaluation results
        """
        if not queries or not predictions or not references or not retrieved_docs:
            return {"error": "Missing required evaluation data"}
        
        # Initialize accumulators
        total_bert_precision = 0.0
        total_bert_recall = 0.0
        total_bert_f1 = 0.0
        total_cosine_sim = 0.0
        total_rouge1 = 0.0
        total_rouge2 = 0.0
        total_rougeL = 0.0
        total_bleu = 0.0
        total_retrieval_avg = 0.0
        total_retrieval_max = 0.0
        
        valid_evaluations = 0
        
        # Evaluate each query-answer pair
        for i, (query, pred, ref, docs) in enumerate(zip(queries, predictions, references, retrieved_docs)):
            try:
                # Evaluate answer quality
                answer_scores = self.evaluate_answer_quality(pred, ref)
                
                # Evaluate retrieval quality
                retrieval_scores = self.evaluate_retrieval_quality(docs, query)
                
                # Accumulate scores
                total_bert_precision += answer_scores["bert_score_precision"]
                total_bert_recall += answer_scores["bert_score_recall"]
                total_bert_f1 += answer_scores["bert_score_f1"]
                total_cosine_sim += answer_scores["cosine_similarity"]
                total_rouge1 += answer_scores["rouge1"]
                total_rouge2 += answer_scores["rouge2"]
                total_rougeL += answer_scores["rougeL"]
                total_bleu += answer_scores["bleu"]
                total_retrieval_avg += retrieval_scores["avg_similarity"]
                total_retrieval_max += retrieval_scores["max_similarity"]
                
                valid_evaluations += 1
                
            except Exception as e:
                print(f"[WARNING] Failed to evaluate query {i}: {e}")
                continue
        
        if valid_evaluations == 0:
            return {"error": "No valid evaluations completed"}
        
        # Calculate averages
        avg_bert_precision = total_bert_precision / valid_evaluations
        avg_bert_recall = total_bert_recall / valid_evaluations
        avg_bert_f1 = total_bert_f1 / valid_evaluations
        avg_cosine_sim = total_cosine_sim / valid_evaluations
        avg_rouge1 = total_rouge1 / valid_evaluations
        avg_rouge2 = total_rouge2 / valid_evaluations
        avg_rougeL = total_rougeL / valid_evaluations
        avg_bleu = total_bleu / valid_evaluations
        avg_retrieval_avg = total_retrieval_avg / valid_evaluations
        avg_retrieval_max = total_retrieval_max / valid_evaluations
        
        return {
            "total_evaluations": len(queries),
            "valid_evaluations": valid_evaluations,
            "avg_bert_score_precision": avg_bert_precision,
            "avg_bert_score_recall": avg_bert_recall,
            "avg_bert_score_f1": avg_bert_f1,
            "avg_cosine_similarity": avg_cosine_sim,
            "avg_rouge1": avg_rouge1,
            "avg_rouge2": avg_rouge2,
            "avg_rougeL": avg_rougeL,
            "avg_bleu": avg_bleu,
            "avg_retrieval_avg_similarity": avg_retrieval_avg,
            "avg_retrieval_max_similarity": avg_retrieval_max,
            "overall_bert_score": (avg_bert_precision + avg_bert_recall + avg_bert_f1) / 3,
            "overall_retrieval_score": (avg_retrieval_avg + avg_retrieval_max) / 2,
            "overall_quality_score": (avg_bert_f1 + avg_cosine_sim + avg_rougeL) / 3
        }

def main():
    """Main function to demonstrate BERTScore evaluation"""
    print("=" * 80)
    print("BERTSCORE EVALUATION DEMO")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = BERTScoreEvaluator()
    
    # Example evaluation
    predicted_answer = "Bangladesh Bank is the central bank of Bangladesh. It regulates monetary policy and oversees the banking system."
    reference_answer = "Bangladesh Bank serves as the central bank of Bangladesh, responsible for regulating monetary policy and supervising the country's banking system."
    
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Reference Answer: {reference_answer}")
    print("-" * 80)
    
    # Evaluate answer quality
    scores = evaluator.evaluate_answer_quality(predicted_answer, reference_answer)
    
    print("BERTScore Evaluation Results:")
    print(f"  Precision: {scores['bert_score_precision']:.4f}")
    print(f"  Recall:    {scores['bert_score_recall']:.4f}")
    print(f"  F1 Score:  {scores['bert_score_f1']:.4f}")
    print(f"  Cosine:    {scores['cosine_similarity']:.4f}")
    print(f"  ROUGE-1:   {scores['rouge1']:.4f}")
    print(f"  ROUGE-2:   {scores['rouge2']:.4f}")
    print(f"  ROUGE-L:   {scores['rougeL']:.4f}")
    print(f"  BLEU:      {scores['bleu']:.4f}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()