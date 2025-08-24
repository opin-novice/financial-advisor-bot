#!/usr/bin/env python3
"""
Comprehensive Evaluation for RAG Pipeline
Combines BERTScore, ROUGE, BLEU, and embedding-based metrics
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
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class ComprehensiveEvaluator:
    """Comprehensive evaluation combining multiple metrics"""
    
    def __init__(self):
        """Initialize comprehensive evaluator"""
        print("[INFO] Initializing Comprehensive Evaluator...")
        
        # Initialize components
        self.bert_scorer = None
        self.cross_encoder = None
        self.sentence_transformer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self._init_components()
        print("[INFO] [OK] Comprehensive Evaluator initialized successfully")
    
    def _init_components(self):
        """Initialize evaluation components"""
        try:
            # Initialize sentence transformer for embedding-based metrics
            print("[INFO] Loading sentence transformer for embedding metrics...")
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            print("[INFO] [OK] Sentence transformer loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
        
        try:
            # Initialize cross-encoder for re-ranking evaluation
            print("[INFO] Loading cross-encoder for re-ranking evaluation...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("[INFO] [OK] Cross-encoder loaded successfully")
        except Exception as e:
            print(f"[WARNING] Failed to load cross-encoder: {e}")
            self.cross_encoder = None
    
    def evaluate_answer_comprehensively(self, predicted: str, reference: str, context: List[str] = None) -> Dict[str, float]:
        """
        Comprehensive answer evaluation using multiple metrics
        
        Args:
            predicted: Predicted answer from RAG system
            reference: Reference/gold standard answer
            context: Retrieved context documents (optional)
            
        Returns:
            Dictionary of comprehensive evaluation scores
        """
        if not predicted or not reference:
            return self._empty_scores()
        
        # Initialize results dictionary
        results = {}
        
        # BERTScore evaluation
        try:
            P, R, F1 = score([predicted], [reference], lang='en', verbose=False)
            results.update({
                "bert_score_precision": P.mean().item(),
                "bert_score_recall": R.mean().item(),
                "bert_score_f1": F1.mean().item()
            })
        except Exception as e:
            print(f"[WARNING] BERTScore evaluation failed: {e}")
            results.update({
                "bert_score_precision": 0.0,
                "bert_score_recall": 0.0,
                "bert_score_f1": 0.0
            })
        
        # Cosine similarity using sentence transformers
        try:
            if self.sentence_transformer:
                pred_embedding = self.sentence_transformer.encode([predicted])
                ref_embedding = self.sentence_transformer.encode([reference])
                cosine_sim = cosine_similarity(pred_embedding, ref_embedding)[0][0]
                results["cosine_similarity"] = float(cosine_sim)
            else:
                results["cosine_similarity"] = 0.0
        except Exception as e:
            print(f"[WARNING] Cosine similarity evaluation failed: {e}")
            results["cosine_similarity"] = 0.0
        
        # ROUGE scores
        try:
            rouge_scores = self.rouge_scorer.score(reference, predicted)
            results.update({
                "rouge1": rouge_scores['rouge1'].fmeasure,
                "rouge2": rouge_scores['rouge2'].fmeasure,
                "rougeL": rouge_scores['rougeL'].fmeasure
            })
        except Exception as e:
            print(f"[WARNING] ROUGE evaluation failed: {e}")
            results.update({
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            })
        
        # BLEU score
        try:
            pred_tokens = word_tokenize(predicted.lower())
            ref_tokens = [word_tokenize(reference.lower())]
            bleu = sentence_bleu(ref_tokens, pred_tokens)
            results["bleu"] = bleu
        except Exception as e:
            print(f"[WARNING] BLEU evaluation failed: {e}")
            results["bleu"] = 0.0
        
        # Context relevance (if context provided)
        if context and self.sentence_transformer:
            try:
                context_relevance = self._evaluate_context_relevance(predicted, context)
                results.update(context_relevance)
            except Exception as e:
                print(f"[WARNING] Context relevance evaluation failed: {e}")
                results["context_relevance"] = 0.0
        else:
            results["context_relevance"] = 0.0
        
        # Cross-encoder relevance (if available)
        if context and self.cross_encoder:
            try:
                cross_encoder_scores = self._evaluate_cross_encoder_relevance(predicted, context)
                results.update(cross_encoder_scores)
            except Exception as e:
                print(f"[WARNING] Cross-encoder relevance evaluation failed: {e}")
                results["cross_encoder_relevance"] = 0.0
        else:
            results["cross_encoder_relevance"] = 0.0
        
        # Calculate composite scores
        results["composite_score"] = self._calculate_composite_score(results)
        
        return results
    
    def _empty_scores(self) -> Dict[str, float]:
        """Return empty scores dictionary"""
        return {
            "bert_score_precision": 0.0,
            "bert_score_recall": 0.0,
            "bert_score_f1": 0.0,
            "cosine_similarity": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "context_relevance": 0.0,
            "cross_encoder_relevance": 0.0,
            "composite_score": 0.0
        }
    
    def _evaluate_context_relevance(self, answer: str, context: List[str]) -> Dict[str, float]:
        """
        Evaluate answer relevance to provided context
        
        Args:
            answer: Generated answer
            context: Retrieved context documents
            
        Returns:
            Context relevance scores
        """
        if not context or not self.sentence_transformer:
            return {"context_relevance": 0.0}
        
        try:
            # Encode answer and context
            answer_embedding = self.sentence_transformer.encode([answer])
            context_embeddings = self.sentence_transformer.encode(context)
            
            # Calculate similarities
            similarities = cosine_similarity(answer_embedding, context_embeddings)[0]
            
            # Return average similarity
            return {"context_relevance": float(np.mean(similarities))}
            
        except Exception as e:
            print(f"[WARNING] Context relevance calculation failed: {e}")
            return {"context_relevance": 0.0}
    
    def _evaluate_cross_encoder_relevance(self, answer: str, context: List[str]) -> Dict[str, float]:
        """
        Evaluate answer relevance using cross-encoder
        
        Args:
            answer: Generated answer
            context: Retrieved context documents
            
        Returns:
            Cross-encoder relevance scores
        """
        if not context or not self.cross_encoder:
            return {"cross_encoder_relevance": 0.0}
        
        try:
            # Create query-answer pairs
            pairs = [[answer, doc] for doc in context[:5]]  # Limit to top 5 docs
            
            # Get relevance scores
            scores = self.cross_encoder.predict(pairs)
            
            # Return average score
            return {"cross_encoder_relevance": float(np.mean(scores))}
            
        except Exception as e:
            print(f"[WARNING] Cross-encoder relevance calculation failed: {e}")
            return {"cross_encoder_relevance": 0.0}
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate composite evaluation score
        
        Args:
            scores: Dictionary of individual scores
            
        Returns:
            Composite score (0-1)
        """
        # Weighted combination of key metrics
        weights = {
            "bert_score_f1": 0.3,
            "cosine_similarity": 0.2,
            "rougeL": 0.2,
            "context_relevance": 0.15,
            "cross_encoder_relevance": 0.15
        }
        
        composite = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in scores:
                composite += scores[metric] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite = composite / total_weight
        
        return composite
    
    def evaluate_batch(self, predictions: List[str], references: List[str], 
                      contexts: List[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate batch of predictions
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            contexts: List of retrieved contexts (optional)
            
        Returns:
            Average scores across all predictions
        """
        if not predictions or not references or len(predictions) != len(references):
            return self._empty_scores()
        
        if contexts and len(contexts) != len(predictions):
            contexts = None  # Disable context evaluation if mismatched
        
        # Initialize accumulators
        total_scores = {}
        count = len(predictions)
        
        # Evaluate each prediction
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            context = contexts[i] if contexts else None
            scores = self.evaluate_answer_comprehensively(pred, ref, context)
            
            # Accumulate scores
            for key, value in scores.items():
                if key not in total_scores:
                    total_scores[key] = 0.0
                total_scores[key] += value
        
        # Calculate averages
        avg_scores = {}
        for key, total in total_scores.items():
            avg_scores[key] = total / count if count > 0 else 0.0
        
        return avg_scores

def main():
    """Main function to demonstrate comprehensive evaluation"""
    print("=" * 80)
    print("COMPREHENSIVE RAG EVALUATION")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Example evaluation
    predicted_answers = [
        "Bangladesh Bank is the central bank of Bangladesh. It regulates monetary policy and oversees the banking system.",
        "To open a savings account in Bangladesh, you need to visit a bank branch with your National ID and some initial deposit.",
        "The interest rate for home loans in Bangladesh typically ranges from 8% to 12% depending on the bank and loan amount."
    ]
    
    reference_answers = [
        "Bangladesh Bank serves as the central bank of Bangladesh, responsible for regulating monetary policy and supervising the country's banking system.",
        "To open a savings account in Bangladesh, you must go to a bank branch with your National ID card and make an initial deposit.",
        "Home loan interest rates in Bangladesh generally vary between 8% and 12%, depending on the financial institution and loan size."
    ]
    
    context_documents = [
        ["Bangladesh Bank is the central bank that regulates monetary policy and supervises banking operations in Bangladesh."],
        ["To open a savings account, customers must provide National ID and make an initial deposit at any bank branch in Bangladesh."],
        ["Home loan interest rates in Bangladesh typically range from 8% to 12%, varying by bank and loan amount."]
    ]
    
    print(f"Evaluating {len(predicted_answers)} answer pairs...")
    print("-" * 80)
    
    # Evaluate batch
    avg_scores = evaluator.evaluate_batch(predicted_answers, reference_answers, context_documents)
    
    print("Comprehensive Evaluation Results:")
    print(f"  BERTScore Precision: {avg_scores['bert_score_precision']:.4f}")
    print(f"  BERTScore Recall:    {avg_scores['bert_score_recall']:.4f}")
    print(f"  BERTScore F1:        {avg_scores['bert_score_f1']:.4f}")
    print(f"  Cosine Similarity:   {avg_scores['cosine_similarity']:.4f}")
    print(f"  ROUGE-1:             {avg_scores['rouge1']:.4f}")
    print(f"  ROUGE-2:             {avg_scores['rouge2']:.4f}")
    print(f"  ROUGE-L:             {avg_scores['rougeL']:.4f}")
    print(f"  BLEU:                {avg_scores['bleu']:.4f}")
    print(f"  Context Relevance:   {avg_scores['context_relevance']:.4f}")
    print(f"  Cross-Encoder Rel:   {avg_scores['cross_encoder_relevance']:.4f}")
    print(f"  Composite Score:     {avg_scores['composite_score']:.4f}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()