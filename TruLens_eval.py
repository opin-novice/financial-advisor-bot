#!/usr/bin/env python3
"""
TruLens Evaluation for Advanced Multilingual RAG System

Evaluate the multilingual RAG pipeline with local (fast) metrics and optional TruLens LLM-judge metrics.

Usage:
    # Basic run on your QA file
    python TruLens_eval.py --qa dataqa/eval_set.json --limit 50

    # Save CSV
    python TruLens_eval.py --qa dataqa/eval_set.json --out reports/rag_eval.csv

    # Enable TruLens + OpenAI judge (optional)
    export OPENAI_API_KEY=sk-...
    python TruLens_eval.py --qa dataqa/eval_set.json --trulens

Notes:
- This script uses your existing RAG stack with multilingual support
- TruLens section is optional; local cross-encoder metrics run regardless
- Supports both English and Bangla evaluation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Your existing imports ---
from language_utils import LanguageDetector, BilingualResponseFormatter
from rag_utils import RAGUtils
from config import config
from advanced_rag_feedback import AdvancedRAGFeedbackLoop

# --- LangChain / Vector store / Models ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Cross-encoder for re-ranking/offline metrics
from sentence_transformers import CrossEncoder

# --- Optional TruLens ---
TRULENS_OK = False
TRULENS_ERR = None
try:
    from trulens_eval import Tru, Feedback, Select
    from trulens_eval.tru_custom_app import TruCustomApp, instrument
    from trulens_eval.feedback.provider.openai import OpenAI as TruOpenAI
    TRULENS_OK = True
except Exception as e:
    TRULENS_ERR = str(e)

# --- Configuration (using your existing config) ---
FAISS_INDEX_PATH = config.FAISS_INDEX_PATH
EMBEDDING_MODEL = config.EMBEDDING_MODEL
GROQ_MODEL = config.GROQ_MODEL
GROQ_API_KEY =os.getenv("GROQ_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Evaluation Result Class ---
@dataclass
class MultilingualRAGResult:
    question: str
    answer: str
    contexts: List[str]
    sources: List[Dict[str, Any]]
    detected_language: str
    confidence: float
    feedback_iterations: int = 0

class MultilingualRAGEvaluator:
    """
    Multilingual RAG system for evaluation using your existing components
    """
    
    def __init__(self):
        """Initialize the multilingual RAG system"""
        print("🔧 Initializing Multilingual RAG Evaluator...")
        
        # Initialize your existing components
        self.language_detector = LanguageDetector()
        self.formatter = BilingualResponseFormatter(self.language_detector)
        self.rag_utils = RAGUtils()
        
        # Initialize embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},  # Use CPU for stability
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Load vector store
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.3,
            max_tokens=500,
        )
        
        # Initialize advanced RAG feedback loop
        self.feedback_loop = AdvancedRAGFeedbackLoop(
            vectorstore=self.vectorstore,
            rag_utils=self.rag_utils,
            config=config.get_feedback_loop_config()
        )
        
        # Cross-encoder for evaluation metrics
        self.reranker = CrossEncoder(
            config.CROSS_ENCODER_MODEL, 
            device="cpu"  # Use CPU for stability
        )
        
        print("✅ Multilingual RAG Evaluator initialized successfully!")
    
    def ask(self, question: str) -> MultilingualRAGResult:
        """
        Process a question through the multilingual RAG system
        
        Args:
            question: User's question in English or Bangla
            
        Returns:
            MultilingualRAGResult with answer and metadata
        """
        # Detect language
        language, confidence = self.language_detector.detect_language(question)
        
        # Get language-specific prompt
        prompt_template = self.language_detector.get_language_specific_prompt(language)
        
        # Use advanced RAG feedback loop for better results
        try:
            result = self.feedback_loop.retrieve_with_feedback_loop(
                original_query=question,
                category="financial"  # Default category for financial queries
            )
            
            documents = result.get("documents", [])
            contexts = [doc.page_content for doc in documents]
            sources = [doc.metadata for doc in documents]
            iterations = result.get("total_iterations", 0)
            
            # Generate answer using retrieved documents
            if documents:
                context_text = "\n\n".join(contexts)
                prompt = prompt_template.format(context=context_text, input=question)
                
                try:
                    response = self.llm.invoke(prompt)
                    answer = response.content.strip()
                except Exception as llm_error:
                    print(f"⚠️ LLM generation failed: {llm_error}")
                    answer = "I apologize, but I cannot generate an answer at this time."
            else:
                answer = self.language_detector.translate_system_messages(
                    "I could not find relevant information in my database for your query.",
                    language
                )
            
        except Exception as e:
            print(f"⚠️ Feedback loop failed, using basic retrieval: {e}")
            # Fallback to basic retrieval
            docs = self.vectorstore.similarity_search(question, k=5)
            contexts = [doc.page_content for doc in docs]
            sources = [doc.metadata for doc in docs]
            
            # Generate answer using basic method
            context_text = "\n\n".join(contexts)
            prompt = prompt_template.format(context=context_text, input=question)
            
            try:
                response = self.llm.invoke(prompt)
                answer = response.content.strip()
            except Exception as llm_error:
                print(f"⚠️ LLM generation failed: {llm_error}")
                answer = "I apologize, but I cannot generate an answer at this time."
            
            iterations = 0
        
        return MultilingualRAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            sources=sources,
            detected_language=language,
            confidence=confidence,
            feedback_iterations=iterations
        )

# --- Local (fast) metrics ---
def sigmoid(x: float) -> float:
    """Sigmoid function for score normalization"""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def ce_max_similarity(reranker: CrossEncoder, answer: str, contexts: List[str]) -> float:
    """Calculate maximum similarity between answer and contexts using cross-encoder"""
    if not contexts or not answer.strip():
        return 0.0
    
    pairs = [[answer, ctx] for ctx in contexts if ctx.strip()]
    if not pairs:
        return 0.0
        
    try:
        scores = reranker.predict(pairs)
        if hasattr(scores, '__iter__'):
            valid_scores = [sigmoid(float(s)) for s in scores if not (math.isnan(float(s)) or math.isinf(float(s)))]
            return float(max(valid_scores)) if valid_scores else 0.0
        else:
            score = float(scores)
            return sigmoid(score) if not (math.isnan(score) or math.isinf(score)) else 0.0
    except Exception as e:
        print(f"⚠️ Cross-encoder similarity failed: {e}")
        return 0.0

def ce_question_context_similarity(reranker: CrossEncoder, question: str, contexts: List[str]) -> float:
    """Calculate maximum similarity between question and contexts using cross-encoder"""
    if not contexts or not question.strip():
        return 0.0
    
    pairs = [[question, ctx] for ctx in contexts if ctx.strip()]
    if not pairs:
        return 0.0
        
    try:
        scores = reranker.predict(pairs)
        if hasattr(scores, '__iter__'):
            valid_scores = [sigmoid(float(s)) for s in scores if not (math.isnan(float(s)) or math.isinf(float(s)))]
            return float(max(valid_scores)) if valid_scores else 0.0
        else:
            score = float(scores)
            return sigmoid(score) if not (math.isnan(score) or math.isinf(score)) else 0.0
    except Exception as e:
        print(f"⚠️ Cross-encoder similarity failed: {e}")
        return 0.0

def calculate_language_consistency(detected_lang: str, answer: str, detector: LanguageDetector) -> float:
    """Calculate consistency between detected query language and answer language"""
    try:
        answer_lang, answer_conf = detector.detect_language(answer)
        if detected_lang == answer_lang:
            return 1.0
        else:
            return 0.5  # Partial credit for mixed language scenarios
    except Exception:
        return 0.5

# --- Optional TruLens instrumentation ---
if TRULENS_OK:
    @instrument
    def trulens_query(app: MultilingualRAGEvaluator, question: str) -> Dict[str, Any]:
        """Instrumented function for TruLens recording"""
        res = app.ask(question)
        return {
            "answer": res.answer,
            "contexts": res.contexts,
            "language": res.detected_language,
            "confidence": res.confidence
        }

# --- Load QA pairs ---
def load_qa_pairs(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Load QA pairs from JSON file"""
    if not os.path.exists(path):
        print(f"❌ QA file not found: {path}")
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                # JSONL format
                rows = []
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
                        if limit and len(rows) >= limit:
                            break
            else:
                # JSON format
                data = json.load(f)
                if isinstance(data, list):
                    rows = data[:limit] if limit else data
                else:
                    rows = [data]
        
        print(f"✅ Loaded {len(rows)} QA pairs from {path}")
        return rows
        
    except Exception as e:
        print(f"❌ Error loading QA pairs: {e}")
        return []

# --- Main evaluation function ---
def main():
    parser = argparse.ArgumentParser(description="TruLens Evaluation for Multilingual RAG")
    parser.add_argument("--qa", default="dataqa/eval_set.json", 
                       help="JSON/JSONL file with QA pairs")
    parser.add_argument("--limit", type=int, default=10, 
                       help="Number of samples to evaluate (0=all)")
    parser.add_argument("--out", type=str, default=None, 
                       help="Optional CSV path to save results")
    parser.add_argument("--trulens", action="store_true", 
                       help="Enable TruLens LLM-judge metrics (requires OPENAI_API_KEY)")
    args = parser.parse_args()
    
    print("🚀 Starting Multilingual RAG Evaluation")
    print("=" * 60)
    
    # Load QA pairs
    qa_pairs = load_qa_pairs(args.qa, limit=args.limit if args.limit > 0 else None)
    if not qa_pairs:
        print("❌ No QA pairs found. Exiting.")
        return
    
    # Initialize evaluator
    try:
        evaluator = MultilingualRAGEvaluator()
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return
    
    # Optional TruLens setup
    tru_app = None
    if args.trulens:
        if not TRULENS_OK:
            print(f"⚠️ TruLens not available: {TRULENS_ERR}")
        else:
            try:
                tru = Tru()
                provider = TruOpenAI()  # requires OPENAI_API_KEY
                
                f_ans_rel = Feedback(provider.relevance, name="Answer Relevance") \
                    .on_input() \
                    .on(Select.Record.app.output["answer"])
                f_ctx_rel = Feedback(provider.context_relevance, name="Context Relevance") \
                    .on_input() \
                    .on(Select.Record.app.output["contexts"])
                f_grounded = Feedback(provider.groundedness_with_cot_reasons, name="Groundedness") \
                    .on(Select.Record.app.output["contexts"]) \
                    .on(Select.Record.app.output["answer"])
                
                tru_app = TruCustomApp(
                    app=evaluator,
                    app_id="Multilingual-RAG-Eval",
                    record_app=trulens_query,
                    feedbacks=[f_ans_rel, f_ctx_rel, f_grounded],
                )
                print("✅ TruLens enabled. Records will be stored locally.")
                
            except Exception as e:
                print(f"⚠️ Could not initialize TruLens: {e}")
    
    # Run evaluation
    print(f"\n📊 Evaluating {len(qa_pairs)} samples...")
    results = []
    
    for i, qa_pair in enumerate(tqdm(qa_pairs, desc="Evaluating")):
        # Extract question from various possible field names
        question = qa_pair.get("query") or qa_pair.get("question") or qa_pair.get("input") or ""
        if not question.strip():
            continue
        
        try:
            # Get RAG result
            result = evaluator.ask(question)
            
            # Calculate local metrics
            ctx_relevance = ce_question_context_similarity(
                evaluator.reranker, question, result.contexts
            )
            ans_groundedness = ce_max_similarity(
                evaluator.reranker, result.answer, result.contexts
            )
            lang_consistency = calculate_language_consistency(
                result.detected_language, result.answer, evaluator.language_detector
            )
            
            # Store results
            results.append({
                "sample_id": i + 1,
                "question": question,
                "answer": result.answer,
                "detected_language": result.detected_language,
                "language_confidence": round(result.confidence, 3),
                "feedback_iterations": result.feedback_iterations,
                "ctx_relevance_ce": round(ctx_relevance, 3),
                "ans_groundedness_ce": round(ans_groundedness, 3),
                "language_consistency": round(lang_consistency, 3),
                "n_contexts": len(result.contexts),
                "first_source": result.sources[0] if result.sources else {}
            })
            
            # TruLens recording
            if tru_app is not None:
                try:
                    tru_app.record(question)
                except Exception as e:
                    print(f"⚠️ TruLens record failed: {e}")
                    
        except Exception as e:
            print(f"⚠️ Error processing sample {i+1}: {e}")
            continue
    
    # Create results DataFrame
    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        
        # Show sample results
        print("\n🔍 Sample Results:")
        print(df[["sample_id", "question", "detected_language", "ctx_relevance_ce", 
                 "ans_groundedness_ce", "language_consistency"]].head(10))
        
        # Show averages
        print("\n📈 Average Metrics:")
        numeric_cols = ["ctx_relevance_ce", "ans_groundedness_ce", "language_consistency", 
                       "language_confidence", "feedback_iterations"]
        averages = df[numeric_cols].mean()
        for col, avg in averages.items():
            print(f"  {col}: {avg:.3f}")
        
        # Language distribution
        print("\n🌍 Language Distribution:")
        lang_dist = df["detected_language"].value_counts()
        for lang, count in lang_dist.items():
            print(f"  {lang}: {count} samples ({count/len(df)*100:.1f}%)")
        
        # Save to CSV if requested
        if args.out:
            os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
            df.to_csv(args.out, index=False, encoding='utf-8')
            print(f"\n💾 Results saved to: {args.out}")
    
    else:
        print("❌ No results generated.")
    
    # TruLens dashboard info
    if args.trulens and TRULENS_OK:
        print("\n" + "=" * 60)
        print("📊 TRULENS DASHBOARD")
        print("=" * 60)
        print("To view the TruLens dashboard, run:")
        print("    python -m trulens_eval.dashboard --port 8501")
        print("Then open: http://localhost:8501/")
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()
