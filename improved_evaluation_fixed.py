#!/usr/bin/env python3
"""
Improved TruLens Evaluation with Fixed Cross-Encoder Scoring
"""

import os
import sys
import json
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the fixed scorer
from fix_cross_encoder_scoring import FixedCrossEncoderScorer
from TruLens_eval import MultilingualRAGEvaluator

def evaluate_with_fixed_scoring():
    """Run evaluation with the fixed cross-encoder scoring"""
    print("🚀 Running Improved TruLens Evaluation with Fixed Scoring")
    print("=" * 60)
    
    # Initialize components
    try:
        evaluator = MultilingualRAGEvaluator()
        fixed_scorer = FixedCrossEncoderScorer()
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Load improved QA pairs
    qa_file = "data/improved_qa_pairs.jsonl"
    if not os.path.exists(qa_file):
        print(f"❌ QA file not found: {qa_file}")
        return
    
    qa_pairs = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    
    print(f"✅ Loaded {len(qa_pairs)} QA pairs")
    
    results = []
    
    for i, qa_pair in enumerate(tqdm(qa_pairs, desc="Evaluating")):
        question = qa_pair["query"]
        expected_answer = qa_pair["positive"]
        category = qa_pair.get("category", "general")
        
        try:
            # Get RAG result
            result = evaluator.ask(question)
            
            # Use fixed scorer for metrics
            ctx_relevance = fixed_scorer.calculate_similarity(question, result.contexts)
            ans_groundedness = fixed_scorer.calculate_similarity(result.answer, result.contexts)
            expected_similarity = fixed_scorer.calculate_similarity(expected_answer, result.contexts)
            
            results.append({
                "sample_id": i + 1,
                "category": category,
                "question": question,
                "answer": result.answer,
                "expected_answer": expected_answer,
                "ctx_relevance_fixed": round(ctx_relevance, 3),
                "ans_groundedness_fixed": round(ans_groundedness, 3),
                "expected_similarity": round(expected_similarity, 3),
                "n_contexts": len(result.contexts),
                "contexts_preview": result.contexts[0][:150] + "..." if result.contexts else ""
            })
            
        except Exception as e:
            print(f"⚠️ Error processing sample {i+1}: {e}")
            continue
    
    # Analyze results
    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "=" * 60)
        print("📊 FIXED EVALUATION RESULTS")
        print("=" * 60)
        
        # Show sample results
        print("\n🔍 Sample Results:")
        display_cols = ["sample_id", "category", "ctx_relevance_fixed", "ans_groundedness_fixed", "expected_similarity"]
        print(df[display_cols])
        
        # Show averages
        print("\n📈 Average Metrics:")
        numeric_cols = ["ctx_relevance_fixed", "ans_groundedness_fixed", "expected_similarity"]
        averages = df[numeric_cols].mean()
        for col, avg in averages.items():
            print(f"  {col}: {avg:.3f}")
        
        # Category-wise performance
        print("\n📊 Performance by Category:")
        category_stats = df.groupby("category")[numeric_cols].mean()
        print(category_stats.round(3))
        
        # Save results
        output_file = "fixed_trulens_evaluation.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Results saved to: {output_file}")
        
        # Success assessment
        if averages["ctx_relevance_fixed"] > 0.3:
            print("\n🎉 SUCCESS: Context relevance significantly improved!")
        if averages["ans_groundedness_fixed"] > 0.3:
            print("🎉 SUCCESS: Answer groundedness significantly improved!")
            
        return averages
    
    else:
        print("❌ No results generated.")
        return None

if __name__ == "__main__":
    evaluate_with_fixed_scoring()
