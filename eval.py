#!/usr/bin/env python3
""" 
RAGAS Evaluation with Groq API
-----------------------------
RAG evaluation using Ragas + Groq's llama3-8b-8192 model.
"""
import json
import os
# Silence HuggingFace tokenizers fork warnings in multiprocessing contexts
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
FAISS_INDEX_PATH = "faiss_index"
# IMPORTANT: Must match the model used to build the FAISS index
# Default uses a 768-dim MPNet family model; override with EVAL_EMBEDDING_MODEL if needed
EMBEDDING_MODEL  = os.getenv(
    "EVAL_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
GROQ_MODEL       = "llama3-8b-8192"
# GROQ API Key
GROQ_API_KEY =  os.getenv("GROQ_API_KEY")
EVAL_FILE        = "dataqa/eval_set.json"
LOG_FILE         = "logs/ragas_evaluation.json"

# Rate limiting settings - Balanced for Groq free tier
MAX_TOKENS       = 500  # Higher token limit for better RAGAS responses
# Allow overriding delay and number of samples via environment variables
DELAY_SECONDS    = int(os.getenv("EVAL_DELAY_SECONDS", "5"))   # Reduced delay for testing
NUM_SAMPLES      = int(os.getenv("EVAL_NUM_SAMPLES", "3"))      # Reduced to 3 samples for testing

# ------------------------------------------------------------------
# RATE LIMITED GROQ WRAPPER CLASS
# ------------------------------------------------------------------
class RateLimitedGroq:
    """Wrapper for ChatGroq with intelligent rate limiting and retry logic"""
    
    def __init__(self, model, api_key, temperature=0.0, max_tokens=500, delay=40):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay = delay
        self.llm = ChatGroq(
            model=model,
            groq_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={"top_p": 0.9}
        )
    
    def invoke(self, *args, **kwargs):
        """Invoke with rate limiting and retry logic"""
        max_retries = 3
        base_wait = 60  # Start with 60 second wait
        
        for attempt in range(max_retries):
            try:
                # Add delay before each request
                if attempt > 0:
                    print(f"⏳ Waiting {self.delay}s before retry {attempt}...")
                time.sleep(self.delay)
                
                result = self.llm.invoke(*args, **kwargs)
                print("✅ Request successful")
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "rate_limit" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = base_wait * (2 ** attempt)  # Exponential backoff
                        print(f"🔄 Rate limit hit! Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("❌ Max retries exceeded due to rate limits")
                        raise Exception("Rate limit exceeded after multiple retries") from e
                else:
                    print(f"❌ Non-rate-limit error: {e}")
                    raise e
        
        raise Exception("Max retries exceeded")
    
    def __getattr__(self, name):
        """Delegate other attributes to the underlying LLM"""
        return getattr(self.llm, name)

# ------------------------------------------------------------------
def load_eval_dataset(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

def build_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def run_eval():
    print("🚀 Starting RAGAS Evaluation with Groq")
    print(f"📊 Model: {GROQ_MODEL}")
    print(f"🔬 Testing with {NUM_SAMPLES} samples")
    print("⚠️  RATE LIMIT NOTICE: Using balanced settings for Groq free tier")
    print(f"    - {MAX_TOKENS} max tokens per request")
    print(f"    - {DELAY_SECONDS}s delay between requests")
    print(f"    - {NUM_SAMPLES} samples only")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1. Load data & prepare for RAGAS format
    # ------------------------------------------------------------------
    print("📖 Loading evaluation dataset...")
    ds = load_eval_dataset(EVAL_FILE)
    retriever = build_retriever()

    def prepare_for_ragas(row):
        # Retrieve contexts for the question
        docs = retriever.invoke(row["question"])
        retrieved_contexts = [d.page_content for d in docs]
        
        # Map to RAGAS expected format
        return {
            "user_input": row["question"],  # RAGAS expects 'user_input'
            "retrieved_contexts": retrieved_contexts,  # RAGAS expects 'retrieved_contexts'
            "response": row["answer"],  # RAGAS expects 'response'
            "reference": row["ground_truth"]  # RAGAS expects 'reference' for ground truth
        }

    print("🔍 Preparing data for RAGAS format...")
    ds = ds.map(prepare_for_ragas)
    
    # Limit to small number of samples to avoid rate limits
    test_ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
    print(f"✅ Prepared {len(test_ds)} samples for evaluation")
    
    # Debug: Print first sample to verify format
    print("\n🔍 Sample data format:")
    sample = test_ds[0]
    for key, value in sample.items():
        if key == "retrieved_contexts":
            print(f"   {key}: [{len(value)} contexts]")
        else:
            print(f"   {key}: {str(value)[:100]}..." if len(str(value)) > 100 else f"   {key}: {value}")

    # ------------------------------------------------------------------
    # 2. Initialize Simple Groq LLM for RAGAS
    # ------------------------------------------------------------------
    print(f"🤖 Initializing Groq LLM ({GROQ_MODEL})...")
    print(f"⚙️ Settings: {MAX_TOKENS} max tokens, temperature=0.0")
    
    # Use simple ChatGroq directly - let RAGAS handle rate limiting
    simple_llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        model_kwargs={"top_p": 0.9}
    )
    
    # Wrap for RAGAS
    ragas_llm = LangchainLLMWrapper(simple_llm)
    
    # Initialize embeddings for RAGAS
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
    )

    # ------------------------------------------------------------------
    # 3. Configure metrics with rate limiting
    # ------------------------------------------------------------------
    print("⚙️ Configuring RAGAS metrics...")
    
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    
    for metric in metrics:
        metric.llm = ragas_llm
        metric.embeddings = ragas_embeddings

    # ------------------------------------------------------------------
    # 4. Run evaluation with careful rate limiting
    # ------------------------------------------------------------------
    print(f"🧪 Starting evaluation...")
    print(f"⏱️ Using {DELAY_SECONDS}s delay between requests...")
    print(f"🔢 Processing {len(test_ds)} samples with {len(metrics)} metrics each")
    print(f"⏳ Estimated time: ~{(len(test_ds) * len(metrics) * DELAY_SECONDS) // 60} minutes")
    print("🚨 IMPORTANT: Do not interrupt - rate limits apply!")
    
    # Add a small delay before starting
    print("⏳ Starting in 3 seconds...")
    time.sleep(3)
    
    try:
        print("🚀 Beginning RAGAS evaluation...")
        result = evaluate(
            test_ds, 
            metrics=metrics,
        )
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Evaluation failed: {error_msg}")
        
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            print("💡 Rate limit suggestions:")
            print("   - Reduce NUM_SAMPLES (currently {})".format(NUM_SAMPLES))
            print("   - Increase DELAY_SECONDS (currently {})".format(DELAY_SECONDS))
            print("   - Wait a few minutes and try again")
        elif "timeout" in error_msg.lower():
            print("💡 Timeout suggestions:")
            print("   - Reduce MAX_TOKENS (currently {})".format(MAX_TOKENS))
            print("   - Check your internet connection")
        else:
            print("💡 General suggestions:")
            print("   - Check your Groq API key")
            print("   - Verify your account has API access")
            print(f"   - Full error: {error_msg}")
        return

    # ------------------------------------------------------------------
    # 5. Save & display results
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("📊 RAGAS EVALUATION RESULTS")
    print("=" * 50)
    
    # Print results (handle EvaluationResult object)
    print("📈 SCORES:")
    if hasattr(result, 'to_pandas'):
        # Convert to pandas DataFrame first
        df = result.to_pandas()
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Only process numeric metric columns
        metric_columns = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        
        for column in df.columns:
            if column in metric_columns:
                try:
                    # Check if column exists and has numeric data
                    if not df[column].empty:
                        score = df[column].mean()
                        print(f"   {column}: {score:.4f}")
                    else:
                        print(f"   {column}: No data")
                except Exception as col_error:
                    print(f"   {column}: Error calculating mean - {col_error}")
        
        # Also show individual row values for debugging
        print("\n🔍 Individual sample scores:")
        for idx, row in df.iterrows():
            print(f"   Sample {idx + 1}:")
            for col in metric_columns:
                if col in df.columns:
                    try:
                        value = row[col]
                        if pd.isna(value):
                            print(f"     {col}: NaN")
                        else:
                            print(f"     {col}: {float(value):.4f}")
                    except:
                        print(f"     {col}: {row[col]}")
    else:
        # Fallback: try to access as dict or print raw
        try:
            for key in ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']:
                if hasattr(result, key):
                    score = getattr(result, key)
                    print(f"   {key}: {score:.4f}")
        except:
            print(f"   Raw result: {result}")
    
    # Save to file
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        try:
            if hasattr(result, 'to_pandas'):
                # Save as JSON from DataFrame
                df = result.to_pandas()
                results_dict = {}
                
                # Only save numeric metric columns
                metric_columns = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
                
                for column in metric_columns:
                    if column in df.columns:
                        try:
                            if not df[column].empty and not df[column].isna().all():
                                results_dict[column] = float(df[column].mean())
                            else:
                                results_dict[column] = None
                        except Exception as col_err:
                            results_dict[column] = f"Error: {col_err}"
                
                # Add metadata
                results_dict["metadata"] = {
                    "samples_evaluated": len(df),
                    "model": GROQ_MODEL,
                    "max_tokens": MAX_TOKENS,
                    "delay_seconds": DELAY_SECONDS
                }
                
                json.dump(results_dict, f, indent=2)
            else:
                # Fallback: save string representation
                json.dump({"result": str(result)}, f, indent=2)
        except Exception as save_error:
            # Ultimate fallback
            json.dump({"error": f"Could not serialize results: {save_error}", "raw_result": str(result)}, f, indent=2)
    
    print(f"\n💾 Results saved to: {LOG_FILE}")
    print("🎉 Evaluation complete!")

if __name__ == "__main__":
    run_eval()