#!/usr/bin/env python3
""" 
for larger LLM model only other wise small model will have timeout issues and empty logs 
evaluate.py  (CPU-only, Ragas ≥ 0.1.2)
---------------------------------------
Nightly RAG evaluation using Ragas + local Ollama.
"""
import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
# NEW: use the wrapper classes instead of LangchainLLM
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL  = "sentence-transformers/all-mpnet-base-v2"     # or "BAAI/bge-base-en-v1.5"
OLLAMA_MODEL     = "gemma3n:e2b"  # must be served locally
EVAL_FILE        = "dataqa/eval_set.json"

# ------------------------------------------------------------------
def load_eval_dataset(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        return Dataset.from_list(json.load(f))

def build_retriever():
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vs = FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": 5})

def run_eval():
    # ------------------------------------------------------------------
    # 1. Load data & add retrieved contexts
    # ------------------------------------------------------------------
    ds = load_eval_dataset(EVAL_FILE)
    retriever = build_retriever()

    def add_contexts(row):
        docs = retriever.invoke(row["question"])
        row["contexts"] = [d.page_content for d in docs]
        return row

    ds = ds.map(add_contexts)

    # ------------------------------------------------------------------
    # 2. Wrap local llama3.2:1b for Ragas
    # ------------------------------------------------------------------
    local_llm   = Ollama(model=OLLAMA_MODEL, temperature=0.0)
    ragas_llm   = LangchainLLMWrapper(local_llm)

    local_emb   = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                        model_kwargs={"device": "cpu"})
    ragas_emb   = LangchainEmbeddingsWrapper(local_emb)

    # ------------------------------------------------------------------
    # 3. Configure metrics
    # ------------------------------------------------------------------
    for metric in [context_precision, context_recall, faithfulness, answer_relevancy]:
        metric.llm         = ragas_llm
        metric.embeddings  = ragas_emb

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    result = evaluate(ds, metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ])

    # ------------------------------------------------------------------
    # 5. Save & print
    # ------------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(dict(result), f, indent=2)
    print("\n===== RAGAS Report =====")
    print(result)

if __name__ == "__main__":
    run_eval()