#!/usr/bin/env python3
"""
1b model cant run this either model is too small 
evaluate_stat.py  – CPU-only, no LLM judge
"""
import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision  # statistical only
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL  = "./ft_bge"
EVAL_FILE        = "data/eval_set.json"
LOG_FILE         = "logs/eval_stat.json"

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
    ds = load_eval_dataset(EVAL_FILE)
    retriever = build_retriever()

    def add_contexts(row):
        docs = retriever.invoke(row["question"])
        row["contexts"] = [d.page_content for d in docs]
        return row

    ds = ds.map(add_contexts)

    result = evaluate(ds, metrics=[context_precision, context_recall])
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(dict(result), f, indent=2)
    print("Statistical metrics:", dict(result))

if __name__ == "__main__":
    run_eval()