#!/usr/bin/env python3
import os, json, time
from datasets import Dataset
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------- Configuration --------------------
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL  = "./ft_bge"
OLLAMA_MODEL     = "llama3.2:1b"
EVAL_FILE        = "data/eval_set.json"
LOG_FILE         = "logs/eval_ollama.json"
USE_LIMIT        = 10  # limit eval entries

# -------------------- Load Dataset --------------------
def load_eval_dataset(path, limit=USE_LIMIT):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data[:limit])

# -------------------- Main Eval Logic --------------------
def main():
    ds = load_eval_dataset(EVAL_FILE)

    # 1. Embeddings and Retriever
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    vs = FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    def add_contexts(row):
        docs = retriever.invoke(row["question"])
        row["contexts"] = [d.page_content for d in docs]
        return row

    ds = ds.map(add_contexts)

    # 2. LLM + Wrappers
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url="http://localhost:11434",
        temperature=0.0,
        timeout=300
    )
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(emb)

    for m in [context_precision, context_recall, faithfulness, answer_relevancy]:
        m.llm = ragas_llm
        m.embeddings = ragas_emb

    # 3. Evaluate Sequentially (no ray)
    print(f"[⏳] Evaluating {len(ds)} entries sequentially...")
    results = {
        "context_precision": [],
        "context_recall": [],
        "faithfulness": [],
        "answer_relevancy": [],
    }

    for i, example in enumerate(ds):
        print(f" - Example {i + 1}/{len(ds)}")
        try:
            results["context_precision"].append(context_precision.score(example))
            results["context_recall"].append(context_recall.score(example))
            results["faithfulness"].append(faithfulness.score(example))
            results["answer_relevancy"].append(answer_relevancy.score(example))
        except Exception as e:
            print(f"[❌] Error in example {i + 1}: {e}")
            for k in results.keys():
                results[k].append(None)

    # 4. Calculate Averages
    summary = {}
    for metric, scores in results.items():
        filtered = [s for s in scores if s is not None]
        summary[metric] = sum(filtered) / len(filtered) if filtered else 0.0

    # 5. Save Output
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "per_example": results,
            "average": summary
        }, f, indent=2)

    print("[✅] Evaluation done. Results saved to:", LOG_FILE)
    print(json.dumps(summary, indent=2))

# -------------------- Run --------------------
if __name__ == "__main__":
    main()
