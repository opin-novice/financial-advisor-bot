# TruLens_eval.py
"""
Evaluate the FinAuxi RAG pipeline with local (fast) metrics and optional TruLens LLM-judge metrics.

Usage:
    # Basic run on your QA file (uses data/qa_pairs.jsonl)
    python TruLens_eval.py --qa data/qa_pairs.jsonl --limit 50

    # Save CSV
    python TruLens_eval.py --qa data/qa_pairs.jsonl --out reports/rag_eval.csv

    # Enable TruLens + OpenAI judge (optional)
    set OPENAI_API_KEY=sk-...   # PowerShell: $env:OPENAI_API_KEY="sk-..."
    python TruLens_eval.py --qa data/qa_pairs.jsonl --trulens

Notes:
- This script reconstructs the RAG stack (index/retriever/LLM/prompt) similarly to your main.py,
  but runs synchronously for offline evaluation.
- TruLens section is optional; local cross-encoder metrics run regardless.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

# --- LangChain / Vector store / Models --------------------------------------
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Cross-encoder for re-ranking/offline metrics
from sentence_transformers import CrossEncoder

# --- Optional TruLens --------------------------------------------------------
TRULENS_OK = False
TRULENS_ERR = None
try:
    from trulens_eval import Tru, Feedback, Select
    from trulens_eval.tru_custom_app import TruCustomApp, instrument
    from trulens_eval.feedback.provider.openai import OpenAI as TruOpenAI
    TRULENS_OK = True
except Exception as e:
    TRULENS_ERR = str(e)

# ---------------- Config: mirror main.py ------------------------------------
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "./ft_bge"           # must match the index
OLLAMA_MODEL    = "llama3.2:1b"        # consider a larger instruct model for better answers
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Prompts: mirror main.py -----------------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a financial advisor specializing in Bangladeshi tax law and regulations. Use the following context to answer the question accurately and professionally.

IMPORTANT GUIDELINES:
- Only use information from the provided context
- If the context doesn't contain the answer, say "I cannot find specific information about this in the provided documents"
- Be precise with numbers, percentages, and dates
- Cite specific rules, sections, or forms when mentioned
- Keep answers concise but complete
- Use professional language appropriate for financial advice

Context:
{context}

Question: {question}

Answer:"""
)

REFORMAT_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="Rephrase the following question in one concise sentence:\n\n{question}"
)

HYDE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="Provide a short hypothetical answer (2-3 sentences) to the question:\n\n{question}"
)

# -------------- Utility: fast token-ish truncation --------------------------
def truncate_documents(docs: List[Document], max_chars: int = 12_000) -> List[Document]:
    out, total = [], 0
    for d in docs:
        c = len(d.page_content)
        if total + c > max_chars:
            break
        out.append(d)
        total += c
    return out

# -------------- Build RAG components (same spirit as main.py) ---------------
def build_rag_stack() -> Dict[str, Any]:
    # embeddings (normalize for BGE)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

    # vector store (allow local pickle — ensure you trust your own index)
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # retrievers
    dense_ret = vectorstore.as_retriever(search_kwargs={"k": 5})
    # NOTE: building BM25 from FAISS results; for production prefer persisted chunks
    all_docs = vectorstore.similarity_search("", k=10000)
    bm25_ret = BM25Retriever.from_documents(all_docs)
    bm25_ret.k = 5

    ensemble_ret = EnsembleRetriever(
        retrievers=[dense_ret, bm25_ret],
        weights=[0.7, 0.3]
    )

    # LLM
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=0.5,
        top_p=0.8,
        top_k=35,
        max_tokens=1024,
        repeat_penalty=1.2
    )

    # Chains
    chain_qa   = LLMChain(llm=llm, prompt=QA_PROMPT)
    chain_ref  = LLMChain(llm=llm, prompt=REFORMAT_PROMPT)
    chain_hyde = LLMChain(llm=llm, prompt=HYDE_PROMPT)
    combine    = StuffDocumentsChain(llm_chain=chain_qa, document_variable_name="context")

    # cross-encoder (also used for evaluation)
    reranker = CrossEncoder("BAAI/bge-reranker-large", device=DEVICE)

    return dict(
        vectorstore=vectorstore,
        dense_ret=dense_ret,
        bm25_ret=bm25_ret,
        ensemble_ret=ensemble_ret,
        llm=llm,
        chain_qa=chain_qa,
        chain_ref=chain_ref,
        chain_hyde=chain_hyde,
        combine=combine,
        reranker=reranker
    )

# -------------- Retrieval helpers (HyDE -> dense only) ----------------------
def expand_query(question: str, chain_hyde: LLMChain, chain_ref: LLMChain) -> List[Tuple[str, str]]:
    out = [("orig", question)]
    try:
        hyde = chain_hyde.run(question)
        out.append(("hyde", hyde))
    except Exception:
        pass
    try:
        ref = chain_ref.run(question)
        out.append(("reformat", ref))
    except Exception:
        pass
    return out

def multi_query_retrieval(
    question: str,
    ensemble_ret,
    dense_ret,
    chain_hyde: LLMChain,
    chain_ref: LLMChain,
    k: int = 25
) -> List[Document]:
    expanded = expand_query(question, chain_hyde, chain_ref)
    docs: List[Document] = []
    for kind, q in expanded:
        if kind == "hyde":
            docs.extend(dense_ret.get_relevant_documents(q))
        else:
            docs.extend(ensemble_ret.get_relevant_documents(q))
    # de-dupe by content
    seen, uniq = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            uniq.append(d)
    return uniq[:k]

def rerank_documents(query: str, docs: List[Document], reranker: CrossEncoder, top_n: int = 7) -> List[Document]:
    if not docs:
        return []
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    reranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return reranked[:top_n]

# -------------- Core RAG call -----------------------------------------------
@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: List[str]
    sources: List[Dict[str, Any]]

class FinAuxiRAG:
    def __init__(self):
        self.env = build_rag_stack()

    def ask(self, question: str) -> RAGResult:
        E = self.env
        retrieved = multi_query_retrieval(
            question, E["ensemble_ret"], E["dense_ret"], E["chain_hyde"], E["chain_ref"], k=25
        )
        top_docs = rerank_documents(question, truncate_documents(retrieved), E["reranker"], top_n=7)
        ctx_text = "\n\n".join(d.page_content for d in top_docs)
        answer = E["combine"].run(input_documents=top_docs, question=question)

        return RAGResult(
            question=question,
            answer=answer.strip(),
            contexts=[d.page_content for d in top_docs],
            sources=[d.metadata for d in top_docs]
        )

# -------------- Local (fast) metrics ----------------------------------------
def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def ce_max_similarity(reranker: CrossEncoder, a: str, ctxs: List[str]) -> float:
    if not ctxs or not a.strip():
        return 0.0
    pairs = [[a, c] for c in ctxs]
    scores = reranker.predict(pairs)  # higher is better
    return float(max(sigmoid(s) for s in scores))

def ce_question_context_similarity(reranker: CrossEncoder, q: str, ctxs: List[str]) -> float:
    if not ctxs or not q.strip():
        return 0.0
    pairs = [[q, c] for c in ctxs]
    scores = reranker.predict(pairs)
    return float(max(sigmoid(s) for s in scores))

def recall_at_k_with_positive(
    retriever, positive_chunk: str, query: str, k: int = 5
) -> float:
    docs = retriever.get_relevant_documents(query)
    return 1.0 if any(positive_chunk.strip() == d.page_content.strip() for d in docs[:k]) else 0.0

# -------------- Optional TruLens instrumentation ----------------------------
# We instrument a tiny "app" that returns {"answer": ..., "contexts": [...]}
if TRULENS_OK:
    @instrument
    def trulens_query(app: FinAuxiRAG, question: str) -> Dict[str, Any]:
        res = app.ask(question)
        return {"answer": res.answer, "contexts": res.contexts}

# -------------- Load QA pairs (your format) ---------------------------------
def load_qa_pairs(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows

# -------------- Main evaluation loop ----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="data/qa_pairs.jsonl", help="JSONL with fields: query, positive, negatives[] (optional)")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to run (0=all)")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV path to save results")
    parser.add_argument("--trulens", action="store_true", help="Also log LLM-judge metrics to TruLens (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa, limit=args.limit if args.limit > 0 else None)
    if not qa_pairs:
        print(f"[warn] No rows found in {args.qa}")
        return

    rag = FinAuxiRAG()
    reranker = rag.env["reranker"]

    # Optional TruLens setup
    tru_app = None
    if args.trulens:
        if not TRULENS_OK:
            print(f"[warn] TruLens not available: {TRULENS_ERR}")
        else:
            tru = Tru()  # will use local sqlite DB in ~/.trulens
            try:
                provider = TruOpenAI()  # requires OPENAI_API_KEY
                f_ans_rel = Feedback(provider.relevance, name="Answer Relevance") \
                    .on_input() \
                    .on(Select.Record.app.output["answer"])
                f_ctx_rel = Feedback(provider.context_relevance, name="Context Relevance") \
                    .on_input() \
                    .on(Select.Record.app.output["contexts"])
                f_grounded = Feedback(provider.groundedness_with_cot_reasons, name="Groundedness (CoT)") \
                    .on(Select.Record.app.output["contexts"]) \
                    .on(Select.Record.app.output["answer"])

                tru_app = TruCustomApp(
                    app=rag,                                  # our RAG object
                    app_id="FinAuxi-RAG",
                    record_app=trulens_query,                 # instrumented function
                    feedbacks=[f_ans_rel, f_ctx_rel, f_grounded],
                )
                print("[info] TruLens enabled. Records will be stored locally.")
            except Exception as e:
                print(f"[warn] Could not initialize TruLens provider/judges: {e}")

    # Run eval
    import pandas as pd
    rows = []
    for row in tqdm(qa_pairs, desc="Evaluating"):
        q = row.get("query") or row.get("question") or ""
        if not q.strip():
            continue

        res = rag.ask(q)

        # local metrics
        m_ctx_rel = ce_question_context_similarity(reranker, q, res.contexts)
        m_ans_ground = ce_max_similarity(reranker, res.answer, res.contexts)

        m_recall = None
        if "positive" in row and isinstance(row["positive"], str) and row["positive"].strip():
            # approximate: recall@5 on dense retriever only
            m_recall = recall_at_k_with_positive(rag.env["dense_ret"], row["positive"], q, k=5)

        rows.append(dict(
            question=q,
            answer=res.answer,
            ctx_relevance_ce=round(m_ctx_rel, 3),
            ans_groundedness_ce=round(m_ans_ground, 3),
            recall_at5=m_recall,
            n_ctx=len(res.contexts),
            first_source=(res.sources[0] if res.sources else {})
        ))

        # TruLens record
        if tru_app is not None:
            try:
                # This will record inputs/outputs and run the feedbacks asynchronously.
                tru_app.record(q)
            except Exception as e:
                print(f"[warn] TruLens record failed for one sample: {e}")

    df = pd.DataFrame(rows)
    print(df.head(10))
    print("\nAverages (local metrics):")
    print(df[["ctx_relevance_ce", "ans_groundedness_ce"]].mean(numeric_only=True))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[info] Saved CSV to {args.out}")

    if args.trulens and TRULENS_OK:
        print("\n[TruLens] To view the dashboard, run:")
        print("    python -m trulens_eval.dashboard --port 8501")
        print("Then open http://localhost:8501/")

if __name__ == "__main__":
    main()
