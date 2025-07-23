import os
import re
import warnings
from datasets import Dataset
import pandas as pd

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# --- Suppress async cleanup warnings ---
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configurations ---
INDEX_DIR = "faiss_index"
TEST_PDF_DIR = "test_questions"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- Load FAISS index ---
print("[INFO] Loading FAISS index...")
hf_embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
ragas_embedding = LangchainEmbeddingsWrapper(hf_embedding)

vectorstore = FAISS.load_local(
    INDEX_DIR,
    embeddings=hf_embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# --- Parse test PDFs ---
print("[INFO] Parsing test PDFs from:", TEST_PDF_DIR)
examples = []
for f in os.listdir(TEST_PDF_DIR):
    if not f.lower().endswith(".pdf"):
        continue
    path = os.path.join(TEST_PDF_DIR, f)
    loader = PyPDFLoader(path)
    pages = loader.load()
    full_text = " ".join(p.page_content for p in pages)

    qa_pairs = re.findall(
        r"(?i)(?:\d+\.\s*)?Question[:\s]+(.+?)\s+Answer[:\s]+(.+?)(?=(?:\n\s*\d+\.\s*Question:|\Z))",
        full_text,
        re.DOTALL
    )
    if not qa_pairs:
        print(f"[WARN] No Q/A found in: {f}")
    else:
        for q, a in qa_pairs:
            examples.append({"question": q.strip(), "ground_truth": a.strip()})

if not examples:
    raise ValueError("No valid Q/A pairs found in test PDFs.")
dataset = Dataset.from_list(examples)

# --- Setup RAG pipeline ---
llm = OllamaLLM(model="llama3.2:3b")
ragas_llm = LangchainLLMWrapper(llm)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

print("[INFO] Running RAG pipeline...")
answers, contexts = [], []
for item in dataset:
    res = qa_chain.invoke({"query": item["question"]})
    answers.append(res["result"])
    contexts.append([d.page_content for d in res["source_documents"]])

dataset = dataset.add_column("answer", answers)
dataset = dataset.add_column("contexts", contexts)

# --- Evaluate using RAGAS ---
print("[INFO] Evaluating with RAGAS...")
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embedding,
    raise_exceptions=False,    # Don't crash on timeouts or parser failures
    show_progress=True,
    batch_size=4               # Helps reduce TimeoutErrors
)

# --- Safe Mean Display for Numeric Columns Only ---
df = results.to_pandas()

print("\n📊 RAGAS Evaluation Results (averages across samples):")
for col in df.columns:
    try:
        # Convert column to numeric where possible (non-numeric cells become NaN)
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean(skipna=True)
            print(f"{col}: {mean_val:.3f}")
        else:
            print(f"{col}: [non-numeric]")
    except Exception as e:
        print(f"[ERROR] Could not compute mean for column '{col}': {e}")

# --- Save results ---
df.to_csv("ragas_output.csv", index=False)
print("\n✅ Full results saved to ragas_output.csv")
