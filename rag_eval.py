import os
import re
from datasets import Dataset
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper  # ✅ FIXED: correct wrapper

# --- Configuration ---
INDEX_DIR = "faiss_index"
TEST_PDF_DIR = "test_questions"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- Load FAISS index ---
print("[INFO] Loading FAISS index...")
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(
    INDEX_DIR,
    embeddings=embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# --- Load and parse PDFs into question/ground_truth ---
print("[INFO] Parsing test PDFs from:", TEST_PDF_DIR)
examples = []

pdf_files = [
    os.path.join(TEST_PDF_DIR, f)
    for f in os.listdir(TEST_PDF_DIR)
    if f.lower().endswith(".pdf")
]

if not pdf_files:
    raise FileNotFoundError(f"No PDF test files found in {TEST_PDF_DIR}")

for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    full_text = " ".join([page.page_content for page in pages])

    # Extract all question-answer pairs using improved regex
    qa_pairs = re.findall(
        r"(?i)(?:\d+\.\s*)?Question[:\s]+(.+?)\s+Answer[:\s]+(.+?)(?=(?:\n\s*\d+\.\s*Question:|\Z))",
        full_text,
        re.DOTALL
    )

    if not qa_pairs:
        print(f"[WARN] Could not parse any question/answer pairs from: {pdf_path}")
    else:
        for question, answer in qa_pairs:
            examples.append({
                "question": question.strip(),
                "ground_truth": answer.strip()
            })

if not examples:
    raise ValueError("No valid question/answer pairs found in test PDFs.")

dataset = Dataset.from_list(examples)

# --- Setup RAG pipeline ---
llm = OllamaLLM(model="llama3.2:3b")  # ✅ UPDATED to use llama3.2:3b
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- Run RAG and collect answers ---
print("[INFO] Running RAG pipeline...")
answers, contexts = [], []
for item in dataset:
    result = qa_chain.invoke({"query": item["question"]})
    answers.append(result["result"])
    contexts.append([doc.page_content for doc in result["source_documents"]])

dataset = dataset.add_column("answer", answers)
dataset = dataset.add_column("contexts", contexts)

# --- Setup custom RAGAS LLM evaluator (no OpenAI needed) ---
ragas_llm = LangchainLLMWrapper(llm)  # ✅ Use the same llama3.2:3b model for eval

# --- Evaluate using RAGAS ---
print("[INFO] Evaluating with RAGAS...")
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ],
    llm=ragas_llm
)

print("\n📊 RAGAS Evaluation Results:")
for metric, score in results.items():
    print(f"{metric}: {score:.3f}")

dataset.to_csv("ragas_output.csv")
print("\n✅ Results saved to ragas_output.csv")
