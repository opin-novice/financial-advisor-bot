import os
import re
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
PDF_DIR = "data"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- Auto-detect PDF files ---
PDF_PATHS = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
if not PDF_PATHS:
    raise FileNotFoundError(f"No PDF files found in directory: {PDF_DIR}")

print(f"[INFO] Detected {len(PDF_PATHS)} PDF files:")
for path in PDF_PATHS:
    print(" -", path)

# --- Text Cleaning Function ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple whitespace into single space
    text = text.lower()               # Normalize case
    # Optional content removals
    text = re.sub(r'(all rights reserved|table of contents)', '', text)
    # Example redactions
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', text)  # Fake SSN format
    text = re.sub(r'\b\d{16}\b', '[REDACTED CC]', text)              # Fake credit card format
    return text.strip()

# --- Load PDFs and sanitize ---
print("[INFO] Loading and sanitizing PDFs...")
documents = []
for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    loaded_docs = loader.load()

    for doc in loaded_docs:
        if 'page' not in doc.metadata:
            doc.metadata['page'] = 0
        doc.metadata['source'] = os.path.basename(pdf_path)

        # Clean and sanitize text
        raw_text = doc.page_content.replace("\n", " ")
        doc.page_content = clean_text(raw_text)

    documents.extend(loaded_docs)

# --- Text Splitter ---
print("[INFO] Splitting sanitized text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separators=["\n\n", "\n", ".", " "]
)

docs = text_splitter.split_documents(documents)

# Filter out very short chunks
docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]
print(f"[INFO] Number of chunks after filtering: {len(docs)}")

# --- Embedding Setup ---
print("[INFO] Generating embeddings...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device}
)

# --- Create and Save FAISS Index ---
print("[INFO] Creating FAISS index...")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"[INFO] ✅ Index saved at: {FAISS_INDEX_PATH}")
