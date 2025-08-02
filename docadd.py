import os
import fitz  # PyMuPDF
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
PDF_DIR = "data"              # Folder containing your PDF files
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ✅ Tunable Chunk Settings
CHUNK_SIZE = 1500             # Larger chunks for better context
CHUNK_OVERLAP = 150           # Overlap to maintain continuity

# --- Load PDF file paths ---
PDF_PATHS = [
    os.path.join(PDF_DIR, f)
    for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
]

# --- Extract raw text from PDFs ---
def extract_text(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    return full_text

# --- Text Splitter (Configurable) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "]
)

# --- Load and Split PDFs ---
print("[INFO] Starting PDF processing for FAISS indexing...")
documents = []
for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        print(f"[WARNING] File not found: {pdf_path}")
        continue

    print(f"[INFO] Reading: {pdf_path}")
    raw_text = extract_text(pdf_path)

    # Split text into chunks with metadata
    chunks = text_splitter.create_documents([raw_text], metadatas=[{
        "source": os.path.basename(pdf_path)
    }])

    # Filter out very short chunks
    chunks = [doc for doc in chunks if len(doc.page_content.strip()) > 50]
    print(f"[INFO] ✅ {len(chunks)} chunks created from {os.path.basename(pdf_path)}")

    documents.extend(chunks)

print(f"[INFO] Total chunks prepared for embedding: {len(documents)}")

# --- Embedding Setup ---
print("[INFO] Generating embeddings...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device}
)

# --- Create and Save FAISS Index ---
print("[INFO] Creating FAISS vector index...")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"[INFO] ✅ FAISS index saved at: {FAISS_INDEX_PATH}")
print("[DONE] ✅ Embedding and indexing completed successfully.")
