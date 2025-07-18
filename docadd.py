import os
import re
import fitz  # PyMuPDF
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
PDF_DIR = "data"  # Folder containing your PDF files
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- Specific PDF files to load ---
PDF_PATHS = [
    os.path.join(PDF_DIR, f)
    for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
]

# --- Helper Functions for Sanitization ---
def extract_text_and_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        # Remove headers/footers (optional pattern here)
        text = re.sub(r'(HeaderPattern|FooterPattern)', '', text)
        full_text += text + "\n"
    metadata = doc.metadata
    return full_text, metadata

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = text.lower()  # Normalize case

    # Remove noise patterns
    text = re.sub(r'\bpage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(all rights reserved|table of contents)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(confidential|this document is for.*?purposes only)', '', text, flags=re.IGNORECASE)

    # Remove emails, phones, account numbers
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '[EMAIL REDACTED]', text)
    text = re.sub(r'\b\d{10,16}\b', '[ACCOUNT NO REDACTED]', text)
    text = re.sub(r'\+?88[\s-]?\d{11}\b', '[PHONE REDACTED]', text)

    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII characters

    return text.strip()


# --- Text Splitter Setup ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=338,
    chunk_overlap=36,
    separators=["\n\n", "\n", ".", " "]
)

# --- Load and Process PDFs ---
print("[INFO] Sanitizing and processing selected PDFs...")
documents = []
for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        print(f"[WARNING] File not found: {pdf_path}")
        continue

    print(f"[INFO] Processing: {pdf_path}")
    raw_text, meta = extract_text_and_metadata(pdf_path)
    cleaned_text = clean_text(raw_text)

    # Split into chunks with metadata
    chunks = text_splitter.create_documents([cleaned_text], metadatas=[{
        "source": os.path.basename(pdf_path)
    }])

    # Filter out very short chunks
    chunks = [doc for doc in chunks if len(doc.page_content.strip()) > 50]
    documents.extend(chunks)

print(f"[INFO] Total chunks after cleaning and splitting: {len(documents)}")

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
print("[DONE] Embedding and indexing completed successfully.")