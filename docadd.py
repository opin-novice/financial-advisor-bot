import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch

# Configuration
PDF_DIR = "data"               # relative path to the 'data' folder inside 'v2'
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Auto-detect PDF files in the data folder
PDF_PATHS = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

if not PDF_PATHS:
    raise FileNotFoundError(f"No PDF files found in directory: {PDF_DIR}")

print(f"[INFO] Detected {len(PDF_PATHS)} PDF files:")
for path in PDF_PATHS:
    print(" -", path)

# Load PDFs
print("[INFO] Loading PDFs...")
documents = []
for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    loaded_docs = loader.load()

    # Add source filename to metadata and clean text
    for doc in loaded_docs:
        if 'page' not in doc.metadata:
            doc.metadata['page'] = 0
        doc.metadata['source'] = os.path.basename(pdf_path)
        doc.page_content = doc.page_content.replace("\n", " ").strip()

    documents.extend(loaded_docs)

# Split into smaller, overlapping chunks
print("[INFO] Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separators=["\n\n", "\n", ".", " "]
)

docs = text_splitter.split_documents(documents)

# Filter out tiny chunks
docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]
print(f"[INFO] Number of chunks after filtering: {len(docs)}")

# Generate embeddings (using GPU if available)
print("[INFO] Generating embeddings...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device}
)

# Create FAISS index
print("[INFO] Creating FAISS index...")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"[INFO] Index saved at {FAISS_INDEX_PATH}")