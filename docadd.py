import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
PDF_PATH = "got.pdf"
FAISS_INDEX_PATH = "faiss_index_"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Load PDF
print("[INFO] Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Split into chunks
print("[INFO] Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3500,           # Roughly 500 words
    chunk_overlap=200,         # Helps maintain continuity between chunks
    separators=["\n\n", "\n", ".", " ", ""]
)
docs = text_splitter.split_documents(documents)

# Generate embeddings
print("[INFO] Generating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"}  # use "cpu" if needed
)

# Create FAISS index
print("[INFO] Creating FAISS index...")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"[INFO] Index saved at {FAISS_INDEX_PATH}")
