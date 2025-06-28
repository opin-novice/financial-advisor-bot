import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
PDF_PATHS = ["got.pdf","BDtax.pdf"]
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Load PDFs
print("[INFO] Loading PDFs...")
documents = []
for pdf_path in PDF_PATHS:
    loader = PyPDFLoader(pdf_path)
    loaded_docs = loader.load()

    # Add source filename to metadata and clean text
    for doc in loaded_docs:
        if 'page' not in doc.metadata:
            doc.metadata['page'] = 0
        doc.metadata['source'] = pdf_path
        doc.page_content = doc.page_content.replace("\n", " ").strip()

    documents.extend(loaded_docs)

# Split into smaller, overlapping chunks
print("[INFO] Splitting into chunks...")
"""text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # ~500 tokens, focused chunks
    chunk_overlap=150,      # overlap for context continuity
    separators=["\n\n", "\n", ".", " "]
)"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # checked with 500 tokens, now checking with 300 for finer-grained retrieval
    chunk_overlap=35,      
    separators=["\n\n", "\n", ".", " "]
)

docs = text_splitter.split_documents(documents)

# Filter out tiny chunks
docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]

print(f"[INFO] Number of chunks after filtering: {len(docs)}")

# Generate embeddings on CPU
print("[INFO] Generating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

# Create FAISS index
print("[INFO] Creating FAISS index...")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"[INFO] Index saved at {FAISS_INDEX_PATH}")
