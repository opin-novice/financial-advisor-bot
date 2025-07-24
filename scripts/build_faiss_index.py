import os
import json
import logging
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.pdf_processor import PDFProcessor
from src.utils.document_manager import DocumentManager

# Setup logging
logging.basicConfig(
    filename='logs/indexing.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PROCESSED_DIR = "data/processed"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def process_and_index_documents():
    """Process PDF documents and build FAISS index"""
    logger.info("Starting document processing and indexing")
    
    # Initialize document manager
    document_manager = DocumentManager()
    
    # Process each category directory in the processed folder
    all_documents = []
    categories = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    
    for category in categories:
        category_dir = os.path.join(PROCESSED_DIR, category)
        
        logger.info(f"Processing category: {category}")
        
        # Get all PDF files in the category directory
        pdf_files = [f for f in os.listdir(category_dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(category_dir, pdf_file)
            
            logger.info(f"Processing file: {file_path}")
            
            # Load and process the PDF
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata["category"] = category
                    doc.metadata["file_name"] = pdf_file
                
                all_documents.extend(documents)
                logger.info(f"Successfully processed {file_path} with {len(documents)} pages")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
    
    if not all_documents:
        logger.warning("No documents to process. Index not updated.")
        return
    
    # Split documents into chunks and preserve metadata
    logger.info("Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    all_chunks = []
    for doc in all_documents:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata = doc.metadata
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(all_documents)} documents")
    
    # Generate embeddings and create FAISS index
    logger.info("Generating embeddings and creating FAISS index")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # Changed to CPU for M1 Mac compatibility
    )
    
    # Check if index already exists
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("Loading existing index")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("Adding new documents to existing index")
        vectorstore.add_documents(all_chunks)
    else:
        logger.info("Creating new index")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
    
    # Save the index
    vectorstore.save_local(FAISS_INDEX_PATH)
    logger.info(f"Index saved at {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    process_and_index_documents()