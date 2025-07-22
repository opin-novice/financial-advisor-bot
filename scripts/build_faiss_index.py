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

# Setup logging
logging.basicConfig(
    filename='logs/indexing.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DOCUMENT_REGISTRY = "data/document_registry.json"

def load_document_registry():
    """Load the document registry or create a new one if it doesn't exist"""
    if os.path.exists(DOCUMENT_REGISTRY):
        with open(DOCUMENT_REGISTRY, 'r') as f:
            return json.load(f)
    return {"documents": []}

def save_document_registry(registry):
    """Save the document registry"""
    with open(DOCUMENT_REGISTRY, 'w') as f:
        json.dump(registry, f, indent=2)

def process_and_index_documents():
    """Process PDF documents and build FAISS index"""
    logger.info("Starting document processing and indexing")
    
    # Initialize PDF processor for quality checks
    pdf_processor = PDFProcessor()
    
    # Load document registry
    registry = load_document_registry()
    processed_files = [doc["file_path"] for doc in registry["documents"]]
    
    # Process each category directory
    all_documents = []
    categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for category in categories:
        category_dir = os.path.join(DATA_DIR, category)
        processed_category_dir = os.path.join(PROCESSED_DIR, category)
        os.makedirs(processed_category_dir, exist_ok=True)
        
        logger.info(f"Processing category: {category}")
        
        # Get all PDF files in the category directory
        pdf_files = [f for f in os.listdir(category_dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(category_dir, pdf_file)
            
            # Skip already processed files
            if file_path in processed_files:
                logger.info(f"Skipping already processed file: {file_path}")
                continue
            
            logger.info(f"Processing file: {file_path}")
            
            # Check PDF quality
            quality_result = pdf_processor.process_pdf(file_path)
            
            if quality_result["status"] == "error":
                logger.error(f"Error processing {file_path}: {quality_result['error']}")
                continue
                
            if not quality_result["passes_threshold"]:
                logger.warning(f"File {file_path} did not pass quality thresholds: {quality_result['metrics']}")
                continue
            
            # Load and process the PDF
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata["category"] = category
                    doc.metadata["file_name"] = pdf_file
                
                all_documents.extend(documents)
                
                # Add to registry
                registry["documents"].append({
                    "file_path": file_path,
                    "category": category,
                    "processed_date": datetime.now().isoformat(),
                    "quality_metrics": quality_result["metrics"],
                    "pages": len(documents)
                })
                
                logger.info(f"Successfully processed {file_path} with {len(documents)} pages")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
    
    # Save updated registry
    save_document_registry(registry)
    
    if not all_documents:
        logger.warning("No documents to process. Index not updated.")
        return
    
    # Split documents into chunks
    logger.info("Splitting documents into chunks")
    # Adjust chunk size for better memory management
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced from 1000 for better memory management
        chunk_overlap=100,  # Reduced from 200
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    logger.info(f"Created {len(chunks)} chunks from {len(all_documents)} documents")
    
    # Generate embeddings and create FAISS index
    logger.info("Generating embeddings and creating FAISS index")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"}  # use "cpu" if needed
    )
    
    # Check if index already exists
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("Loading existing index")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("Adding new documents to existing index")
        vectorstore.add_documents(chunks)
    else:
        logger.info("Creating new index")
        vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the index
    vectorstore.save_local(FAISS_INDEX_PATH)
    logger.info(f"Index saved at {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    process_and_index_documents()