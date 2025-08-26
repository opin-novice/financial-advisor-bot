#!/usr/bin/env python3
"""
Vanilla RAG Pipeline with Ollama
================================
A simple, clean implementation of a basic RAG (Retrieval-Augmented Generation) pipeline using Ollama.
This demonstrates the core concepts without advanced features for educational purposes.

Core Components:
1. Document Loading & Chunking
2. Embedding Generation
3. Vector Storage (FAISS)
4. Retrieval
5. Generation with Ollama LLM

Requirements:
- Ollama installed and running locally
- A model pulled (e.g., llama3.2, mistral, codellama)
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any
from pathlib import Path

# Core RAG libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Telegram bot libraries
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VanillaRAGPipeline:
    """
    A simple RAG pipeline implementation using Ollama for generation.
    """
    
    def __init__(self, 
                 pdf_dir: str = "data",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 ollama_model: str = "llama3.2",
                 ollama_base_url: str = "http://localhost:11434",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the vanilla RAG pipeline with Ollama.
        
        Args:
            pdf_dir: Directory containing PDF documents
            embedding_model: HuggingFace model for embeddings
            ollama_model: Ollama model name for generation
            ollama_base_url: Ollama server URL
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.pdf_dir = pdf_dir
        self.embedding_model = embedding_model
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
        logger.info("Vanilla RAG Pipeline with Ollama initialized")
    
    def load_documents(self) -> List[Any]:
        """
        Load PDF documents from the specified directory.
        
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading documents from {self.pdf_dir}")
        
        documents = []
        pdf_dir = Path(self.pdf_dir)
        
        if not pdf_dir.exists():
            logger.error(f"Directory {self.pdf_dir} does not exist")
            return documents
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Load each PDF
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of loaded documents
            
        Returns:
            List of chunked documents
        """
        logger.info("Chunking documents")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def create_embeddings(self):
        """
        Initialize the embedding model.
        """
        logger.info(f"Initializing embedding model: {self.embedding_model}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"}
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def create_vectorstore(self, chunks: List[Any], save_path: str = "faiss_index"):
        """
        Create and store embeddings in a FAISS vector store.
        
        Args:
            chunks: List of document chunks
            save_path: Path to save the FAISS index
        """
        logger.info("Creating vector store")
        
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call create_embeddings() first.")
        
        try:
            # Create vector store
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            # Save the index
            self.vectorstore.save_local(save_path)
            logger.info(f"Vector store created and saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vectorstore(self, load_path: str = "faiss_index"):
        """
        Load an existing FAISS vector store.
        
        Args:
            load_path: Path to the FAISS index
        """
        logger.info(f"Loading vector store from {load_path}")
        
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call create_embeddings() first.")
        
        try:
            self.vectorstore = FAISS.load_local(load_path, self.embeddings)
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def initialize_llm(self):
        """
        Initialize the Ollama language model for generation.
        """
        logger.info(f"Initializing Ollama LLM: {self.ollama_model}")
        
        try:
            self.llm = OllamaLLM(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=0.1
            )
            logger.info("Ollama LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM: {e}")
            logger.error("Make sure Ollama is running and the model is pulled")
            logger.error(f"Try: ollama pull {self.ollama_model}")
            raise
    
    def create_qa_chain(self):
        """
        Create the question-answering chain.
        """
        if not self.vectorstore or not self.llm:
            raise ValueError("Vector store and LLM must be initialized first")
        
        logger.info("Creating QA chain")
        
        # Create a simple prompt template
        prompt_template = """
        You are a helpful assistant. Answer the question based on the provided context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("QA chain created successfully")
    
    def query(self, question: str) -> str:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            The generated answer
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call create_qa_chain() first.")
        
        logger.info(f"Processing question: {question}")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            answer = result.get("result", "No answer generated")
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error: {str(e)}"
    
    def build_pipeline(self, force_rebuild: bool = False):
        """
        Build the complete RAG pipeline.
        
        Args:
            force_rebuild: If True, rebuild the vector store even if it exists
        """
        logger.info("Building RAG pipeline")
        
        # Step 1: Initialize embeddings
        self.create_embeddings()
        
        # Step 2: Check if vector store exists
        if not force_rebuild and os.path.exists("faiss_index"):
            logger.info("Loading existing vector store")
            self.load_vectorstore()
        else:
            logger.info("Creating new vector store")
            # Step 3: Load and chunk documents
            documents = self.load_documents()
            if not documents:
                raise ValueError("No documents loaded")
            
            chunks = self.chunk_documents(documents)
            
            # Step 4: Create vector store
            self.create_vectorstore(chunks)
        
        # Step 5: Initialize LLM
        self.initialize_llm()
        
        # Step 6: Create QA chain
        self.create_qa_chain()
        
        logger.info("RAG pipeline built successfully")
    
    def get_relevant_documents(self, question: str, k: int = 3) -> List[Any]:
        """
        Retrieve relevant documents for a question.
        
        Args:
            question: The question to retrieve documents for
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        documents = retriever.get_relevant_documents(question)
        
        return documents

def check_ollama_status():
    """Check if Ollama is running and available."""
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"Ollama is running. Available models: {[m['name'] for m in models]}")
            return True, models
        else:
            logger.error(f"Ollama responded with status code: {response.status_code}")
            return False, []
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not connect to Ollama: {e}")
        logger.error("Make sure Ollama is running on http://localhost:11434")
        return False, []

def run_demo(rag: VanillaRAGPipeline):
    """Run the demo with example questions."""
    try:
        # Build the pipeline
        rag.build_pipeline()
        
        # Example questions
        example_questions = [
            "What are the requirements for opening a bank account in Bangladesh?",
            "How do I apply for a loan?",
            "What are the current interest rates?",
            "What documents do I need for tax filing?"
        ]
        
        print("\n" + "="*60)
        print("VANILLA RAG PIPELINE WITH OLLAMA DEMO")
        print("="*60)
        
        # Process each question
        for i, question in enumerate(example_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 50)
            
            # Get relevant documents
            docs = rag.get_relevant_documents(question, k=2)
            print(f"Retrieved {len(docs)} relevant documents")
            
            # Generate answer
            answer = rag.query(question)
            print(f"Answer: {answer}")
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"Error: {e}")

def main():
    """
    Main function with choice between demo and Telegram bot.
    """
    # ========================================
    # CONFIGURATION - EDIT THESE VALUES
    # ========================================
    
    # Other configuration options
    PDF_DIR = "data"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    OLLAMA_MODEL = "llama3.2:3b"  # Change this to your preferred Ollama model
    OLLAMA_BASE_URL = "http://localhost:11434"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # ========================================
    
    # Check Ollama status first
    print("🔍 Checking Ollama status...")
    ollama_running, available_models = check_ollama_status()
    
    if not ollama_running:
        print("❌ Error: Ollama is not running or not accessible")
        print("💡 To fix this:")
        print("   1. Install Ollama from https://ollama.ai/")
        print("   2. Start Ollama service")
        print("   3. Pull a model: ollama pull llama3.2")
        print("   4. Make sure Ollama is running on http://localhost:11434")
        return
    
    # Check if the specified model is available
    model_names = [m['name'] for m in available_models]
    if OLLAMA_MODEL not in model_names:
        print(f"⚠️ Warning: Model '{OLLAMA_MODEL}' not found in available models")
        print(f"💡 Available models: {', '.join(model_names)}")
        print(f"💡 To pull the model: ollama pull {OLLAMA_MODEL}")
        print(f"💡 Or change OLLAMA_MODEL in the script to one of the available models")
        
        # Ask user if they want to continue
        choice = input(f"\nDo you want to continue with '{OLLAMA_MODEL}' anyway? (y/n): ").strip().lower()
        if choice != 'y':
            return
    
    # Configuration dictionary
    config = {
        "pdf_dir": PDF_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "ollama_model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }
    
    # Initialize pipeline
    rag = VanillaRAGPipeline(**config)
    
    # Run demo
    run_demo(rag)

if __name__ == "__main__":
    main()
