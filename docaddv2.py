#!/usr/bin/env python3
"""
Simple and Stable Text Chunking and ChromaDB Storage
===================================================
A lightweight script that:
- Chunks text files into 6 sentences with 3 sentence overlap using sentence-based chunking
- Uses BGE-M3 embeddings for superior multilingual support
- Processes files one at a time to avoid memory issues
- Simple error handling and logging
- No complex state management or file locking

Usage:
  python docadd.py
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import torch
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('docadd.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TEXT_DIR = Path("data")
CHROMA_DB_PATH = Path("chroma_db_bge_m3")
SENTENCES_PER_CHUNK = 6
SENTENCE_OVERLAP = 3
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000  # Adjusted for sentence-based chunks

# Use BGE-M3 embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"

def simple_sentence_split(text: str) -> List[str]:
    """Simple sentence splitting that works with Bengali and English."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split on sentence endings (including Bengali)
    sentences = re.split(r'[.!?।]+', text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def chunk_text(text: str) -> List[str]:
    """Create chunks with specified number of sentences and overlap."""
    sentences = simple_sentence_split(text)
    
    if len(sentences) <= SENTENCES_PER_CHUNK:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []
    
    chunks = []
    start = 0
    
    while start < len(sentences):
        end = start + SENTENCES_PER_CHUNK
        chunk_sentences = sentences[start:end]
        chunk_text = ' '.join(chunk_sentences).strip()
        
        # Validate chunk size
        if MIN_CHUNK_SIZE <= len(chunk_text) <= MAX_CHUNK_SIZE:
            chunks.append(chunk_text)
        
        # Move start position with overlap
        start = end - SENTENCE_OVERLAP
        
        # Prevent infinite loop
        if start >= len(sentences):
            break
    
    return chunks

def extract_text_from_file(file_path: Path) -> str:
    """Extract text from file, handling metadata headers."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip metadata headers if present
        if content.startswith("=== Advanced OCR Text Extraction Result ===") or content.startswith("=== Bangla Text Extraction Result ==="):
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("=") and len(line.strip()) > 30:
                    start_idx = i + 1
                    while start_idx < len(lines) and lines[start_idx].strip() == "":
                        start_idx += 1
                    return '\n'.join(lines[start_idx:])
        
        return content
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return ""

def process_single_file(file_path: Path, model, collection) -> int:
    """Process a single file and add chunks to ChromaDB."""
    logger.info(f"Processing: {file_path.name}")
    
    # Extract text
    text_content = extract_text_from_file(file_path)
    
    if not text_content or len(text_content.strip()) < 50:
        logger.warning(f"Skipping {file_path.name} - insufficient content")
        return 0
    
    # Create chunks
    chunks = chunk_text(text_content)
    logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
    
    if not chunks:
        logger.warning(f"No valid chunks created from {file_path.name}")
        return 0
    
    # Process chunks in smaller batches to avoid memory issues
    batch_size = 5  # Reduced batch size for BGE-M3
    total_added = 0
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # Create metadata for each chunk
        metadatas = []
        ids = []
        
        for j, chunk in enumerate(batch_chunks):
            chunk_id = f"{file_path.stem}_{i + j}"
            metadata = {
                'source': file_path.name,
                'chunk_id': i + j,
                'chunk_size': len(chunk),
                'chunking_method': 'SentenceBased',
                'sentences_per_chunk': SENTENCES_PER_CHUNK,
                'sentence_overlap': SENTENCE_OVERLAP,
                'file_path': str(file_path),
                'processed_at': datetime.now().isoformat()
            }
            
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        try:
            # Add to ChromaDB with embeddings
            collection.add(
                documents=batch_chunks,
                metadatas=metadatas,
                ids=ids,
                embeddings=model.embed_documents(batch_chunks)
            )
            total_added += len(batch_chunks)
            logger.info(f"Added {len(batch_chunks)} chunks to ChromaDB")
            
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            continue
        
        # Clear memory
        del batch_chunks
        del metadatas
        del ids
        gc.collect()
    
    return total_added

def main():
    """Main function with simple, stable processing."""
    logger.info("Starting simple text chunking and ChromaDB storage")
    logger.info("=" * 50)
    
    # Check if data directory exists
    if not TEXT_DIR.exists():
        logger.error(f"Data directory {TEXT_DIR} not found")
        return
    
    # Get text files
    text_files = list(TEXT_DIR.glob("*.txt"))
    if not text_files:
        logger.error(f"No text files found in {TEXT_DIR}")
        return
    
    logger.info(f"Found {len(text_files)} text files")
    
    # Initialize embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": device,
                "trust_remote_code": True  # Required for BGE-M3
            },
            encode_kwargs={
                "normalize_embeddings": True  # BGE-M3 works better with normalized embeddings
            }
        )
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return
    
    # Initialize ChromaDB
    logger.info("Initializing ChromaDB")
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name="text_chunks",
            metadata={"description": "Text chunks with sentence-based chunking (6 sentences, 3 overlap) and BGE-M3"}
        )
        
        logger.info("ChromaDB initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        return
    
    # Process files one by one
    total_chunks = 0
    processed_files = 0
    
    for file_path in text_files:
        try:
            logger.info(f"Starting file {processed_files + 1}/{len(text_files)}: {file_path.name}")
            chunks_added = process_single_file(file_path, model, collection)
            total_chunks += chunks_added
            processed_files += 1
            
            logger.info(f"Completed {processed_files}/{len(text_files)} files - Total chunks: {total_chunks}")
            
            # Force garbage collection after each file
            gc.collect()
            
            # Small delay to prevent overwhelming the system
            import time
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            continue
    
    # Summary
    logger.info("=" * 50)
    logger.info("Processing completed!")
    logger.info(f"Files processed: {processed_files}/{len(text_files)}")
    logger.info(f"Total chunks added: {total_chunks}")
    logger.info(f"ChromaDB location: {CHROMA_DB_PATH}")
    
    # Clean up
    del model
    gc.collect()

if __name__ == "__main__":
    main()
