#!/usr/bin/env python3
"""
Embed JSON Chunks with BGE-M3 Model

This script processes JSON files containing text chunks from the data/ directory,
generates embeddings using the BGE-M3 model, and saves the embedded data to
a new embedded_data/ directory while preserving the original structure.

The script is designed to be standalone and will not modify any existing files.
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_bge_m3_model():
    """
    Load the BGE-M3 model locally.
    
    Returns:
        SentenceTransformer: The loaded BGE-M3 model
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading BGE-M3 model...")
        model = SentenceTransformer('BAAI/bge-m3', device='cpu')
        logger.info("BGE-M3 model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load BGE-M3 model: {e}")
        raise

def process_json_files(input_dir: Path, output_dir: Path, model):
    """
    Process JSON files and generate embeddings for each chunk's content.
    
    Args:
        input_dir (Path): Directory containing input JSON files
        output_dir (Path): Directory to save embedded JSON files
        model: BGE-M3 model for embedding generation
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        logger.warning("No JSON files found in the input directory")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Process each JSON file
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            process_single_json_file(json_file, output_dir, model)
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {e}")
            continue

def process_single_json_file(json_file: Path, output_dir: Path, model):
    """
    Process a single JSON file and generate embeddings.
    
    Args:
        json_file (Path): Path to the input JSON file
        output_dir (Path): Directory to save the embedded JSON file
        model: BGE-M3 model for embedding generation
    """
    logger.info(f"Processing {json_file.name}")
    
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"  -> Loaded {len(chunks)} chunks")
    
    # Process each chunk to add embeddings
    embedded_chunks = []
    for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding chunks in {json_file.name}", leave=False)):
        try:
            # Extract content
            content = chunk.get("content", "")
            if not content.strip():
                logger.warning(f"  -> Chunk {i} has empty content, skipping")
                continue
            
            # Generate embedding
            embedding = model.encode(content)
            
            # Convert to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Add embedding to chunk
            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding_list
            
            embedded_chunks.append(embedded_chunk)
            
        except Exception as e:
            logger.error(f"  -> Error embedding chunk {i} in {json_file.name}: {e}")
            # Keep original chunk even if embedding fails
            embedded_chunks.append(chunk)
    
    # Save embedded data to new file
    output_file = output_dir / json_file.name
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"  -> Saved {len(embedded_chunks)} embedded chunks to {output_file.name}")

def main():
    """Main function to run the embedding process."""
    logger.info("Starting JSON chunk embedding process with BGE-M3 model")
    
    # Define directories
    project_root = Path(__file__).parent
    input_dir = project_root / "data"
    output_dir = project_root / "embedded_data"
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    # Load model
    try:
        model = load_bge_m3_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return
    
    # Process files
    try:
        process_json_files(input_dir, output_dir, model)
        logger.info("Embedding process completed successfully")
    except Exception as e:
        logger.error(f"Embedding process failed: {e}")

if __name__ == "__main__":
    main()