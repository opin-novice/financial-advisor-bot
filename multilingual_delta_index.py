#!/usr/bin/env python3
"""
multilingual_delta_index.py
---------------------------
Incremental / full index update for multilingual financial advisor bot.
Supports both Bangla and English PDFs with semantic chunking.

Commands
--------
# nightly cron - incremental update
python multilingual_delta_index.py

# force full rebuild
python multilingual_delta_index.py --full

# process only English PDFs
python multilingual_delta_index.py --english-only

# process only Bangla PDFs
python multilingual_delta_index.py --bangla-only
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# Import our multilingual components
from multilingual_semantic_chunking import (
    MultilingualSemanticChunker, 
    MultilingualLanguageDetector,
    extract_text_multilingual
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import torch

# ---------- Configuration ----------
ENGLISH_PDF_DIR = Path("data")
BANGLA_PDF_DIR = Path("unsant_data")
STATE_FILE = Path(".multilingual_delta_state.json")
LIVE_INDEX_DIR = Path("faiss_index_multilingual")
TMP_INDEX_DIR = Path("faiss_index_multilingual_tmp")
BACKUP_INDEX_DIR = Path("faiss_index_multilingual_backup")

# Multilingual embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Semantic chunking settings (from multilingual_semantic_chunking.py)
MAX_CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 250
SIMILARITY_THRESHOLD = 0.65
OVERLAP_SENTENCES = 2

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multilingual_delta_index.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- Helper Functions ----------

def file_hash(path: Path) -> str:
    """SHA-256 of file content."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {path}: {e}")
        return ""

def load_state() -> dict:
    """Load the current state of processed files."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"Error loading state file: {e}")
            return {}
    return {}

def save_state(state: dict):
    """Save the current state of processed files."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
        logger.info(f"State saved with {len(state)} files")
    except Exception as e:
        logger.error(f"Error saving state file: {e}")

def get_pdf_files(directory: Path) -> List[Path]:
    """Get all PDF files from a directory."""
    if not directory.exists():
        logger.warning(f"Directory {directory} does not exist")
        return []
    
    pdf_files = [p for p in directory.glob("*.pdf") if not p.name.startswith(".")]
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    return pdf_files

def detect_file_language(pdf_path: Path) -> str:
    """Detect the primary language of a PDF file."""
    try:
        # Extract a sample of text for language detection
        text_sample = extract_text_multilingual(pdf_path)[:2000]  # First 2000 chars
        
        if not text_sample.strip():
            return 'unknown'
        
        detector = MultilingualLanguageDetector()
        language = detector.detect_language(text_sample)
        logger.debug(f"Detected language for {pdf_path.name}: {language}")
        return language
        
    except Exception as e:
        logger.error(f"Error detecting language for {pdf_path}: {e}")
        return 'unknown'

def build_multilingual_index_for_files(files: List[Path]) -> Optional[FAISS]:
    """Build FAISS index for given files using multilingual semantic chunking."""
    if not files:
        logger.warning("No files provided for indexing")
        return None
    
    logger.info(f"Building multilingual index for {len(files)} files...")
    
    try:
        # Initialize multilingual chunker
        chunker = MultilingualSemanticChunker(
            max_chunk_size=MAX_CHUNK_SIZE,
            min_chunk_size=MIN_CHUNK_SIZE,
            similarity_threshold=SIMILARITY_THRESHOLD,
            overlap_sentences=OVERLAP_SENTENCES
        )
        
        all_documents = []
        language_stats = {'bangla': 0, 'english': 0, 'mixed': 0, 'unknown': 0}
        
        for pdf_path in files:
            logger.info(f"Processing: {pdf_path.name}")
            
            try:
                # Extract text
                raw_text = extract_text_multilingual(pdf_path)
                
                if len(raw_text.strip()) < 100:
                    logger.warning(f"Very little text extracted from {pdf_path.name}")
                    continue
                
                # Detect language
                doc_language = detect_file_language(pdf_path)
                language_stats[doc_language] += 1
                
                # Determine source directory for metadata
                source_dir = "bangla" if str(BANGLA_PDF_DIR) in str(pdf_path) else "english"
                
                # Semantic chunking
                chunks = chunker.chunk_text(raw_text, metadata={
                    "source": pdf_path.name,
                    "full_path": str(pdf_path),
                    "document_language": doc_language,
                    "source_directory": source_dir,
                    "processed_date": datetime.now().isoformat(),
                    "file_hash": file_hash(pdf_path)
                })
                
                logger.info(f"Created {len(chunks)} chunks from {pdf_path.name} ({doc_language})")
                all_documents.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                continue
        
        if not all_documents:
            logger.error("No documents were successfully processed")
            return None
        
        # Log language statistics
        logger.info(f"Language distribution: {language_stats}")
        logger.info(f"Total chunks created: {len(all_documents)}")
        
        # Create embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device}
        )
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        vectorstore = FAISS.from_documents(all_documents, embeddings)
        logger.info("✅ Multilingual FAISS index built successfully")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error building multilingual index: {e}")
        return None

def create_backup():
    """Create backup of current index."""
    if LIVE_INDEX_DIR.exists():
        try:
            if BACKUP_INDEX_DIR.exists():
                shutil.rmtree(BACKUP_INDEX_DIR)
            shutil.copytree(LIVE_INDEX_DIR, BACKUP_INDEX_DIR)
            logger.info(f"Backup created at {BACKUP_INDEX_DIR}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")

def restore_backup():
    """Restore from backup if available."""
    if BACKUP_INDEX_DIR.exists():
        try:
            if LIVE_INDEX_DIR.exists():
                shutil.rmtree(LIVE_INDEX_DIR)
            shutil.copytree(BACKUP_INDEX_DIR, LIVE_INDEX_DIR)
            logger.info("Index restored from backup")
            return True
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
    return False

def multilingual_delta_update(force_full: bool = False, english_only: bool = False, bangla_only: bool = False):
    """Perform incremental or full update of multilingual index."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Multilingual Delta Index Update")
    logger.info("=" * 60)
    
    # Load current state
    state = load_state()
    
    # Get PDF files from both directories
    pdf_files = []
    
    if not bangla_only:
        english_files = get_pdf_files(ENGLISH_PDF_DIR)
        pdf_files.extend(english_files)
    
    if not english_only:
        bangla_files = get_pdf_files(BANGLA_PDF_DIR)
        pdf_files.extend(bangla_files)
    
    if not pdf_files:
        logger.warning("No PDF files found in any directory")
        return
    
    logger.info(f"Total PDF files found: {len(pdf_files)}")
    
    # Compute current fingerprints
    current_state = {}
    for pdf_path in pdf_files:
        try:
            relative_path = str(pdf_path.relative_to(pdf_path.parent.parent))
            file_stat = pdf_path.stat()
            current_state[relative_path] = {
                "mtime": file_stat.st_mtime,
                "sha": file_hash(pdf_path),
                "size": file_stat.st_size
            }
        except Exception as e:
            logger.error(f"Error processing file metadata for {pdf_path}: {e}")
    
    # Determine files to process
    if force_full:
        to_process = pdf_files
        logger.info("🔄 Full rebuild requested - processing all files")
    else:
        to_process = []
        for pdf_path in pdf_files:
            try:
                relative_path = str(pdf_path.relative_to(pdf_path.parent.parent))
                if relative_path not in state or state[relative_path] != current_state[relative_path]:
                    to_process.append(pdf_path)
            except Exception as e:
                logger.error(f"Error comparing file state for {pdf_path}: {e}")
                to_process.append(pdf_path)  # Process if unsure
    
    if not to_process:
        logger.info("✅ No new/changed PDFs found. Index is up to date.")
        return
    
    logger.info(f"📄 Processing {len(to_process)} PDF(s)...")
    for pdf in to_process:
        logger.info(f"  - {pdf.name}")
    
    # Create backup before making changes
    if not force_full:
        create_backup()
    
    try:
        # Clean up temporary directory
        if TMP_INDEX_DIR.exists():
            shutil.rmtree(TMP_INDEX_DIR)
        
        # Build new index
        new_index = build_multilingual_index_for_files(to_process)
        
        if new_index is None:
            logger.error("❌ Failed to build new index")
            if not force_full:
                logger.info("Attempting to restore from backup...")
                restore_backup()
            return
        
        # Handle incremental vs full update
        if not force_full and LIVE_INDEX_DIR.exists():
            logger.info("🔄 Performing incremental update - merging with existing index")
            try:
                # Load existing index
                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
                
                old_index = FAISS.load_local(
                    str(LIVE_INDEX_DIR),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Merge indices
                logger.info("Merging new index with existing index...")
                old_index.merge_from(new_index)
                
                # Save merged index to temporary location
                old_index.save_local(str(TMP_INDEX_DIR))
                logger.info("✅ Index merge completed")
                
            except Exception as e:
                logger.error(f"Error during incremental update: {e}")
                logger.info("Falling back to full rebuild...")
                new_index.save_local(str(TMP_INDEX_DIR))
        else:
            logger.info("💾 Saving new index...")
            new_index.save_local(str(TMP_INDEX_DIR))
        
        # Atomic swap - replace live index
        if LIVE_INDEX_DIR.exists():
            shutil.rmtree(LIVE_INDEX_DIR)
        TMP_INDEX_DIR.rename(LIVE_INDEX_DIR)
        
        # Update state for processed files
        for pdf_path in to_process:
            try:
                relative_path = str(pdf_path.relative_to(pdf_path.parent.parent))
                state[relative_path] = current_state[relative_path]
            except Exception as e:
                logger.error(f"Error updating state for {pdf_path}: {e}")
        
        save_state(state)
        
        logger.info("=" * 60)
        logger.info("✅ Multilingual index updated successfully!")
        logger.info(f"📊 Total files in state: {len(state)}")
        logger.info(f"📊 Files processed this run: {len(to_process)}")
        logger.info(f"📁 Index location: {LIVE_INDEX_DIR}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Critical error during index update: {e}")
        if not force_full:
            logger.info("Attempting to restore from backup...")
            if restore_backup():
                logger.info("✅ Successfully restored from backup")
            else:
                logger.error("❌ Failed to restore from backup")
        raise

def verify_index():
    """Verify that the multilingual index is working correctly."""
    logger.info("🔍 Verifying multilingual index...")
    
    try:
        if not LIVE_INDEX_DIR.exists():
            logger.error("❌ Index directory does not exist")
            return False
        
        # Load the index
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        
        vectorstore = FAISS.load_local(
            str(LIVE_INDEX_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Test searches in both languages
        test_queries = [
            ("bank account opening", "english"),
            ("ব্যাংক অ্যাকাউন্ট", "bangla"),
            ("loan application", "english"),
            ("ঋণের আবেদন", "bangla")
        ]
        
        for query, lang in test_queries:
            try:
                results = vectorstore.similarity_search(query, k=3)
                logger.info(f"✅ {lang.title()} search '{query}': {len(results)} results")
                
                if results:
                    # Show sample result
                    sample = results[0]
                    doc_lang = sample.metadata.get('document_language', 'unknown')
                    source = sample.metadata.get('source', 'unknown')
                    logger.info(f"   Sample result from {source} ({doc_lang}): {sample.page_content[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Error testing {lang} search: {e}")
                return False
        
        logger.info("✅ Index verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Index verification failed: {e}")
        return False

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multilingual Delta Index Update for Financial Advisor Bot"
    )
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Rebuild entire index from scratch"
    )
    parser.add_argument(
        "--english-only", 
        action="store_true", 
        help="Process only English PDFs"
    )
    parser.add_argument(
        "--bangla-only", 
        action="store_true", 
        help="Process only Bangla PDFs"
    )
    parser.add_argument(
        "--verify", 
        action="store_true", 
        help="Verify index integrity after update"
    )
    parser.add_argument(
        "--backup", 
        action="store_true", 
        help="Create backup of current index"
    )
    parser.add_argument(
        "--restore", 
        action="store_true", 
        help="Restore index from backup"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.backup:
        create_backup()
        return
    
    if args.restore:
        if restore_backup():
            logger.info("✅ Index restored from backup")
        else:
            logger.error("❌ Failed to restore from backup")
        return
    
    # Validate arguments
    if args.english_only and args.bangla_only:
        logger.error("❌ Cannot specify both --english-only and --bangla-only")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Perform delta update
        multilingual_delta_update(
            force_full=args.full,
            english_only=args.english_only,
            bangla_only=args.bangla_only
        )
        
        # Verify index if requested
        if args.verify:
            if not verify_index():
                logger.error("❌ Index verification failed")
                sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
