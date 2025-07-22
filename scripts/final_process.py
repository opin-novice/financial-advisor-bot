#!/usr/bin/env python3
"""
Final process script for the complete automated PDF processing workflow
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_processed_folder import clean_processed_folder
from process_english_pdfs import EnglishPDFProcessor
from build_faiss_index import process_and_index_documents

# Setup logging
logging.basicConfig(
    filename='logs/workflow.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_workflow():
    try:
        print("\n1. Cleaning processed folder...")
        clean_processed_folder()
        
        print("\n2. Processing English PDFs...")
        processor = EnglishPDFProcessor()
        total_found, processed = processor.process_all_pdfs()
        print(f"   Processed {processed}/{total_found} PDFs")
        
        print("\n3. Building FAISS index...")
        process_and_index_documents()
        
        print("\n✅ Workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    run_workflow()