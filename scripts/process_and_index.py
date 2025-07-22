#!/usr/bin/env python3
"""
Main script for processing new PDFs with quality checks and indexing
This script handles the complete workflow:
1. Clean processed folder
2. Process English PDFs with quality checks
3. Build/rebuild FAISS index
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clean_processed_folder import clean_processed_folder
from process_english_pdfs import EnglishPDFProcessor
from build_faiss_index import process_and_index_documents

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/process_and_index.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_directory_structure():
    """Validate that required directories exist"""
    required_dirs = [
        "data/raw",
        "data/processed",
        "logs"
    ]
    
    categories = ['banking', 'investment', 'loans', 'regulations', 'sme', 'taxation']
    
    # Create raw directories if they don't exist
    for category in categories:
        required_dirs.append(f"data/raw/{category}")
        required_dirs.append(f"data/processed/{category}")
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return True

def print_workflow_header():
    """Print workflow header"""
    print("="*70)
    print("           FINANCIAL ADVISOR BOT - PDF PROCESSING WORKFLOW")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_workflow_summary(stats):
    """Print workflow summary"""
    print("\n" + "="*70)
    print("                        WORKFLOW SUMMARY")
    print("="*70)
    print(f"Total PDFs found in raw folder: {stats.get('total_found', 0)}")
    print(f"PDFs processed successfully: {stats.get('processed', 0)}")
    print(f"PDFs failed quality checks: {stats.get('failed_quality', 0)}")
    print(f"PDFs failed language checks: {stats.get('failed_language', 0)}")
    print(f"Index updated: {'Yes' if stats.get('index_updated', False) else 'No'}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

def main():
    """Main workflow function"""
    logger = setup_logging()
    
    try:
        print_workflow_header()
        
        # Validate directory structure
        print("🔍 Validating directory structure...")
        validate_directory_structure()
        print("✅ Directory structure validated")
        
        # Step 1: Clean processed folder
        print("\n📁 Step 1: Cleaning processed folder...")
        clean_processed_folder()
        print("✅ Processed folder cleaned")
        
        # Step 2: Process English PDFs
        print("\n🔍 Step 2: Processing and validating PDFs...")
        processor = EnglishPDFProcessor()
        total_found, processed = processor.process_all_pdfs()
        
        failed_quality = total_found - processed
        print(f"✅ PDF processing completed: {processed}/{total_found} successful")
        
        # Step 3: Build FAISS index
        print("\n📊 Step 3: Building FAISS index...")
        try:
            process_and_index_documents()
            index_updated = True
            print("✅ FAISS index updated successfully")
        except Exception as e:
            logger.error(f"Error updating FAISS index: {str(e)}")
            index_updated = False
            print(f"❌ FAISS index update failed: {str(e)}")
        
        # Print summary
        stats = {
            'total_found': total_found,
            'processed': processed,
            'failed_quality': failed_quality,
            'failed_language': 0,  # This would need to be tracked separately
            'index_updated': index_updated
        }
        
        print_workflow_summary(stats)
        
        if processed > 0 and index_updated:
            print("\n🎉 Workflow completed successfully!")
            print("Your RAG system is ready to use with the processed documents.")
        else:
            print("\n⚠️  Workflow completed with issues. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        print(f"\n❌ Workflow failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
