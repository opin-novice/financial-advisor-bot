#!/usr/bin/env python3
"""
Script to verify if PDFs are in English
"""

import os
import logging
import langdetect
import fitz  # PyMuPDF
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.pdf_processor import PDFProcessor

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/verify_english.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def is_english(text: str) -> bool:
    """Check if the text is in English"""
    try:
        lang = langdetect.detect(text)
        return lang == 'en'
    except:
        return False

def verify_pdf(file_path: str, logger) -> bool:
    """Verify if the PDF is in English"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Check if the document is in English
        if is_english(text):
            logger.info(f"File {file_path} is in English")
            return True
        else:
            logger.warning(f"File {file_path} is not in English")
            return False
        
    except Exception as e:
        logger.error(f"Error verifying {file_path}: {str(e)}")
        return False

def verify_all_pdfs(directory: str, logger):
    """Verify all PDFs in the directory"""
    try:
        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(directory, pdf_file)
            verify_pdf(file_path, logger)
        
    except Exception as e:
        logger.error(f"Error in verification process: {str(e)}")

if __name__ == "__main__":
    logger = setup_logging()
    
    data_dir = "data/raw"
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            logger.info(f"Verifying PDFs in category: {category}")
            verify_all_pdfs(category_path, logger)

