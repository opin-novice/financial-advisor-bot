import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.pdf_processor import PDFProcessor
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pdf_quality_checks():
    """Test PDF quality assessment functionality"""
    pdf_processor = PDFProcessor()
    
    # Test with sample PDFs
    test_files = [
        "data/got.pdf",  # Assuming this is a good quality PDF
        # Add more test files as needed
    ]
    
    for file_path in test_files:
        logger.info(f"Testing PDF quality for: {file_path}")
        start_time = time.time()
        result = pdf_processor.process_pdf(file_path)
        processing_time = time.time() - start_time
        
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Quality metrics: {result['metrics']}")
        logger.info(f"Passes threshold: {result['passes_threshold']}")
        logger.info("-" * 50)

if __name__ == "__main__":
    logger.info("Starting PDF processor tests")
    test_pdf_quality_checks()
    logger.info("PDF processor tests completed")