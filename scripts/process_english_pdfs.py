import os
import shutil
import logging
import langdetect
import fitz  # PyMuPDF
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.pdf_processor import PDFProcessor
from src.utils.document_manager import DocumentManager

# Setup logging
logging.basicConfig(
    filename='logs/pdf_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnglishPDFProcessor:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.document_manager = DocumentManager()
        self.raw_dir = "data/raw"
        self.processed_dir = "data/processed"
        self.categories = ['banking', 'investment', 'loans', 'regulations', 'sme', 'taxation']
    
    def is_english(self, text: str) -> bool:
        """Check if the text is in English"""
        try:
            lang = langdetect.detect(text)
            return lang == 'en'
        except:
            return False
    
    def process_pdf(self, file_path: str, category: str) -> bool:
        """Process a single PDF file"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Check if document has changed since last processing
            if not self.document_manager.is_document_changed(file_path):
                logger.info(f"File {file_path} hasn't changed since last processing, skipping")
                return True  # Consider it successfully processed
            
            # First, check PDF quality and structure
            quality_result = self.pdf_processor.process_pdf(file_path)
            
            if quality_result['status'] == 'error':
                logger.error(f"Error processing {file_path}: {quality_result['error']}")
                return False
            
            if not quality_result['passes_threshold']:
                logger.warning(f"File {file_path} failed quality checks: {quality_result['metrics']}")
                return False
            
            # Extract text for language detection
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Check if the document is in English
            if not self.is_english(text):
                logger.warning(f"File {file_path} is not in English")
                return False
            
            # If all checks pass, copy to processed directory
            dest_dir = os.path.join(self.processed_dir, category)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
            
            # Update document registry
            self.document_manager.add_or_update_document(
                file_path=file_path,
                category=category,
                source='local'
            )
            
            logger.info(f"Successfully processed and moved {file_path} to {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return False
    
    def process_all_pdfs(self):
        """Process all PDFs in the raw directory"""
        total_files = 0
        processed_files = 0
        
        for category in self.categories:
            raw_category_dir = os.path.join(self.raw_dir, category)
            if not os.path.exists(raw_category_dir):
                continue
            
            pdf_files = [f for f in os.listdir(raw_category_dir) if f.endswith('.pdf')]
            total_files += len(pdf_files)
            
            for pdf_file in pdf_files:
                file_path = os.path.join(raw_category_dir, pdf_file)
                if self.process_pdf(file_path, category):
                    processed_files += 1
        
        return total_files, processed_files

def main():
    processor = EnglishPDFProcessor()
    total, processed = processor.process_all_pdfs()
    print(f"Successfully processed: {processed}")
    print(f"Failed/Skipped: {total - processed}")

if __name__ == "__main__":
    main()