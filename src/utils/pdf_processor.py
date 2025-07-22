import fitz  # PyMuPDF
from typing import Dict, List, Tuple
import logging
import os
import re
import hashlib
from PIL import Image
import io
import pytesseract

class PDFProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_thresholds = {
            'min_text_length': 50,    # minimum characters per page (reduced from 100)
            'min_text_density': 0.1,  # minimum ratio of text area to page area (reduced from 0.2)
            'max_image_ratio': 20.0,  # maximum ratio of image area to page area (increased from 0.7)
            'min_confidence': 0.6     # minimum OCR confidence score (reduced from 0.8)
        }
        
    def process_pdf(self, file_path: str) -> Dict:
        try:
            doc = fitz.open(file_path)
            quality_metrics = self._calculate_quality_metrics(doc)
            structure_info = self._analyze_document_structure(doc)
            ocr_needed = self._check_if_ocr_needed(doc)
            
            result = {
                'status': 'success',
                'metrics': quality_metrics,
                'structure': structure_info,
                'ocr_needed': ocr_needed,
                'passes_threshold': self._check_quality_thresholds(quality_metrics)
            }
            
            # Perform OCR if needed and requested
            if ocr_needed:
                self.logger.info(f"OCR needed for {file_path}")
                # OCR processing would be implemented here
                # This is a placeholder for actual OCR implementation
                result['ocr_performed'] = False  # Set to True when implemented
            
            doc.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_quality_metrics(self, doc) -> Dict:
        total_pages = len(doc)
        total_text = 0
        total_images = 0
        text_density = 0
        font_sizes = []
        
        for page in doc:
            text = page.get_text()
            total_text += len(text)
            images = page.get_images()
            total_images += len(images)
            
            # Calculate text density
            text_areas = page.get_text("dict")["blocks"]
            page_area = page.rect.width * page.rect.height
            text_area = sum((block['bbox'][2] - block['bbox'][0]) * (block['bbox'][3] - block['bbox'][1]) for block in text_areas if 'bbox' in block)
            if page_area > 0:
                text_density += text_area / page_area
            
            # Extract font sizes
            for block in text_areas:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                if "size" in span:
                                    font_sizes.append(span["size"])
        
        # Calculate average font size if available
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
        
        return {
            'total_pages': total_pages,
            'avg_text_per_page': total_text / total_pages if total_pages > 0 else 0,
            'text_density': text_density / total_pages if total_pages > 0 else 0,
            'image_ratio': total_images / total_pages if total_pages > 0 else 0,
            'avg_font_size': avg_font_size
        }

    def _check_quality_thresholds(self, metrics: Dict) -> bool:
        return (
            metrics['avg_text_per_page'] >= self.quality_thresholds['min_text_length'] and
            metrics['text_density'] >= self.quality_thresholds['min_text_density'] and
            metrics['image_ratio'] <= self.quality_thresholds['max_image_ratio']
        )
    
    def _analyze_document_structure(self, doc) -> Dict:
        """Analyze the document structure to identify headings, paragraphs, tables, etc."""
        headings = []
        tables = 0
        lists = 0
        
        for page_num, page in enumerate(doc):
            # Extract text with formatting information
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                # Check for headings (larger font size, bold, etc.)
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                # Heuristic for headings: larger font or bold text
                                if ("size" in span and span["size"] > 12) or \
                                   ("flags" in span and span["flags"] & 2):  # 2 is the flag for bold text
                                    text = span["text"].strip()
                                    if text and len(text) < 100:  # Headings are usually short
                                        headings.append({
                                            "text": text,
                                            "page": page_num + 1,
                                            "font_size": span.get("size", 0)
                                        })
            
            # Detect tables (simplified heuristic)
            # Look for grid-like structures or multiple columns
            page_text = page.get_text("text")
            if re.search(r'\|[-+]+\|', page_text) or re.search(r'\+[-+]+\+', page_text):
                tables += 1
            
            # Detect lists (simplified heuristic)
            # Look for bullet points or numbered lists
            if re.search(r'^\s*[•\-\*]\s', page_text, re.MULTILINE) or \
               re.search(r'^\s*\d+\.\s', page_text, re.MULTILINE):
                lists += 1
        
        return {
            "headings_count": len(headings),
            "tables_count": tables,
            "lists_count": lists,
            "top_headings": headings[:5]  # Return only the first few headings
        }
    
    def _check_if_ocr_needed(self, doc) -> bool:
        """Check if OCR is needed for the document"""
        # Heuristic: If the document has images but little text, it might need OCR
        total_text = sum(len(page.get_text()) for page in doc)
        total_images = sum(len(page.get_images()) for page in doc)
        
        if total_images > 0 and total_text / (len(doc) * 500) < 0.3:  # Less than 30% of expected text
            return True
        
        return False
    
    def extract_text_with_ocr(self, pdf_path: str, output_dir: str = None) -> Dict:
        """Extract text from PDF using OCR when necessary"""
        try:
            doc = fitz.open(pdf_path)
            result = {
                "status": "success",
                "pages": [],
                "ocr_performed": False
            }
            
            # Create output directory if specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Process each page
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                page_result = {
                    "page_num": page_num + 1,
                    "text": page_text,
                    "ocr_applied": False
                }
                
                # Check if OCR is needed for this page
                if len(page_text.strip()) < 100:  # Minimal text on page
                    # Extract image from page
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Apply OCR
                    ocr_text = pytesseract.image_to_string(img)
                    
                    if len(ocr_text.strip()) > len(page_text.strip()):
                        page_result["text"] = ocr_text
                        page_result["ocr_applied"] = True
                        result["ocr_performed"] = True
                
                result["pages"].append(page_result)
                
                # Save extracted text to file if output directory is specified
                if output_dir:
                    output_file = os.path.join(
                        output_dir, 
                        f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{page_num+1}.txt"
                    )
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(page_result["text"])
            
            doc.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting text with OCR from {pdf_path}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def generate_pdf_hash(self, pdf_path: str) -> str:
        """Generate a hash for the PDF file to use as a unique identifier"""
        try:
            with open(pdf_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            self.logger.error(f"Error generating hash for {pdf_path}: {str(e)}")
            return ""