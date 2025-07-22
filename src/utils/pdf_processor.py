import fitz  # PyMuPDF
from typing import Dict, List
import logging

class PDFProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_thresholds = {
            'min_text_length': 100,  # minimum characters per page
            'min_text_density': 0.2,  # minimum ratio of text area to page area
            'max_image_ratio': 0.7,   # maximum ratio of image area to page area
            'min_confidence': 0.8     # minimum OCR confidence score
        }

    def process_pdf(self, file_path: str) -> Dict:
        try:
            doc = fitz.open(file_path)
            quality_metrics = self._calculate_quality_metrics(doc)
            doc.close()
            
            return {
                'status': 'success',
                'metrics': quality_metrics,
                'passes_threshold': self._check_quality_thresholds(quality_metrics)
            }
        except Exception as e:
            self.logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_quality_metrics(self, doc) -> Dict:
        total_pages = len(doc)
        total_text = 0
        total_images = 0
        text_density = 0

        for page in doc:
            text = page.get_text()
            total_text += len(text)
            images = page.get_images()
            total_images += len(images)
            
            # Calculate text density
            text_areas = page.get_text("dict")["blocks"]
            page_area = page.rect.width * page.rect.height
            text_area = sum(block["bbox"][2] * block["bbox"][3] for block in text_areas)
            text_density += text_area / page_area

        return {
            'total_pages': total_pages,
            'avg_text_per_page': total_text / total_pages,
            'text_density': text_density / total_pages,
            'image_ratio': total_images / total_pages
        }

    def _check_quality_thresholds(self, metrics: Dict) -> bool:
        return (
            metrics['avg_text_per_page'] >= self.quality_thresholds['min_text_length'] and
            metrics['text_density'] >= self.quality_thresholds['min_text_density'] and
            metrics['image_ratio'] <= self.quality_thresholds['max_image_ratio']
        )