import pytesseract
import numpy as np
import os
import re
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from pypdf import PdfReader
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF

# ✅ Explicitly set Tesseract path (important for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- CONFIGURATION ---
INPUT_FOLDER = "unsant_data"
OUTPUT_FOLDER = "data"
OCR_LANG = "ben+eng"       # Enables mixed Bangla + English OCR
OCR_THRESHOLD = 0.5        # If >50% image coverage, trigger OCR
MIN_TEXT_THRESHOLD = 50    # Minimum characters to consider page has sufficient text

# --- Extract text using PyPDF ---
def extract_text_from_pdf(pdf_path):
    """Extract text using PyPDF; return empty if no text found."""
    try:
        reader = PdfReader(pdf_path)
        extracted = "".join([page.extract_text() or "" for page in reader.pages])
        return extracted.strip()
    except Exception as e:
        print(f"[WARNING] Failed to extract text from {pdf_path}: {e}")
        return ""

# --- Enhanced Preprocess Image for OCR ---
def preprocess_image(image):
    """Enhance image quality for OCR using advanced OpenCV techniques."""
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply median filter to reduce noise
    img = cv2.medianBlur(img, 3)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Resize image for better OCR (2x upscale)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Apply threshold to get binary image (black and white)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return img

# --- Enhanced OCR Decision Logic for Website PDFs ---
def needs_ocr(page):
    """Decide if a page requires OCR by checking text quality and content."""
    try:
        # Get extractable text
        extractable_text = page.get_text().strip()
        
        # For website PDFs: Check if we have substantial text content
        # Many website PDFs have minimal extractable text but lots of visual content
        if len(extractable_text) >= MIN_TEXT_THRESHOLD:
            # Check if the text seems meaningful (not just headers/footers/metadata)
            words = extractable_text.split()
            if len(words) >= 10:  # At least 10 words suggests real content
                return False  # text is sufficient, no OCR needed
        
        # If minimal text or suspicious content, check visual complexity
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        
        # Check for non-white content (text, images, graphics)
        non_white = np.sum(img_array != 255)
        coverage = non_white / img_array.size
        
        # For website PDFs, be much more aggressive with OCR
        # Even minimal visual content might contain important text
        if len(extractable_text) < MIN_TEXT_THRESHOLD:
            # If little/no extractable text, use very low threshold
            if coverage >= 0.01:  # Even 1% coverage worth trying OCR
                return True
        
        # For pages with some text, use moderate thresholds
        if coverage >= 0.05:  # 5% coverage suggests content worth OCR'ing
            return True
            
        # Fallback to original threshold for dense content
        return coverage >= OCR_THRESHOLD
        
    except Exception as e:
        print(f"[WARNING] Error checking OCR need: {e}")
        return True  # be conservative

# --- Perform OCR on Image Pages with Structure Detection ---
def ocr_page(page):
    """Run Tesseract OCR on a page image with structure preservation."""
    pix = page.get_pixmap(dpi=350)  # Higher DPI for better recognition
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    processed = preprocess_image(img)
    
    # Try table-aware OCR with structure preservation
    config_table = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    text_with_structure = pytesseract.image_to_string(processed, lang=OCR_LANG, config=config_table)
    
    # Get TSV output for table detection
    try:
        tsv_data = pytesseract.image_to_data(processed, lang=OCR_LANG, config=config_table, output_type=pytesseract.Output.DICT)
        return {"text": text_with_structure, "tsv_data": tsv_data}
    except:
        return {"text": text_with_structure, "tsv_data": None}

# --- Structure-Aware Text Processing ---
def escape_xml(text):
    """Escape special XML/HTML characters for ReportLab."""
    if not text:
        return ""
    
    # Replace problematic characters
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#39;")
    
    return text

def detect_table_structure(tsv_data):
    """Detect table structures from OCR TSV data."""
    if not tsv_data or 'text' not in tsv_data:
        return []
    
    tables = []
    current_table = []
    current_row = []
    last_top = None
    
    for i, text in enumerate(tsv_data['text']):
        if not text.strip():
            continue
            
        left = tsv_data['left'][i]
        top = tsv_data['top'][i]
        width = tsv_data['width'][i]
        height = tsv_data['height'][i]
        
        # New row detection (significant change in top position)
        if last_top is not None and abs(top - last_top) > height * 0.5:
            if current_row:
                current_table.append(current_row)
                current_row = []
        
        current_row.append({'text': text.strip(), 'left': left, 'top': top})
        last_top = top
    
    if current_row:
        current_table.append(current_row)
    if current_table and len(current_table) > 1:  # Only consider as table if multiple rows
        tables.append(current_table)
    
    return tables

def preserve_structure_clean_text(text, tsv_data=None):
    """Clean text while preserving important structural elements."""
    if not text:
        return ""
    
    # Detect tables if TSV data is available
    tables = detect_table_structure(tsv_data) if tsv_data else []
    
    # Only remove clearly problematic elements, preserve structure
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)        # URLs
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", text)  # emails
    text = re.sub(r"\b\+?\d[\d\-\s]{7,}\d\b", " ", text)      # phones
    
    # Clean problematic characters that can cause PDF parsing issues
    text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\&@#\$%\+\=\*\n\t]", " ", text)  # Keep only safe characters
    text = re.sub(r"<[^>]*>", " ", text)                       # Remove HTML-like tags
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)              # Remove HTML entities
    
    # Preserve paragraph breaks and spacing
    text = re.sub(r"\n{3,}", "\n\n", text)                    # Limit excessive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)                     # Normalize spaces but preserve single spaces
    
    # Preserve common header/footer patterns that might be meaningful
    # Only remove page numbers that are clearly standalone
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", text, flags=re.MULTILINE)
    
    return text.strip()

def detect_headers_footers(text_lines):
    """Identify potential headers and footers based on position and repetition."""
    headers = []
    footers = []
    
    if len(text_lines) < 5:
        return headers, footers
    
    # Check first few lines for headers
    for i in range(min(3, len(text_lines))):
        line = text_lines[i].strip()
        if line and len(line) < 100:  # Reasonable header length
            headers.append((i, line))
    
    # Check last few lines for footers
    for i in range(max(0, len(text_lines) - 3), len(text_lines)):
        line = text_lines[i].strip()
        if line and len(line) < 100:  # Reasonable footer length
            footers.append((i, line))
    
    return headers, footers

# --- Clean Text ---
def clean_text(text):
    """Remove headers, footers, and extra spaces."""
    text = re.sub(r"\bPage\s+\d+(\s*/\s*\d+)?\b", " ", text)  # "Page 3 / 12"
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)        # URLs
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", text)  # emails
    text = re.sub(r"\b\+?\d[\d\-\s]{7,}\d\b", " ", text)      # phones
    text = re.sub(r"\n{2,}", "\n", text)                      # Normalize newlines
    text = re.sub(r"[ \t]+", " ", text)                       # Normalize spaces
    return text.strip()

def _remove_repeated_boilerplate(text: str) -> str:
    """Remove only clearly repetitive boilerplate while preserving meaningful structure."""
    lines = [l.strip() for l in text.splitlines()]
    freq = {}
    
    # Count frequency of short lines
    for l in lines:
        if 2 <= len(l) <= 60:  # Reduced upper limit to be more conservative
            freq[l] = freq.get(l, 0) + 1
    
    # Higher threshold to preserve more content
    threshold = max(5, len(lines) // 30)  # More conservative threshold
    
    # Filter out only very repetitive elements
    filtered = []
    for l in lines:
        # Keep line if:
        # 1. It's not in the repetitive range, OR
        # 2. It's not repeated above threshold, OR
        # 3. It looks like a header/title (title case, shorter), OR
        # 4. It contains numbers (might be important data)
        if (not (2 <= len(l) <= 60) or 
            freq.get(l, 0) < threshold or
            (l.istitle() and len(l) < 40) or
            re.search(r'\d', l)):
            filtered.append(l)
    
    return "\n".join(filtered)

def detect_table_like_structure(line):
    """Detect if a line looks like part of a table structure."""
    if not line.strip():
        return False
    
    # Look for multiple columns separated by spaces
    parts = re.split(r'\s{2,}', line.strip())
    if len(parts) >= 2:
        return True
    
    # Look for common table indicators
    if re.search(r'\|', line) or re.search(r'\t', line):
        return True
    
    # Look for aligned numbers/data
    if re.search(r'\b\d+\.?\d*\s+\d+\.?\d*\b', line):
        return True
    
    return False

# --- Create Structure-Aware PDF ---
def create_sanitized_pdf(text, output_path):
    """Rebuild sanitized PDF preserving structure like headers, tables, and formatting."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            story.append(Spacer(1, 6))  # Small space for empty lines
            i += 1
            continue
        
        # Detect potential table (lines with multiple columns indicated by spacing)
        if detect_table_like_structure(line):
            table_lines = []
            j = i
            # Collect consecutive table-like lines
            while j < len(lines) and lines[j].strip() and detect_table_like_structure(lines[j]):
                table_lines.append(lines[j].strip())
                j += 1
            
            if len(table_lines) > 1:  # Only create table if multiple rows
                table_data = []
                for table_line in table_lines:
                    # Split on multiple spaces (common table separator)
                    cells = re.split(r'\s{2,}', table_line)
                    if len(cells) > 1:
                        table_data.append(cells)
                
                if table_data:
                    # Create table with basic styling
                    try:
                        # Clean table data to prevent parsing errors
                        cleaned_table_data = []
                        for row in table_data:
                            cleaned_row = []
                            for cell in row:
                                # Clean cell content
                                cleaned_cell = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\&@#\$%\+\=\*]", " ", str(cell))
                                cleaned_cell = re.sub(r"<[^>]*>", " ", cleaned_cell)
                                cleaned_row.append(cleaned_cell.strip())
                            cleaned_table_data.append(cleaned_row)
                        
                        table = Table(cleaned_table_data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 9),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 12))
                        i = j
                        continue
                    except Exception as e:
                        print(f"[WARNING] Failed to create table: {e}")
                        # If table creation fails, treat as regular text
                        pass
        
        # Detect headers (short lines, potentially in title case)
        if len(line) < 80 and (line.isupper() or line.istitle()) and not line.endswith('.'):
            try:
                # Escape HTML-like characters and create paragraph
                escaped_line = escape_xml(line)
                para = Paragraph(escaped_line, styles['Heading2'])
                story.append(para)
                story.append(Spacer(1, 6))
            except Exception as e:
                print(f"[WARNING] Failed to create header paragraph: {e}")
                # Fallback to simple text
                try:
                    clean_line = re.sub(r'[<>&"\']', '', line)  # Remove problematic chars
                    para = Paragraph(clean_line, styles['Heading2'])
                    story.append(para)
                    story.append(Spacer(1, 6))
                except:
                    # Last resort - skip problematic line
                    print(f"[WARNING] Skipping problematic header line: {line[:50]}...")
                    story.append(Spacer(1, 6))
        else:
            # Regular paragraph
            try:
                # Escape HTML-like characters and create paragraph
                escaped_line = escape_xml(line)
                para = Paragraph(escaped_line, styles['Normal'])
                story.append(para)
                story.append(Spacer(1, 3))
            except Exception as e:
                print(f"[WARNING] Failed to create paragraph: {e}")
                # Fallback to simple text
                try:
                    clean_line = re.sub(r'[<>&"\']', '', line)  # Remove problematic chars
                    para = Paragraph(clean_line, styles['Normal'])
                    story.append(para)
                    story.append(Spacer(1, 3))
                except:
                    # Last resort - skip problematic line
                    print(f"[WARNING] Skipping problematic line: {line[:50]}...")
                    story.append(Spacer(1, 3))
        
        i += 1
    
    doc.build(story)

# --- Enhanced Main Process for Website PDFs ---
def sanitize_pdfs(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    print(f"[INFO] Starting sanitization. Looking for PDFs in {input_folder}...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".pdf"):
            print(f"[INFO] Skipping non-PDF: {filename}")
            continue

        input_pdf = os.path.join(input_folder, filename)
        output_pdf = os.path.join(output_folder, filename)
        print(f"[INFO] Processing: {filename}")

        # Initial text extraction
        final_text = extract_text_from_pdf(input_pdf)
        initial_text_length = len(final_text)
        
        doc = fitz.open(input_pdf)
        pages_with_ocr = 0
        
        for i in range(doc.page_count):
            page = doc.load_page(i)
            
            # Enhanced logic: Always check if OCR is needed
            if needs_ocr(page):
                print(f"[INFO] OCR triggered on page {i+1} of {filename}")
                pages_with_ocr += 1
                
                ocr_result = ocr_page(page)
                if isinstance(ocr_result, dict):
                    ocr_text = ocr_result["text"]
                    tsv_data = ocr_result["tsv_data"]
                else:
                    ocr_text = ocr_result
                    tsv_data = None
                
                if ocr_text and ocr_text.strip():
                    cleaned_ocr = preserve_structure_clean_text(ocr_text, tsv_data)
                    final_text += "\n" + cleaned_ocr

        # Report OCR usage
        print(f"[INFO] OCR applied to {pages_with_ocr}/{doc.page_count} pages")
        print(f"[INFO] Text extracted: {initial_text_length} chars initial + {len(final_text) - initial_text_length} chars from OCR")
        
        # Process final text
        final_text = _remove_repeated_boilerplate(preserve_structure_clean_text(final_text))
        
        # Only create PDF if we have substantial content
        if len(final_text.strip()) < 10:
            print(f"[WARNING] Very little text extracted from {filename} - check if it's a pure image PDF")
        
        create_sanitized_pdf(final_text, output_pdf)
        print(f"[INFO] Sanitized PDF saved at: {output_pdf}")

    print("[DONE] All PDFs processed successfully.")

# --- Run ---
if __name__ == "__main__":
    sanitize_pdfs()