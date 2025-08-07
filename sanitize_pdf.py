import pytesseract
import numpy as np
import os
import re
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pypdf import PdfReader
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF

# ✅ Explicitly set Tesseract path (important for different systems)
# For macOS with Homebrew: pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# For Windows: pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# For Linux: Usually not needed if installed via package manager

# --- CONFIGURATION ---
INPUT_FOLDER = "unsant_data"
OUTPUT_FOLDER = "data"
OCR_LANG = "ben+eng"       # Enables mixed Bangla + English OCR
OCR_THRESHOLD = 0.5        # If >50% image coverage, trigger OCR

# --- Extract text using PyPDF ---
def extract_text_from_pdf(pdf_path):
    """Extract text using PyPDF; return empty if no text found."""
    try:
        reader = PdfReader(pdf_path)
        extracted = "".join([page.extract_text() or "" for page in reader.pages])
        return extracted.strip()
    except:
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

# --- Smarter OCR Trigger ---
def needs_ocr(pdf_path, page_number):
    """Decide if a page requires OCR by checking if it's image-only."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    if page.get_text().strip():
        return False  # text is already extractable

    # Fallback: check image coverage
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_array = np.array(img)
    non_white = np.sum(img_array != 255)
    coverage = non_white / img_array.size
    return coverage >= OCR_THRESHOLD

# --- Perform OCR on Image Pages ---
def ocr_page(pdf_path, page_number):
    """Run Tesseract OCR on a page image with enhanced preprocessing."""
    page = fitz.open(pdf_path).load_page(page_number)
    pix = page.get_pixmap(dpi=350)  # Higher DPI for better recognition
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    processed = preprocess_image(img)
    config = "--oem 3 --psm 6"  # LSTM OCR engine, treat page as a text block
    return pytesseract.image_to_string(processed, lang=OCR_LANG, config=config)

# --- Clean Text ---
def clean_text(text):
    """Remove headers, footers, and extra spaces."""
    text = re.sub(r"\bPage\s+\d+\b", "", text)  # Remove page numbers
    text = re.sub(r"\n{2,}", "\n", text)        # Normalize newlines
    return text.strip()

# --- Create Sanitized PDF ---
def create_sanitized_pdf(text, output_path):
    """Rebuild sanitized PDF from cleaned text."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    x, y = 50, height - 50
    c.setFont("Helvetica", 10)

    for line in text.split("\n"):
        if y <= 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50
        c.drawString(x, y, line)
        y -= 12
    c.save()

# --- Main Sanitization Process ---
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

        # Extract initial text with PyPDF
        final_text = extract_text_from_pdf(input_pdf)

        # Process pages for OCR where needed
        doc = fitz.open(input_pdf)
        for i in range(doc.page_count):
            if needs_ocr(input_pdf, i):
                print(f"[INFO] 🟢 OCR triggered on page {i+1} of {filename}")
                ocr_text = ocr_page(input_pdf, i)
                if ocr_text.strip():
                    final_text += "\n" + clean_text(ocr_text)

        final_text = clean_text(final_text)
        create_sanitized_pdf(final_text, output_pdf)
        print(f"[INFO] ✅ Sanitized PDF saved at: {output_pdf}")

    print("[DONE] ✅ All PDFs processed successfully.")

# --- Run ---
if __name__ == "__main__":
    sanitize_pdfs()