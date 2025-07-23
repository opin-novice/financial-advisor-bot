import pytesseract
import numpy as np
import os
import re
import torch
import cv2  # OpenCV for image processing
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pypdf import PdfReader
from PIL import Image
import fitz  # PyMuPDF

# Define function to extract text from PDF using PyPDF (avoiding PyMuPDF)
def extract_text_from_pdf(pdf_path):
    """
    Extracts text directly from the PDF (non-image text).
    Uses PyPDF (pypdf library) for this.
    """
    print(f"Extracting text from: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()  # Extracts raw text from each page
    return text

# Preprocess image for OCR (using pytesseract)
def preprocess_image(image):
    """
    Preprocesses the image using OpenCV for better OCR accuracy.
    """
    print("Preprocessing image for OCR...")
    # Convert image to grayscale and then apply threshold
    gray = np.array(image.convert("L"))
    _, thresh_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh_image

# Calculate image coverage on a page
def calculate_image_coverage(pdf_path, page_number, threshold=0.6):
    """
    Calculates the percentage of image coverage on a page.
    If more than the threshold percentage is image, OCR is triggered.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap()  # Convert page to a pixmap (image)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL Image

    # Count non-white pixels (indicating an image)
    img_array = np.array(img)
    non_white_pixels = np.sum(img_array != 255)  # Assuming white background
    total_pixels = img_array.size
    image_coverage = non_white_pixels / total_pixels

    print(f"Page {page_number + 1} image coverage: {image_coverage * 100:.2f}%")
    return image_coverage >= threshold

# OCR and text sanitization from PDF using PyMuPDF
def extract_and_process_text_from_pdf(pdf_path):
    """
    Extracts images from the PDF using PyMuPDF, performs OCR with Tesseract, preprocesses with OpenCV, 
    and uses a HuggingFace model to process the text. OCR triggers only when image coverage is above the threshold.
    """
    # First, extract text directly from the PDF using PyPDF
    print(f"Extracting and processing text from PDF: {pdf_path}")
    direct_text = extract_text_from_pdf(pdf_path)

    # Open the PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    sanitized_text = direct_text  # Start with the text extracted directly from the PDF

    print(f"Total pages found in {pdf_path}: {doc.page_count}")

    for i in range(doc.page_count):
        print(f"Processing page {i + 1}")

        # If the page has more than 60% image coverage, perform OCR
        if calculate_image_coverage(pdf_path, i, threshold=0.6):
            # Extract page as an image using PyMuPDF
            page = doc.load_page(i)
            pix = page.get_pixmap()  # Convert page to a pixmap (image)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL Image

            # Preprocess the image (convert to numpy array and apply threshold)
            preprocessed_image = preprocess_image(img)

            # Perform OCR using pytesseract
            page_text = pytesseract.image_to_string(preprocessed_image)

            # Remove unwanted elements like headers, footers, logos using regex
            page_text = remove_unwanted_elements(page_text)

            # Add the extracted text to the sanitized text
            sanitized_text += page_text + "\n"
        else:
            print(f"Skipping OCR for page {i + 1} due to low image coverage.")

    return sanitized_text

def remove_unwanted_elements(text):
    """
    Removes common unwanted elements like footers, headers, and logos from the extracted text.
    You can further expand this with more specific regex or AI-based techniques.
    """
    print("Removing unwanted elements from text...")
    # Example: Remove page numbers, common footer/header patterns
    text = re.sub(r"(Page \d+|\d+/\d+|\d+)", "", text)  # Remove page numbers
    text = re.sub(r"^[A-Za-z0-9]+.*$", "", text, flags=re.MULTILINE)  # Remove lines with certain header/footer patterns
    text = re.sub(r"logo_image", "", text)  # Remove logo-related text (example)
    return text

def create_sanitized_pdf(text, output_path):
    """
    Creates a sanitized PDF from the cleaned and structured text.
    """
    print(f"Creating sanitized PDF: {output_path}")
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    x = 50
    y = height - 50
    c.setFont("Helvetica", 10)

    for line in text.split("\n"):
        if y <= 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50

        c.drawString(x, y, line)
        y -= 12

    c.save()

def sanitize_pdfs_in_folder(input_folder, output_folder):
    """
    Process all PDFs in the unsant_data folder, sanitizes them, and saves them to the data folder.
    """
    print(f"Starting sanitization process. Looking for PDFs in {input_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            print(f"Found PDF: {filename}")  # Debug print to confirm processing
            input_pdf_path = os.path.join(input_folder, filename)
            output_pdf_path = os.path.join(output_folder, filename)

            print(f"Processing: {filename}")

            # Extract and sanitize text from the PDF images
            sanitized_text = extract_and_process_text_from_pdf(input_pdf_path)

            # Create a new sanitized PDF with the processed text
            create_sanitized_pdf(sanitized_text, output_pdf_path)

            print(f"Sanitized PDF saved: {output_pdf_path}")
        else:
            print(f"Skipping non-PDF file: {filename}")  # Debug for non-PDF files

# Example usage
input_folder = "unsant_data"  # Folder with unsanitized PDFs
output_folder = "data"        # Folder to save sanitized PDFs
sanitize_pdfs_in_folder(input_folder, output_folder)

