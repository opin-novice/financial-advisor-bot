#!/usr/bin/env python3
"""
PDF Preprocessor for RAG Pipeline - Enhanced Version

Comprehensive PDF preprocessing script that:
- Detects if PDFs are text-based or scanned
- Uses appropriate extraction methods (UnstructuredPDFLoader for text-based, OCR for scanned)
- Handles bilingual content (Bangla + English) with proper language detection
- Cleans text with Unicode normalization and regex
- Performs language-aware recursive character chunking
- Outputs structured JSON files with metadata

Supports:
- Detection of text-based vs scanned PDFs
- Appropriate extraction method selection
- OCR with pytesseract using Bangla+English language model
- Unicode normalization and text cleaning
- Language-aware recursive character chunking
- JSON output with metadata
"""

import os
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import unicodedata
import re

# PDF processing
from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import UnstructuredPDFLoader

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter


def is_scanned_pdf(pdf_path: Path) -> bool:
    """
    Detect if a PDF is scanned (image-based) or text-based.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        bool: True if scanned, False if text-based
    """
    try:
        # Try to extract text using UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(str(pdf_path))
        docs = loader.load()
        
        # If we get very little text, it's likely scanned
        total_text = "".join([doc.page_content for doc in docs])
        if len(total_text.strip()) < 100:  # Arbitrary threshold
            return True
            
        # Check for common scanned PDF characteristics
        # If most characters are spaces or very few words, likely scanned
        words = total_text.split()
        if len(words) < 10:  # Very few words
            return True
            
        return False
    except Exception:
        # If any error occurs, assume it's scanned
        return True


def extract_text_from_text_pdf(pdf_path: Path) -> List[Tuple[str, int]]:
    """
    Extract text from a text-based PDF using UnstructuredPDFLoader.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples (text, page_number)
    """
    try:
        loader = UnstructuredPDFLoader(str(pdf_path))
        docs = loader.load()
        
        # Extract text with page numbers
        page_texts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            if content.strip():  # Only add non-empty pages
                page_texts.append((content, i))
                
        return page_texts
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []


def ocr_pdf(pdf_path: Path) -> List[Tuple[str, int]]:
    """
    OCR a PDF using pdf2image and pytesseract with Bangla+English support.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples (text, page_number)
    """
    try:
        # Convert PDF to images
        pages = convert_from_path(str(pdf_path), dpi=200)
        
        # OCR each page with Bangla+English support
        page_texts = []
        for i, page in enumerate(pages, 1):
            # OCR with both Bangla and English support
            text = pytesseract.image_to_string(page, lang='ben+eng')
            page_texts.append((text, i))
        
        return page_texts
    except Exception as e:
        print(f"Error OCR processing PDF: {e}")
        return []


def is_valid_chunk(text: str) -> bool:
    """
    Check if a text chunk is valid (not corrupted or garbled).
    
    Args:
        text: Text chunk to validate
        
    Returns:
        bool: True if chunk is valid, False if corrupted
    """
    if not text or not text.strip():
        return False
        
    # Check if the text is mostly non-printable or garbled characters
    printable_chars = len([c for c in text if c.isprintable() or c.isspace()])
    if printable_chars / len(text) < 0.7:  # Less than 70% printable characters
        return False
        
    # Check average word length - if too short, likely garbled
    words = text.split()
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 2:  # Average word length less than 2 characters
            return False
    
    # Check for excessive special characters
    special_chars = len(re.findall(r'[^\w\s\u0980-\u09FF]', text))
    if special_chars / len(text) > 0.3:  # More than 30% special characters
        return False
        
    # Additional financial document quality checks
    # Check minimum word count for meaningful content
    if len(words) < 10:  # Too few words for a meaningful financial document chunk
        return False
        
    # Check for complete sentences (at least 2 periods)
    if text.count('.') < 2:
        return False
        
    # Check for form fields or templates (common in financial documents)
    form_field_patterns = [
        r':\s*\.{3,}',  # ": ..."
        r':\s*_{3,}',   # ": ___"
        r':\s*\d+\.\s*$',  # ": 5."
    ]
    for pattern in form_field_patterns:
        if re.search(pattern, text):
            return False
            
    # Enhanced: Check for financial term density
    financial_terms = [
        'loan', 'interest', 'account', 'tax', 'investment', 'deposit', 'balance',
        'rate', 'fee', 'charge', 'document', 'requirement', 'procedure', 'application',
        'regulation', 'policy', 'scheme', 'benefit', 'eligibility', 'criteria',
        'bn', 'tk', 'taka', 'dollar', 'currency', 'bank', 'finance', 'credit',
        'debit', 'transaction', 'statement', 'branch', 'manager', 'officer',
        # Bangla financial terms
        'ঋণ', 'সুদ', 'অ্যাকাউন্ট', 'ট্যাক্স', 'বিনিয়োগ', 'জমা', 'ব্যালেন্স',
        'হার', 'ফি', 'চার্জ', 'নথি', 'প্রয়োজনীয়তা', 'কার্যপ্রণালী', 'আবেদন',
        'বিধি', 'নীতি', 'প্রকল্প', 'সুবিধা', 'যোগ্যতা', 'মানদণ্ড'
    ]
    
    # Count financial terms in the text
    term_count = 0
    text_lower = text.lower()
    for term in financial_terms:
        term_count += len(re.findall(r'\b' + re.escape(term.lower()) + r'\b', text_lower))
    
    # Calculate term density (terms per 100 words)
    word_count = len(words)
    if word_count > 0:
        term_density = (term_count / word_count) * 100
        # Require at least 0.5 financial terms per 100 words
        if term_density < 0.5:
            return False
    
    return True


def clean_text(text: str) -> str:
    """
    Clean extracted text using Unicode normalization and regex.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r' *\n *', '\n', text)  # Remove spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
    
    # Remove common header/footer patterns
    # Remove page numbers at start/end of lines
    text = re.sub(r'^\s*\d+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove common artifacts
    text = re.sub(r'\.{3,}', '', text)  # Multiple dots
    text = re.sub(r'_{3,}', '', text)   # Multiple underscores
    
    # Trim and clean up
    text = text.strip()
    
    return text


def detect_language(text: str) -> str:
    """
    Detect if text is in Bangla, English, or both.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code: 'bn', 'en', or 'bn+en'
    """
    if not text:
        return 'unknown'
    
    # Check for Bangla characters
    has_bangla = bool(re.search(r'[\u0980-\u09FF]', text))
    
    # Check for English characters
    has_english = bool(re.search(r'[a-zA-Z]', text))
    
    if has_bangla and has_english:
        return 'bn+en'
    elif has_bangla:
        return 'bn'
    elif has_english:
        return 'en'
    else:
        return 'unknown'


def chunk_text(page_texts: List[Tuple[str, int]]) -> List[Dict]:
    """
    Chunk text using RecursiveCharacterTextSplitter with language-aware separators.
    
    Args:
        page_texts: List of (text, page_number) tuples
        
    Returns:
        List of chunk dictionaries
    """
    # Initialize text splitter with language-aware separators
    # Increased chunk size for better financial document context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=75,
        separators=["\n\n", "\n", "\u09e4", ".", "?", "!", " "]
    )
    
    chunks = []
    for text, page_number in page_texts:
        if not text.strip():
            continue
            
        # Clean the text
        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue
            
        # Filter out invalid chunks
        if not is_valid_chunk(cleaned_text):
            continue
            
        # Split into chunks
        text_chunks = text_splitter.split_text(cleaned_text)
        
        # Create chunk objects with metadata
        for chunk in text_chunks:
            if chunk.strip() and is_valid_chunk(chunk):  # Only add non-empty, valid chunks
                chunks.append({
                    "content": chunk.strip(),
                    "metadata": {
                        "page_number": page_number,
                        "language": detect_language(chunk)
                    }
                })
    
    return chunks


def process_single_pdf(pdf_path: Path, output_dir: Path) -> bool:
    """
    Process a single PDF file with appropriate extraction method.
    
    Args:
        pdf_path: Path to the input PDF
        output_dir: Directory to save the JSON output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Processing PDF: {pdf_path.name}")
        
        # Determine if PDF is scanned or text-based
        is_scanned = is_scanned_pdf(pdf_path)
        print(f"  -> PDF type: {'Scanned' if is_scanned else 'Text-based'}")
        
        # Extract text using appropriate method
        if is_scanned:
            page_texts = ocr_pdf(pdf_path)
        else:
            page_texts = extract_text_from_text_pdf(pdf_path)
        
        # If no text was extracted, skip
        if not page_texts:
            print(f"  -> No text extracted, skipping")
            return False
        
        # Chunk the text
        chunks = chunk_text(page_texts)
        
        # If no chunks were created, skip
        if not chunks:
            print(f"  -> No valid chunks created, skipping")
            return False
        
        # Save to JSON file with proper encoding
        output_file = output_dir / f"{pdf_path.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"  -> Saved {len(chunks)} chunks to {output_file.name}")
        return True
        
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return False


def process_pdfs(input_dir: Path, output_dir: Path) -> None:
    """
    Process all PDFs in the input directory.
    
    Args:
        input_dir: Directory containing input PDFs
        output_dir: Directory to save JSON outputs
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the input directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF with progress bar
    successful = 0
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        if process_single_pdf(pdf_file, output_dir):
            successful += 1
    
    print(f"\nProcessing complete: {successful}/{len(pdf_files)} PDFs processed successfully")


def main():
    """Main function to run the PDF preprocessing pipeline."""
    # Define directories
    project_root = Path(__file__).parent
    input_dir = project_root / "processed_pdfs"
    output_dir = project_root / "data"
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Process all PDFs
    process_pdfs(input_dir, output_dir)


if __name__ == "__main__":
    main()