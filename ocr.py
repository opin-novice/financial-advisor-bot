# ocr.py
"""
OCR utilities for images (PNG/JPG/JPEG/WebP/TIFF/etc.) and PDFs.

Features:
- Bangla + English OCR by default (override via OCR_LANGS env)
- EXIF auto-orientation for camera shots
- OSD-based deskew (Tesseract)
- Light preprocessing (denoise, contrast, binarize)
- Multi-column heuristic (auto-choose PSM)
- PDF text-layer extraction with per-page OCR fallback
- Dynamic DPI per page for performance

Usage:
    from ocr import (
        ocr_from_image_bytes,
        ocr_from_image_path,
        extract_text_from_pdf,
        auto_ocr_from_bytes,
        auto_ocr_from_path,
    )
"""

from __future__ import annotations

import io
import os
import re
import sys
from typing import Optional, List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Configure Tesseract path on Windows (adjust if installed elsewhere)
if sys.platform == "win32":
    default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_path):
        pytesseract.pytesseract.tesseract_cmd = default_path

# Default OCR languages: Bangla + English (override via env OCR_LANGS="eng", etc.)
OCR_LANGS = os.getenv("OCR_LANGS", "ben+eng")

# Minimum characters on a page's text layer to consider it "has text"
MIN_TEXT_CHARS_PER_PAGE = 32

# Target base DPI when rasterizing PDFs for OCR; may be downscaled per page
BASE_OCR_DPI = 300

# Recognized image file extensions for auto-detection convenience
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _exif_correct(img: Image.Image) -> Image.Image:
    """Respect EXIF orientation (rotated phone photos, etc.)."""
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Light cleanup to improve OCR quality."""
    # upscale smaller images to ~1200px width
    w, h = img.size
    if w < 1200:
        scale = 1200 / max(1, w)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # grayscale -> slight denoise -> contrast boost -> simple binarization
    img = ImageOps.grayscale(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = img.point(lambda p: 255 if p > 180 else 0)
    return img


def _try_osd_rotate(img: Image.Image) -> Image.Image:
    """Use Tesseract OSD to detect rotation and correct it; ignore failures."""
    try:
        osd = pytesseract.image_to_osd(img)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        if m:
            angle = int(m.group(1)) % 360
            if angle:
                return img.rotate(-angle, expand=True)
    except Exception:
        pass
    return img


def _looks_multicolumn(img: Image.Image) -> bool:
    """
    Crude multi-column heuristic using a vertical projection profile.
    If there are many near-empty vertical slices (gaps), likely multi-column.
    """
    gray = ImageOps.grayscale(img)
    w, h = gray.size
    if w == 0 or h == 0:
        return False

    # downsample for speed
    target_w = min(400, w)
    target_h = int(h * (target_w / w))
    small = gray.resize((target_w, max(target_h, 1)))

    # compute vertical histogram of "ink" pixels
    threshold = 200
    hist = [sum(small.getpixel((x, y)) < threshold for y in range(small.height))
            for x in range(small.width)]

    # count consecutive near-empty columns as gaps
    gaps = 0
    consecutive = 0
    for v in hist:
        if v < 5:
            consecutive += 1
        else:
            if consecutive >= 2:
                gaps += 1
            consecutive = 0
    if consecutive >= 2:
        gaps += 1

    return gaps > small.width * 0.02  # tweakable threshold


def _dpi_for_page(page: fitz.Page) -> int:
    """Choose a DPI per page based on page size; large pages get a lower DPI."""
    w, h = page.rect.width, page.rect.height  # points (~1/72 inch)
    longest = max(w, h)
    if longest > 1400:
        return 200
    if longest < 800:
        return 350
    return BASE_OCR_DPI


# ---------------------------------------------------------------------------
# Public OCR functions
# ---------------------------------------------------------------------------

def run_ocr(
    img: Image.Image,
    psm: int = 6,
    langs: Optional[str] = None,
    enable_multicolumn_heuristic: bool = True,
) -> str:
    """
    Run Tesseract OCR on a PIL image with sane defaults and optional multi-column heuristic.
    """
    img = _exif_correct(img)
    img = _try_osd_rotate(img)
    img = _preprocess_for_ocr(img)

    # auto PSM for suspected multi-column
    psm_final = 4 if (enable_multicolumn_heuristic and _looks_multicolumn(img)) else psm
    config = f"--oem 3 --psm {psm_final}"
    return pytesseract.image_to_string(img, lang=(langs or OCR_LANGS), config=config)


def ocr_from_image_bytes(
    image_bytes: bytes,
    psm: int = 6,
    langs: Optional[str] = None,
    enable_multicolumn_heuristic: bool = True,
) -> str:
    """OCR raw image bytes (PNG/JPG/JPEG/WebP/TIFF/BMP…)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Provided bytes are not a recognized image format.")
    return run_ocr(
        img,
        psm=psm,
        langs=langs,
        enable_multicolumn_heuristic=enable_multicolumn_heuristic,
    )


def ocr_from_image_path(
    path: str,
    psm: int = 6,
    langs: Optional[str] = None,
    enable_multicolumn_heuristic: bool = True,
) -> str:
    """OCR an image from a file path (auto-detects PNG/JPG/…)."""
    with open(path, "rb") as f:
        b = f.read()
    return ocr_from_image_bytes(
        b, psm=psm, langs=langs, enable_multicolumn_heuristic=enable_multicolumn_heuristic
    )


def extract_text_from_pdf(
    pdf_bytes: bytes,
    langs: Optional[str] = None,
    min_text_chars_page: int = MIN_TEXT_CHARS_PER_PAGE,
) -> str:
    """
    Extract text from a PDF with per-page logic:
      - Use embedded text layer if the page has enough characters.
      - Otherwise, render the page at a dynamic DPI and OCR it.
    Returns a single string, prefixing each page with "[PAGE N]".
    """
    pdf = fitz.open("pdf", pdf_bytes)
    pages_text: List[str] = []

    for i, page in enumerate(pdf, start=1):
        # 1) Try text layer for this page
        text = (page.get_text() or "").strip()
        if len(text) >= min_text_chars_page:
            pages_text.append(f"[PAGE {i}]\n{text}")
            continue

        # 2) Fallback to OCR for this page
        dpi = _dpi_for_page(page)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = run_ocr(img, psm=6, langs=langs)
        pages_text.append(f"[PAGE {i}]\n{page_text}")

    return "\n\n".join(pages_text)


def auto_ocr_from_bytes(
    file_bytes: bytes,
    file_name: Optional[str] = None,
    mime_hint: Optional[str] = None,
    langs: Optional[str] = None,
) -> str:
    """
    Auto-detect PDF vs image and extract text accordingly.
    - Checks PDF magic header first.
    - If not PDF, attempts to open as an image via PIL.
    """
    # Quick PDF signature check
    if file_bytes[:5] == b"%PDF-":
        return extract_text_from_pdf(file_bytes, langs=langs)

    # Try MIME hint / extension if provided (best-effort)
    if mime_hint and "pdf" in mime_hint.lower():
        return extract_text_from_pdf(file_bytes, langs=langs)
    if file_name:
        _, ext = os.path.splitext(file_name.lower())
        if ext in SUPPORTED_IMAGE_EXTS:
            return ocr_from_image_bytes(file_bytes, langs=langs)

    # Last resort: try to decode as image
    try:
        return ocr_from_image_bytes(file_bytes, langs=langs)
    except ValueError:
        # Not an image → assume (possibly malformed) PDF
        return extract_text_from_pdf(file_bytes, langs=langs)


def auto_ocr_from_path(
    path: str,
    langs: Optional[str] = None,
) -> str:
    """
    Auto-detect based on file path and content.
    Supports PDFs and common image formats (png/jpg/jpeg/webp/tif/tiff/bmp).
    """
    with open(path, "rb") as f:
        b = f.read()
    return auto_ocr_from_bytes(b, file_name=os.path.basename(path), langs=langs)
