# Financial Advisor Bot - Data Processing Scripts

This directory contains all the scripts needed to process PDFs and build your RAG (Retrieval-Augmented Generation) system for the Financial Advisor Bot.

## 🚀 Quick Start

**For the complete automated workflow, run:**
```bash
python scripts/run_workflow.py
```

This will:
1. Clean the processed folder
2. Detect English PDFs only
3. Sanitize and validate PDF quality
4. Move good PDFs to processed folder
5. Build fresh FAISS index for your bot

## 📁 Script Overview

### Main Workflow Scripts

#### 1. `process_and_index.py` ⭐
**Main script for processing new PDFs with quality checks and indexing**
```bash
python scripts/process_and_index.py
```
- Validates directory structure
- Processes all PDFs through quality checks
- Builds/updates FAISS index
- Provides detailed workflow summary

#### 2. `final_process.py`
**Simplified workflow runner**
```bash
python scripts/final_process.py
```
- Runs the complete workflow with minimal output
- Good for automated/batch processing

#### 3. `run_workflow.py`
**User-friendly wrapper for the complete workflow**
```bash
python scripts/run_workflow.py
```
- Most user-friendly option
- Includes helpful next-steps information

### Individual Processing Scripts

#### 4. `clean_processed_folder.py`
**Cleans all PDFs from the processed folder**
```bash
python scripts/clean_processed_folder.py
```
Use this when you want to start fresh.

#### 5. `process_english_pdfs.py`
**Processes only English PDFs with quality checks**
```bash
python scripts/process_english_pdfs.py
```
- Detects language using langdetect
- Validates PDF quality and readability
- Moves valid PDFs to processed folder

#### 6. `build_faiss_index.py`
**Builds/rebuilds FAISS index from processed PDFs**
```bash
python scripts/build_faiss_index.py
```
- Creates embeddings using HuggingFace models
- Builds searchable FAISS index
- Updates existing index incrementally

### Verification and Testing Scripts

#### 7. `verify_english_pdfs.py`
**Checks if PDFs are in English before processing**
```bash
python scripts/verify_english_pdfs.py
```
- Scans all PDFs in raw folder
- Reports language detection results
- Useful for debugging language issues

#### 8. `test_index.py`
**Tests FAISS index functionality**
```bash
python scripts/test_index.py
```
- Comprehensive testing of the search functionality
- Tests multiple categories and query types
- Provides success rate and detailed results

## 📂 Directory Structure

Your project should be organized as follows:

```
financial-advisor-bot/
├── data/
│   ├── raw/                    # Put your PDFs here
│   │   ├── banking/
│   │   ├── investment/
│   │   ├── loans/
│   │   ├── regulations/
│   │   ├── sme/
│   │   └── taxation/
│   └── processed/              # Processed PDFs (auto-created)
│       ├── banking/
│       ├── investment/
│       ├── loans/
│       ├── regulations/
│       ├── sme/
│       └── taxation/
├── faiss_index/               # Generated FAISS index files
├── logs/                      # Processing logs
└── scripts/                   # All processing scripts
```

## 🔧 Configuration

### Quality Thresholds
PDFs must meet these criteria to be processed:
- **Minimum text per page**: 50 characters
- **Text density**: At least 0.1 ratio of text area to page area
- **Maximum image ratio**: 20 images per page maximum
- **Language**: Must be detected as English

You can adjust these thresholds in `src/utils/pdf_processor.py`.

### Embedding Model
The system uses `sentence-transformers/all-mpnet-base-v2` for embeddings. You can change this in `scripts/build_faiss_index.py`.

## 📝 Logging

All scripts create detailed logs in the `logs/` directory:
- `workflow.log` - Main workflow logs
- `pdf_processing.log` - PDF processing details
- `indexing.log` - FAISS indexing logs
- `cleaning.log` - Folder cleaning operations
- `verify_english.log` - Language verification logs
- `test_index.log` - Index testing results

## 🔍 Troubleshooting

### Common Issues

**1. "fitz not found" error**
```bash
pip install PyMuPDF
```

**2. "langdetect not found" error**
```bash
pip install langdetect
```

**3. "No English PDFs found"**
- Check if your PDFs actually contain English text
- Run `python scripts/verify_english_pdfs.py` to debug
- Some scanned PDFs might need OCR

**4. "Quality checks failed"**
- PDFs might be image-heavy or have poor text extraction
- Check the logs for specific quality metrics
- Consider adjusting thresholds in `pdf_processor.py`

**5. "FAISS index not found"**
- Run the full workflow first: `python scripts/run_workflow.py`
- Make sure processed folder contains valid PDFs

### Getting Help

1. Check the logs in the `logs/` directory
2. Run `python scripts/test_index.py` to verify your setup
3. Make sure all dependencies are installed
4. Verify your PDF files are readable and in English

## 🎯 Workflow Steps Explained

### Step 1: Manual Preparation (Your Job)
- Download financial PDFs
- Sort them into appropriate category folders in `data/raw/`
- Categories: banking, investment, loans, regulations, sme, taxation

### Step 2: Automated Processing (Scripts Handle This)
- **Language Detection**: Only English PDFs are processed
- **Quality Validation**: PDFs must have sufficient readable text
- **Sanitization**: Low-quality or unreadable PDFs are filtered out
- **Organization**: Valid PDFs are moved to `data/processed/`

### Step 3: Index Building (Automatic)
- **Text Extraction**: Extract and chunk document content
- **Embeddings**: Generate vector embeddings for search
- **FAISS Index**: Build searchable index for your RAG system

### Step 4: Verification (Optional)
- **Testing**: Verify index works with sample queries
- **Validation**: Ensure good search results across categories

## 🏆 Success Metrics

A successful processing workflow should show:
- ✅ High percentage of PDFs passing quality checks
- ✅ English language detection working correctly
- ✅ FAISS index building without errors
- ✅ Test queries returning relevant results
- ✅ All categories properly represented

Your RAG system is ready when you see these success indicators!
