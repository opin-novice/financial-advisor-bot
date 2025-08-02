#!/usr/bin/env python3
"""
delta_index.py
--------------
Incremental / full index update.

Commands
--------
# nightly cron
python delta_index.py

# force full rebuild
python delta_index.py --full
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF
import torch

# ---------- config ----------
PDF_DIR          = Path("data")
STATE_FILE       = PDF_DIR / ".delta_state.json"
LIVE_INDEX_DIR   = Path("faiss_index")
TMP_INDEX_DIR    = Path("faiss_index_tmp")
EMBEDDING_MODEL  = "./ft_bge"         # or "BAAI/bge-base-en-v1.5"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
# -----------------------------

def file_hash(path: Path) -> str:
    """SHA-256 of file content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def extract_and_clean(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    # light cleaning (same as docadd.py)
    import re
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bpage\s*\d+\b", "", text, flags=re.I)
    return text.strip()

def build_index_for_files(files: list[Path]) -> FAISS:
    docs = []
    for file in files:
        text = extract_and_clean(file)
        # record page numbers if you want later
        doc = Document(page_content=text, metadata={"source": file.name})
        docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
    )
    return FAISS.from_documents(chunks, embeddings)

def delta_update(force_full: bool = False):
    state = load_state()
    pdf_files = [p for p in PDF_DIR.glob("*.pdf") if not p.name.startswith(".")]

    # compute current fingerprints
    current = {str(p.relative_to(PDF_DIR)): {"mtime": p.stat().st_mtime, "sha": file_hash(p)} for p in pdf_files}

    # files to (re)process
    if force_full:
        to_process = pdf_files
    else:
        to_process = [
            PDF_DIR / name
            for name, meta in current.items()
            if state.get(name) != meta
        ]

    if not to_process:
        print("[delta] No new/changed PDFs. Exiting.")
        return

    print(f"[delta] Processing {len(to_process)} PDF(s)…")

    # build index in a shadow dir
    if TMP_INDEX_DIR.exists():
        shutil.rmtree(TMP_INDEX_DIR)
    new_index = build_index_for_files(to_process)

    # if delta: merge with existing
    if not force_full and LIVE_INDEX_DIR.exists():
        old_index = FAISS.load_local(
    str(LIVE_INDEX_DIR),
    embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
    allow_dangerous_deserialization=True
)
        old_index.merge_from(new_index)
        old_index.save_local(str(TMP_INDEX_DIR))
    else:
        new_index.save_local(str(TMP_INDEX_DIR))

    # atomic swap
    if LIVE_INDEX_DIR.exists():
        shutil.rmtree(LIVE_INDEX_DIR)
    TMP_INDEX_DIR.rename(LIVE_INDEX_DIR)

    # update state
    state.update({str(p.relative_to(PDF_DIR)): {"mtime": p.stat().st_mtime, "sha": file_hash(p)} for p in to_process})
    save_state(state)
    print("[delta] Index updated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Rebuild everything")
    args = parser.parse_args()
    delta_update(force_full=args.full)