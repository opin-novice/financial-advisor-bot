import os
import fitz  # PyMuPDF
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import re

# Note: Run setup_nltk.py first to download required NLTK data

# --- Configuration ---
PDF_DIR = "data"              # Folder containing your PDF files
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Using mpnet for better quality

# ✅ Semantic Chunking Settings
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # For sentence similarity (better quality)
MAX_CHUNK_SIZE = 1200          # Maximum tokens per chunk
MIN_CHUNK_SIZE = 200           # Minimum tokens per chunk
SIMILARITY_THRESHOLD = 0.7     # Threshold for grouping sentences
OVERLAP_SENTENCES = 1          # Number of sentences to overlap between chunks

# --- Load PDF file paths ---
PDF_PATHS = [
    os.path.join(PDF_DIR, f)
    for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
]

class SemanticChunker:
    """
    Semantic chunking using sentence embeddings and similarity-based grouping
    """
    
    def __init__(self, 
                 sentence_model_name=SENTENCE_EMBEDDING_MODEL,
                 max_chunk_size=MAX_CHUNK_SIZE,
                 min_chunk_size=MIN_CHUNK_SIZE,
                 similarity_threshold=SIMILARITY_THRESHOLD,
                 overlap_sentences=OVERLAP_SENTENCES):
        
        self.sentence_model = SentenceTransformer(sentence_model_name)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.overlap_sentences = overlap_sentences
        
        print(f"[INFO] Semantic chunker initialized with model: {sentence_model_name}")
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with sentence splitting
        text = re.sub(r'[^\w\s\.\!\?\;\:\,\-\(\)\[\]\"\']+', ' ', text)
        return text.strip()
    
    def split_into_sentences(self, text):
        """Split text into sentences using NLTK"""
        cleaned_text = self.clean_text(text)
        sentences = nltk.sent_tokenize(cleaned_text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def estimate_tokens(self, text):
        """Rough token estimation (1 token ≈ 4 characters for English text)"""
        return len(text) // 4
    
    def calculate_similarity_breakpoints(self, sentences):
        """Calculate similarity between consecutive sentences and find breakpoints"""
        if len(sentences) < 2:
            return []
        
        # Generate embeddings for all sentences
        print(f"[INFO] Generating embeddings for {len(sentences)} sentences...")
        embeddings = self.sentence_model.encode(sentences)
        
        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        # Find breakpoints where similarity drops below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)  # +1 because we want to break after sentence i
        
        return breakpoints
    
    def create_chunks_with_overlap(self, sentences, breakpoints):
        """Create chunks from sentences using breakpoints and add overlap"""
        if not sentences:
            return []
        
        # Add start and end points
        split_points = [0] + breakpoints + [len(sentences)]
        split_points = sorted(list(set(split_points)))  # Remove duplicates and sort
        
        chunks = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            
            # Add overlap from previous chunk
            if i > 0 and self.overlap_sentences > 0:
                overlap_start = max(0, start_idx - self.overlap_sentences)
                chunk_sentences = sentences[overlap_start:end_idx]
            else:
                chunk_sentences = sentences[start_idx:end_idx]
            
            chunk_text = ' '.join(chunk_sentences)
            
            # Check chunk size constraints
            token_count = self.estimate_tokens(chunk_text)
            
            # If chunk is too large, split it further
            if token_count > self.max_chunk_size:
                sub_chunks = self.split_large_chunk(chunk_sentences)
                chunks.extend(sub_chunks)
            # If chunk is too small, try to merge with next chunk
            elif token_count < self.min_chunk_size and i < len(split_points) - 2:
                # This will be handled in post-processing
                chunks.append(chunk_text)
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def split_large_chunk(self, sentences):
        """Split a chunk that's too large into smaller chunks"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = self.estimate_tokens(sentence)
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def post_process_chunks(self, chunks):
        """Post-process chunks to merge small ones and ensure quality"""
        if not chunks:
            return []
        
        processed_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_size = self.estimate_tokens(current_chunk)
            
            # If current chunk is too small, try to merge with next
            if (current_size < self.min_chunk_size and 
                i < len(chunks) - 1 and 
                self.estimate_tokens(chunks[i + 1]) < self.max_chunk_size):
                
                merged_chunk = current_chunk + " " + chunks[i + 1]
                merged_size = self.estimate_tokens(merged_chunk)
                
                if merged_size <= self.max_chunk_size:
                    processed_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                else:
                    processed_chunks.append(current_chunk)
                    i += 1
            else:
                processed_chunks.append(current_chunk)
                i += 1
        
        return processed_chunks
    
    def chunk_text(self, text, metadata=None):
        """Main method to perform semantic chunking"""
        if not text or len(text.strip()) < 50:
            return []
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        if len(sentences) < 2:
            return [Document(page_content=text, metadata=metadata or {})]
        
        print(f"[INFO] Processing {len(sentences)} sentences for semantic chunking...")
        
        # Calculate similarity breakpoints
        breakpoints = self.calculate_similarity_breakpoints(sentences)
        print(f"[INFO] Found {len(breakpoints)} semantic breakpoints")
        
        # Create chunks with overlap
        chunk_texts = self.create_chunks_with_overlap(sentences, breakpoints)
        
        # Post-process chunks
        chunk_texts = self.post_process_chunks(chunk_texts)
        
        # Create Document objects
        documents = []
        for i, chunk_text in enumerate(chunk_texts):
            if len(chunk_text.strip()) > 50:  # Filter very short chunks
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'chunk_id': i,
                    'chunk_method': 'semantic',
                    'token_count': self.estimate_tokens(chunk_text)
                })
                documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))
        
        return documents

# --- Extract raw text from PDFs ---
def extract_text(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    return full_text

# --- Initialize Semantic Chunker ---
print("[INFO] Initializing semantic chunker...")
semantic_chunker = SemanticChunker()

# --- Load and Split PDFs ---
print("[INFO] Starting PDF processing with semantic chunking...")
documents = []

for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        print(f"[WARNING] File not found: {pdf_path}")
        continue

    print(f"[INFO] Processing: {pdf_path}")
    raw_text = extract_text(pdf_path)

    # Semantic chunking with metadata
    chunks = semantic_chunker.chunk_text(raw_text, metadata={
        "source": os.path.basename(pdf_path),
        "full_path": pdf_path
    })

    print(f"[INFO] ✅ {len(chunks)} semantic chunks created from {os.path.basename(pdf_path)}")
    
    # Print chunk statistics
    token_counts = [chunk.metadata.get('token_count', 0) for chunk in chunks]
    if token_counts:
        print(f"[INFO] Chunk size stats - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)//len(token_counts)}")

    documents.extend(chunks)

print(f"[INFO] Total semantic chunks prepared for embedding: {len(documents)}")

# --- Embedding Setup ---
print("[INFO] Generating embeddings...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device}
)

# --- Create and Save FAISS Index ---
print("[INFO] Creating FAISS vector index...")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"[INFO] ✅ FAISS index saved at: {FAISS_INDEX_PATH}")
print("[DONE] ✅ Semantic chunking and indexing completed successfully.")
