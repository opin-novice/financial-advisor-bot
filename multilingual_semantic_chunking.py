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
import langdetect
from langdetect import detect
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Configuration ---
PDF_DIR = "data"              # Folder containing your PDF files
BANGLA_PDF_DIR = "unsant_data"  # Folder containing Bangla PDF files
FAISS_INDEX_PATH = "faiss_index_multilingual"

# Using multilingual embedding models
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Alternative: "BAAI/bge-m3" (better multilingual support)

# ✅ Multilingual Semantic Chunking Settings
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MAX_CHUNK_SIZE = 1500          # Increased for multilingual content
MIN_CHUNK_SIZE = 250           # Increased minimum
SIMILARITY_THRESHOLD = 0.65    # Adjusted for multilingual models
OVERLAP_SENTENCES = 2          # Increased overlap for better context

# Language detection keywords
BANGLA_KEYWORDS = ['ব্যাংক', 'টাকা', 'ঋণ', 'হিসাব', 'সুদ', 'বিনিয়োগ', 'কর', 'আয়কর', 'সঞ্চয়', 'আবেদন', 'ফর্ম', 'নাম', 'ঠিকানা']
ENGLISH_KEYWORDS = ['bank', 'loan', 'account', 'interest', 'investment', 'tax', 'income', 'savings', 'application', 'form', 'name', 'address']

# --- Load PDF file paths from both directories ---
def get_pdf_paths():
    pdf_paths = []
    
    # English PDFs
    if os.path.exists(PDF_DIR):
        english_pdfs = [
            os.path.join(PDF_DIR, f)
            for f in os.listdir(PDF_DIR)
            if f.lower().endswith(".pdf")
        ]
        pdf_paths.extend(english_pdfs)
        print(f"[INFO] Found {len(english_pdfs)} English PDFs in {PDF_DIR}")
    
    # Bangla PDFs
    if os.path.exists(BANGLA_PDF_DIR):
        bangla_pdfs = [
            os.path.join(BANGLA_PDF_DIR, f)
            for f in os.listdir(BANGLA_PDF_DIR)
            if f.lower().endswith(".pdf")
        ]
        pdf_paths.extend(bangla_pdfs)
        print(f"[INFO] Found {len(bangla_pdfs)} Bangla PDFs in {BANGLA_PDF_DIR}")
    
    return pdf_paths

PDF_PATHS = get_pdf_paths()

class MultilingualLanguageDetector:
    """Enhanced language detection for Bangla and English"""
    
    def __init__(self):
        self.bangla_keywords = set(BANGLA_KEYWORDS)
        self.english_keywords = set(ENGLISH_KEYWORDS)
    
    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Bangla or English"""
        if not text or len(text.strip()) < 10:
            return 'unknown'
        
        try:
            # Use langdetect first
            detected = detect(text.lower())
            if detected == 'bn':
                return 'bangla'
            elif detected == 'en':
                return 'english'
        except:
            pass
        
        # Fallback to Unicode and keyword-based detection
        text_lower = text.lower()
        
        # Count Bangla Unicode characters
        bangla_chars = len([c for c in text if '\u0980' <= c <= '\u09FF'])
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars > 0:
            bangla_ratio = bangla_chars / total_chars
            if bangla_ratio > 0.3:  # If more than 30% Bangla characters
                return 'bangla'
        
        # Keyword-based detection
        bangla_count = sum(1 for word in self.bangla_keywords if word in text)
        english_count = sum(1 for word in self.english_keywords if word in text_lower)
        
        if bangla_count > english_count:
            return 'bangla'
        elif english_count > 0:
            return 'english'
        else:
            return 'mixed'

class MultilingualSemanticChunker:
    """
    Multilingual semantic chunking supporting both Bangla and English
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
        self.lang_detector = MultilingualLanguageDetector()
        
        print(f"[INFO] Multilingual semantic chunker initialized with model: {sentence_model_name}")
    
    def clean_text(self, text):
        """Clean and normalize multilingual text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve Bangla characters
        # Keep Bangla Unicode range: \u0980-\u09FF
        # Keep basic punctuation and alphanumeric characters
        # Using a simpler approach to avoid regex escaping issues
        import string
        allowed_chars = string.ascii_letters + string.digits + string.whitespace + '।.!?;:,-()[]"\''
        # Add Bangla Unicode characters
        bangla_chars = ''.join(chr(i) for i in range(0x0980, 0x09FF + 1))
        allowed_chars += bangla_chars
        
        # Keep only allowed characters
        cleaned_text = ''.join(c if c in allowed_chars else ' ' for c in text)
        
        # Remove excessive whitespace again
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def split_into_sentences(self, text, language='mixed'):
        """Split text into sentences with multilingual support"""
        cleaned_text = self.clean_text(text)
        
        if language == 'bangla':
            # For Bangla, use custom sentence splitting
            sentences = self._split_bangla_sentences(cleaned_text)
        else:
            # Use NLTK for English and mixed content
            try:
                sentences = nltk.sent_tokenize(cleaned_text)
            except:
                # Fallback to simple splitting
                sentences = re.split(r'[।\.!?]+', cleaned_text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        return sentences
    
    def _split_bangla_sentences(self, text):
        """Custom sentence splitting for Bangla text"""
        # Bangla sentence enders: । (dari), ?, !
        sentences = re.split(r'[।\.\!\?]+', text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def estimate_tokens(self, text):
        """Improved token estimation for multilingual text"""
        # Different estimation for different languages
        language = self.lang_detector.detect_language(text)
        
        if language == 'bangla':
            # Bangla characters are typically more information-dense
            return len(text) // 3
        else:
            # English estimation
            return len(text) // 4
    
    def calculate_similarity_breakpoints(self, sentences, language='mixed'):
        """Calculate similarity breakpoints with language awareness"""
        if len(sentences) < 2:
            return []
        
        print(f"[INFO] Generating embeddings for {len(sentences)} {language} sentences...")
        
        try:
            embeddings = self.sentence_model.encode(sentences, show_progress_bar=True)
        except Exception as e:
            print(f"[WARNING] Embedding generation failed: {e}")
            # Fallback to simple splitting
            return list(range(1, len(sentences), 3))  # Split every 3 sentences
        
        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        # Adjust threshold based on language
        threshold = self.similarity_threshold
        if language == 'bangla':
            threshold *= 0.9  # Slightly lower threshold for Bangla
        
        # Find breakpoints where similarity drops below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)
        
        return breakpoints
    
    def create_chunks_with_overlap(self, sentences, breakpoints):
        """Create chunks with overlap - same as before but with multilingual awareness"""
        if not sentences:
            return []
        
        # Add start and end points
        split_points = [0] + breakpoints + [len(sentences)]
        split_points = sorted(list(set(split_points)))
        
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
            
            if token_count > self.max_chunk_size:
                sub_chunks = self.split_large_chunk(chunk_sentences)
                chunks.extend(sub_chunks)
            elif token_count < self.min_chunk_size and i < len(split_points) - 2:
                chunks.append(chunk_text)
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def split_large_chunk(self, sentences):
        """Split large chunks with multilingual token estimation"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = self.estimate_tokens(sentence)
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def post_process_chunks(self, chunks):
        """Post-process chunks with multilingual considerations"""
        if not chunks:
            return []
        
        processed_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_size = self.estimate_tokens(current_chunk)
            
            if (current_size < self.min_chunk_size and 
                i < len(chunks) - 1 and 
                self.estimate_tokens(chunks[i + 1]) < self.max_chunk_size):
                
                merged_chunk = current_chunk + " " + chunks[i + 1]
                merged_size = self.estimate_tokens(merged_chunk)
                
                if merged_size <= self.max_chunk_size:
                    processed_chunks.append(merged_chunk)
                    i += 2
                else:
                    processed_chunks.append(current_chunk)
                    i += 1
            else:
                processed_chunks.append(current_chunk)
                i += 1
        
        return processed_chunks
    
    def chunk_text(self, text, metadata=None):
        """Main method to perform multilingual semantic chunking"""
        if not text or len(text.strip()) < 50:
            return []
        
        # Detect language
        language = self.lang_detector.detect_language(text)
        print(f"[INFO] Detected language: {language}")
        
        # Split into sentences
        sentences = self.split_into_sentences(text, language)
        if len(sentences) < 2:
            return [Document(page_content=text, metadata=metadata or {})]
        
        print(f"[INFO] Processing {len(sentences)} {language} sentences for semantic chunking...")
        
        # Calculate similarity breakpoints
        breakpoints = self.calculate_similarity_breakpoints(sentences, language)
        print(f"[INFO] Found {len(breakpoints)} semantic breakpoints")
        
        # Create chunks with overlap
        chunk_texts = self.create_chunks_with_overlap(sentences, breakpoints)
        
        # Post-process chunks
        chunk_texts = self.post_process_chunks(chunk_texts)
        
        # Create Document objects
        documents = []
        for i, chunk_text in enumerate(chunk_texts):
            if len(chunk_text.strip()) > 30:  # Filter very short chunks
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'chunk_id': i,
                    'chunk_method': 'multilingual_semantic',
                    'token_count': self.estimate_tokens(chunk_text),
                    'language': language,
                    'detected_language': language
                })
                documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))
        
        return documents

# --- Enhanced PDF text extraction ---
def extract_text_multilingual(pdf_path):
    """Extract text from PDF with better multilingual support"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num, page in enumerate(doc):
            # Try different text extraction methods
            text = page.get_text()
            
            # If text is empty or very short, try OCR-like extraction
            if len(text.strip()) < 50:
                # Get text with layout preservation
                text = page.get_text("dict")
                extracted_text = ""
                if isinstance(text, dict) and "blocks" in text:
                    for block in text["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                if "spans" in line:
                                    for span in line["spans"]:
                                        if "text" in span:
                                            extracted_text += span["text"] + " "
                text = extracted_text
            
            full_text += text + "\n"
            
        doc.close()
        return full_text
        
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
        return ""

# --- Initialize Multilingual Semantic Chunker ---
print("[INFO] Initializing multilingual semantic chunker...")
multilingual_chunker = MultilingualSemanticChunker()

# --- Load and Split PDFs ---
print("[INFO] Starting multilingual PDF processing with semantic chunking...")
documents = []
language_stats = {'bangla': 0, 'english': 0, 'mixed': 0, 'unknown': 0}

for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        print(f"[WARNING] File not found: {pdf_path}")
        continue

    print(f"[INFO] Processing: {pdf_path}")
    raw_text = extract_text_multilingual(pdf_path)
    
    if len(raw_text.strip()) < 100:
        print(f"[WARNING] Very little text extracted from {pdf_path}")
        continue

    # Detect document language
    doc_language = multilingual_chunker.lang_detector.detect_language(raw_text)
    language_stats[doc_language] += 1
    
    # Semantic chunking with enhanced metadata
    chunks = multilingual_chunker.chunk_text(raw_text, metadata={
        "source": os.path.basename(pdf_path),
        "full_path": pdf_path,
        "document_language": doc_language,
        "source_directory": "bangla" if BANGLA_PDF_DIR in pdf_path else "english"
    })

    print(f"[INFO] ✅ {len(chunks)} multilingual chunks created from {os.path.basename(pdf_path)} ({doc_language})")
    
    # Print chunk statistics
    if chunks:
        token_counts = [chunk.metadata.get('token_count', 0) for chunk in chunks]
        print(f"[INFO] Chunk size stats - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)//len(token_counts)}")

    documents.extend(chunks)

print(f"\n[INFO] Language distribution:")
for lang, count in language_stats.items():
    print(f"  - {lang}: {count} documents")

print(f"[INFO] Total multilingual chunks prepared for embedding: {len(documents)}")

# --- Multilingual Embedding Setup ---
print("[INFO] Generating multilingual embeddings...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device}
)

# --- Create and Save Multilingual FAISS Index ---
print("[INFO] Creating multilingual FAISS vector index...")
if documents:
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"[INFO] ✅ Multilingual FAISS index saved at: {FAISS_INDEX_PATH}")
    print(f"[INFO] Index contains {len(documents)} multilingual document chunks")
else:
    print("[ERROR] No documents were processed successfully!")

print("[DONE] ✅ Multilingual semantic chunking and indexing completed successfully.")

# Print final statistics
print(f"\n[INFO] Final Statistics:")
print(f"  - Total PDFs processed: {len([p for p in PDF_PATHS if os.path.exists(p)])}")
print(f"  - Total chunks created: {len(documents)}")
print(f"  - Bangla documents: {language_stats['bangla']}")
print(f"  - English documents: {language_stats['english']}")
print(f"  - Mixed language documents: {language_stats['mixed']}")
print(f"  - Embedding model: {EMBEDDING_MODEL}")
print(f"  - Index path: {FAISS_INDEX_PATH}")
