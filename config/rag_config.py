# RAG System Configuration

# Embedding model settings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DEVICE = "cpu"  # Use "cpu" if GPU is not available

# LLM settings
LLM_MODEL = "llama3"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

# FAISS index settings
FAISS_INDEX_PATH = "faiss_index"

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ".", " ", ""]

# PDF quality thresholds
PDF_QUALITY = {
    'min_text_length': 100,  # minimum characters per page
    'min_text_density': 0.2,  # minimum ratio of text area to page area
    'max_image_ratio': 0.7,   # maximum ratio of image area to page area
    'min_confidence': 0.8     # minimum OCR confidence score
}

# Cache settings
CACHE_ENABLED = True
CACHE_DIR = "cache"
CACHE_TTL = 86400  # 24 hours in seconds

# Data directories
DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DOCUMENT_REGISTRY = "data/document_registry.json"

# Logging settings
LOG_FILE = "logs/financial_advisor_bot.log"
LOG_LEVEL = "INFO"