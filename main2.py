#!/usr/bin/env python3
"""
🚀 Core RAG System - Telegram Bot with Document Retrieval & Response Generation
===============================================================================

Advanced RAG system with Telegram bot interface featuring:
- Advanced Document Retrieval (FAISS + BM25 hybrid search)
- Cross-encoder re-ranking with quality filtering
- Language-aware response generation (Bangla/English)
- Telegram bot interface with rich formatting

Author: Advanced RAG System
Version: 2.1 - Telegram Bot Edition
"""

# =============================================================================
# 🔧 CONFIGURATION SECTION - EDIT THESE VALUES
# =============================================================================
# 
# ⚠️  IMPORTANT: You MUST configure the TELEGRAM_BOT_TOKEN below before running!
# 
# To get a bot token:
# 1. Message @BotFather on Telegram
# 2. Send /newbot command  
# 3. Follow instructions to create your bot
# 4. Copy the token and paste it below
#
# =============================================================================

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "7596897324:AAG3TsT18amwRF2nRBcr1JS6NdGs96Ie-D0"  # ⚠️ REPLACE WITH YOUR ACTUAL TOKEN!

# Ollama Configuration
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.1      # Controls randomness (0.0 = deterministic, 1.0 = very creative)
OLLAMA_TOP_P = 0.9             # Nucleus sampling (0.1 = focused, 1.0 = diverse)
OLLAMA_NUM_PREDICT = 800      # Maximum tokens to generate (reduced for very concise answers)
OLLAMA_REPEAT_PENALTY = 1.1  # Penalize repetition (1.0 = no penalty, >1.0 = less repetition)

# FAISS Index Path
FAISS_INDEX_PATH = "faiss_index"

# Model Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
CROSS_ENCODER_MODEL ="BAAI/bge-reranker-v2-m3"                #"BAAI/bge-reranker-v2-m3"

# Search Configuration
MAX_DOCS_FOR_RETRIEVAL = 30
MAX_DOCS_FOR_CONTEXT = 6      # Reduced for more focused, concise answers
CONTEXT_CHUNK_SIZE = 1200 
RELEVANCE_THRESHOLD = 0.09
CROSS_ENCODER_BATCH_SIZE = 32

# RRF Configuration
RRF_K = 50  # RRF constant, typically between 10-100
RRF_WEIGHTS = {
    "faiss": 0.35,     # Boost semantic search
    "bm25": 0.25,      # Keep keyword search  
    "colbert": 0.25,   # Dense retrieval
    "dirichlet": 0.2  # Reduce language model weight
}

# ColBERT Configuration  
COLBERT_MODEL = "colbert-ir/colbertv2.0"  # Memory-efficient model
COLBERT_INDEX_NAME = "financial_docs_colbert"

# Dirichlet QLM Configuration
DIRICHLET_MU = 2000  # Smoothing parameter for Dirichlet prior

# =============================================================================
# 📚 IMPORTS
# =============================================================================

import os
import re
import logging
import pickle
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Core ML/NLP Libraries
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from scipy.special import digamma
from scipy.stats import dirichlet
import math
from collections import Counter, defaultdict

# LangChain Components
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# Telegram Bot
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode

# No environment file needed - configuration is at the top of this file

# =============================================================================
# 📊 CONFIGURATION & LOGGING
# =============================================================================

@dataclass
class RAGConfig:
    """Core RAG Configuration"""
    # Core Settings
    FAISS_INDEX_PATH: str = FAISS_INDEX_PATH
    EMBEDDING_MODEL: str = EMBEDDING_MODEL
    OLLAMA_MODEL: str = OLLAMA_MODEL
    OLLAMA_BASE_URL: str = OLLAMA_BASE_URL
    OLLAMA_TEMPERATURE: float = OLLAMA_TEMPERATURE
    OLLAMA_TOP_P: float = OLLAMA_TOP_P
    OLLAMA_NUM_PREDICT: int = OLLAMA_NUM_PREDICT
    OLLAMA_REPEAT_PENALTY: float = OLLAMA_REPEAT_PENALTY
    
    # Retrieval Settings
    MAX_DOCS_FOR_RETRIEVAL: int = MAX_DOCS_FOR_RETRIEVAL
    MAX_DOCS_FOR_CONTEXT: int = MAX_DOCS_FOR_CONTEXT
    CONTEXT_CHUNK_SIZE: int = CONTEXT_CHUNK_SIZE
    
    # Cross-Encoder Settings
    CROSS_ENCODER_MODEL: str = CROSS_ENCODER_MODEL
    RELEVANCE_THRESHOLD: float = RELEVANCE_THRESHOLD
    CROSS_ENCODER_BATCH_SIZE: int = CROSS_ENCODER_BATCH_SIZE
    
    # RRF Settings
    RRF_K: int = RRF_K
    RRF_WEIGHTS: Dict[str, float] = None
    
    # ColBERT Settings
    COLBERT_MODEL: str = COLBERT_MODEL
    COLBERT_INDEX_NAME: str = COLBERT_INDEX_NAME
    
    # Dirichlet QLM Settings
    DIRICHLET_MU: float = DIRICHLET_MU

# Global configuration
config = RAGConfig()
# Initialize mutable defaults
config.RRF_WEIGHTS = RRF_WEIGHTS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import ragatouille, fallback to sentence-transformers if not available
try:
    from ragatouille import RAGPretrainedModel
    RAGATOUILLE_AVAILABLE = True
    logger.info("RAGatouille available - will use for ColBERT")
except ImportError:
    RAGATOUILLE_AVAILABLE = False
    logger.warning("RAGatouille not available - ColBERT will use sentence-transformers fallback")

# =============================================================================
# 🌍 LANGUAGE DETECTION & PROCESSING
# =============================================================================

class LanguageDetector:
    """Advanced Bangla-English language detection and processing"""
    
    def __init__(self):
        # Bangla Unicode ranges
        self.bangla_range = r'[\u0980-\u09FF]'
        
        # Common words for detection
        self.common_bangla_words = {
            'কি', 'কী', 'কে', 'কোথায়', 'কখন', 'কেন', 'কিভাবে', 'কত', 'কোন',
            'আমি', 'তুমি', 'তিনি', 'আমার', 'তোমার', 'তার', 'আমাদের', 'তোমাদের',
            'এই', 'সেই', 'ওই', 'এটা', 'সেটা', 'ওটা', 'এখানে', 'সেখানে', 'ওখানে',
            'হ্যাঁ', 'না', 'নাই', 'আছে', 'নেই', 'হবে', 'হয়েছে', 'করেছে',
            'ব্যাংক', 'টাকা', 'পয়সা','বাজেট','মূলধন','রেমিট্যান্স','আয়কর',
            'চেক', 'রিবেট', 'ভ্যাট', 'মূল্য সংযোজন কর', 'প্রভিডেন্ট ফান্ড', 'ক্ষতি',
            'মুনাফা', 'ব্যয়', 'আয়', 'বিনিয়োগকারী', 'শেয়ার', 'শেয়ার বাজার',
            'বীমা', 'পুঁজিবাজার', 'ঋণগ্রহীতা', 'ঋণদাতা', 'দায়', 'হিসাবরক্ষণ',
            'অর্থনীতি', 'মূল্যস্ফীতি', 'রিটার্ন', 'ডেবিট', 'ক্রেডিট', 'চক্রবৃদ্ধি সুদ',
            'আমানত', 'সুদহার', 'মূল্যায়ন', 'কিস্তি', 'লাভাংশ', 
            'অর্থ', 'পুঁজি', 'মূলধনী খরচ', 'ব্যালেন্স', 'লাভ', 'ক্ষতিপূরণ',
            'চলতি হিসাব', 'সঞ্চয়পত্র', 'জাতীয় সঞ্চয় স্কিম', 'বন্ড', 'ট্রেজারি বিল',
            'রাজস্ব', 'আয় বিবরণী', 'ব্যালেন্স শীট', 'ক্যাশ ফ্লো', 'হিসাব বিবরণী',
            'অডিট', 'হিসাব যাচাই', 'দেনা', 'আর্থিক বিবরণী', 'বার্ষিক প্রতিবেদন',
            'নগদ প্রবাহ', 'বিনিয়োগ ঝুঁকি', 'মূল্যহ্রাস', 'মূল্যবৃদ্ধি',
            'সুদ মওকুফ', 'ঋণ মওকুফ', 'ঋণসীমা', 'ঋণ সুবিধা',
            'পেমেন্ট', 'রসিদ', 'ভাউচার', 'ক্যাশবুক', 'জামানত',
            'নগদ', 'বেতন','পেনশন','ঋণ পরিশোধ','মূলধন',  'ডিভিডেন্ড','সম্পদ',
            'লোন', 'ঋণ', 'সুদ', 'হিসাব', 'একাউন্ট',
            'বিনিয়োগ', 'সঞ্চয়', 'জমা', 'উত্তোলন', 'লেনদেন', 'কর', 'ট্যাক্স'
        }



        self.common_english_words = {
            'bank', 'loan', 'account', 'money', 'investment', 'savings', 'deposit',
            'withdrawal', 'transaction', 'tax', 'interest', 'credit', 'debit',
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'will'
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect if text is primarily in Bangla or English
        
        Returns:
            Tuple of (language, confidence_score)
        """
        if not text or not text.strip():
            return 'english', 0.5
        
        text = text.strip().lower()
        
        # Count Bangla characters
        bangla_chars = len(re.findall(self.bangla_range, text))
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'english', 0.5
        
        bangla_char_ratio = bangla_chars / total_chars if total_chars > 0 else 0
        
        # Count words
        words = text.split()
        bangla_word_count = sum(1 for word in words if word in self.common_bangla_words)
        english_word_count = sum(1 for word in words if word in self.common_english_words)
        
        # Calculate scores
        char_score = bangla_char_ratio
        word_score = 0
        
        if bangla_word_count > 0 or english_word_count > 0:
            word_score = bangla_word_count / (bangla_word_count + english_word_count)
        
        # Combined score
        if bangla_chars > 0:
            combined_score = 0.8 * char_score + 0.2 * word_score
        else:
            combined_score = 0.3 * char_score + 0.7 * word_score
        
        # Determine language
        if combined_score > 0.3:
            language = 'bengali'
            confidence = min(0.95, 0.5 + combined_score)
        else:
            language = 'english'
            confidence = min(0.95, 0.5 + (1 - combined_score))
        
        return language, confidence
    
    def get_language_specific_prompt(self, language: str) -> PromptTemplate:
        """Get language-specific prompt template"""
        if language == 'bengali':
            template = """আপনি একজন বাংলাদেশী আর্থিক পরামর্শদাতা। প্রসঙ্গ ব্যবহার করে সংক্ষিপ্ত ও সঠিক উত্তর দিন।

প্রসঙ্গ:
{context}

প্রশ্ন: {input}

নির্দেশনা:
- বাংলায় সংক্ষিপ্ত ও স্পষ্ট উত্তর দিন
- তালিকা থাকলে বুলেট পয়েন্ট (•) ব্যবহার করুন
- শুধুমাত্র প্রসঙ্গে উল্লিখিত তথ্য ব্যবহার করুন
- অতিরিক্ত বা ভুল তথ্য যোগ করবেন না
- সরল ও বোধগম্য ভাষা ব্যবহার করুন

উত্তর:"""
        else:
            template = """You are a Bangladeshi financial advisor. Provide a concise answer using the context.

Context:
{context}

Question: {input}

Instructions:
- Give a brief, clear answer in English
- Use bullet points (•) for lists
- Only use information from the provided context
- Do not add extra or incorrect information
- Use simple and understandable language

Answer:"""
        
        return PromptTemplate(template=template, input_variables=["context", "input"])

# =============================================================================
# 🔍 RRF FUSION DOCUMENT RETRIEVAL
# =============================================================================

class RRFFusionRetriever:
    """Advanced document retrieval with RRF fusion of multiple retrievers"""
    
    def __init__(self):
        self.faiss_index = None
        self.metadata_mapping = None
        self.embedding_model = None
        self.bm25 = None
        self.bm25_corpus = None
        self.bm25_doc_ids = None
        self.colbert_model = None
        self.dirichlet_model = None
        self.document_stats = None
        
        self._init_embedding_model()
        self._init_faiss_index()
        self._init_bm25()
        self._init_colbert()
        self._init_dirichlet_model()
    
    def _init_embedding_model(self):
        """Initialize embedding model"""
        try:
            logger.info("Loading embedding model (BAAI/bge-m3)...")
            self.embedding_model = SentenceTransformer('BAAI/bge-m3', device='cpu')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    

    
    def _init_faiss_index(self):
        """Initialize FAISS index"""
        try:
            index_file = os.path.join(config.FAISS_INDEX_PATH, "index.faiss")
            metadata_file = os.path.join(config.FAISS_INDEX_PATH, "index.pkl")
            
            if not os.path.exists(index_file) or not os.path.exists(metadata_file):
                logger.warning(f"Index files not found in {config.FAISS_INDEX_PATH}")
                return
            
            logger.info(f"Loading FAISS index from: {index_file}")
            self.faiss_index = faiss.read_index(index_file)
            
            with open(metadata_file, 'rb') as f:
                metadata_tuple = pickle.load(f)
            
            # Handle LangChain FAISS format: (docstore, index_to_docstore_id)
            if isinstance(metadata_tuple, tuple) and len(metadata_tuple) == 2:
                docstore, index_to_id = metadata_tuple
                
                # Convert to our expected format
                self.metadata_mapping = {}
                for idx, doc_id in index_to_id.items():
                    if hasattr(docstore, '_dict') and doc_id in docstore._dict:
                        doc = docstore._dict[doc_id]
                        self.metadata_mapping[idx] = {
                            'content': doc.page_content,
                            'source': doc.metadata.get('source', 'Unknown'),
                            'page_number': doc.metadata.get('page', 0)
                        }
                logger.info(f"✅ FAISS index loaded with {self.faiss_index.ntotal} vectors (LangChain format)")
            else:
                # Fallback to old format
                self.metadata_mapping = metadata_tuple
                logger.info(f"✅ FAISS index loaded with {self.faiss_index.ntotal} vectors (legacy format)")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.faiss_index = None
            self.metadata_mapping = None
    
    def _init_bm25(self):
        """Initialize BM25 for keyword search"""
        try:
            if self.metadata_mapping is None:
                logger.warning("No metadata mapping for BM25 initialization")
                return
            
            corpus = []
            doc_ids = []
            
            # Handle different metadata formats
            if isinstance(self.metadata_mapping, dict):
                for doc_id, metadata in self.metadata_mapping.items():
                    # Handle different metadata structures
                    content = ""
                    if isinstance(metadata, dict):
                        content = metadata.get('content', '')
                    elif isinstance(metadata, tuple):
                        # Skip tuple format for now
                        continue
                    else:
                        # Try to get content as string
                        content = str(metadata) if metadata else ""
                    
                    if content and content.strip():
                        tokens = content.lower().split()
                        corpus.append(tokens)
                        doc_ids.append(doc_id)
            
            if corpus:
                self.bm25 = BM25Okapi(corpus)
                self.bm25_doc_ids = doc_ids
                self.bm25_corpus = corpus
                logger.info(f"✅ BM25 initialized with {len(corpus)} documents")
            else:
                logger.warning("No valid documents for BM25 initialization - skipping BM25")
                self.bm25 = None
                
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            logger.info("Continuing without BM25 - FAISS search will still work")
            self.bm25 = None
    
    def _init_colbert(self):
        """Initialize ColBERT model for dense retrieval"""
        try:
            if RAGATOUILLE_AVAILABLE:
                logger.info("Loading ColBERT model for dense retrieval using RAGatouille...")
                # Use RAGatouille for memory-efficient ColBERT
                self.colbert_model = RAGPretrainedModel.from_pretrained(config.COLBERT_MODEL)
                
                # Check if index exists, if not create it
                if self.metadata_mapping:
                    documents = []
                    for doc_id, metadata in self.metadata_mapping.items():
                        if isinstance(metadata, dict):
                            content = metadata.get('content', '')
                            if content and content.strip():
                                documents.append(content)
                    
                    if documents:
                        logger.info(f"Creating ColBERT index with {len(documents)} documents...")
                        self.colbert_model.index(
                            collection=documents,
                            index_name=config.COLBERT_INDEX_NAME,
                            max_document_length=512,  # Keep memory usage low
                            split_documents=True
                        )
                        logger.info("✅ ColBERT index created successfully")
                    else:
                        logger.warning("No documents found for ColBERT indexing")
                        self.colbert_model = None
                else:
                    logger.warning("No metadata mapping for ColBERT initialization")
                    self.colbert_model = None
            else:
                # Fallback to sentence-transformers for ColBERT-like dense retrieval
                logger.info("Loading ColBERT fallback using sentence-transformers...")
                self.colbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
                
                # Pre-compute embeddings for all documents
                if self.metadata_mapping:
                    self.colbert_documents = []
                    self.colbert_doc_ids = []
                    self.colbert_embeddings = []
                    
                    for doc_id, metadata in self.metadata_mapping.items():
                        if isinstance(metadata, dict):
                            content = metadata.get('content', '')
                            if content and content.strip():
                                # Split into sentences for ColBERT-like processing
                                sentences = content.split('.')[:3]  # Take first 3 sentences
                                combined_content = '. '.join([s.strip() for s in sentences if s.strip()])
                                
                                if combined_content:
                                    self.colbert_documents.append(combined_content)
                                    self.colbert_doc_ids.append(doc_id)
                    
                    if self.colbert_documents:
                        logger.info(f"Computing embeddings for {len(self.colbert_documents)} document fragments...")
                        self.colbert_embeddings = self.colbert_model.encode(
                            self.colbert_documents, 
                            normalize_embeddings=True,
                            show_progress_bar=True
                        )
                        logger.info("✅ ColBERT fallback initialized successfully")
                    else:
                        logger.warning("No documents found for ColBERT fallback")
                        self.colbert_model = None
                else:
                    logger.warning("No metadata mapping for ColBERT fallback initialization")
                    self.colbert_model = None
                
        except Exception as e:
            logger.warning(f"Failed to initialize ColBERT: {e}")
            logger.info("Continuing without ColBERT - other retrievers will still work")
            self.colbert_model = None
    
    def _init_dirichlet_model(self):
        """Initialize Dirichlet Query Language Model"""
        try:
            if self.metadata_mapping is None:
                logger.warning("No metadata mapping for Dirichlet model initialization")
                return
            
            # Build document statistics for Dirichlet smoothing
            self.document_stats = {}
            all_terms = Counter()
            
            for doc_id, metadata in self.metadata_mapping.items():
                if isinstance(metadata, dict):
                    content = metadata.get('content', '')
                    if content and content.strip():
                        # Tokenize and count terms
                        tokens = content.lower().split()
                        doc_terms = Counter(tokens)
                        self.document_stats[doc_id] = {
                            'terms': doc_terms,
                            'length': len(tokens),
                            'unique_terms': len(doc_terms)
                        }
                        all_terms.update(tokens)
            
            # Store collection statistics
            self.collection_stats = {
                'total_terms': sum(all_terms.values()),
                'unique_terms': len(all_terms),
                'term_frequencies': all_terms,
                'avg_doc_length': np.mean([stats['length'] for stats in self.document_stats.values()]) if self.document_stats else 0
            }
            
            logger.info(f"✅ Dirichlet QLM initialized with {len(self.document_stats)} documents")
            logger.info(f"Collection stats: {self.collection_stats['total_terms']} total terms, {self.collection_stats['unique_terms']} unique terms")
            
        except Exception as e:
            logger.error(f"Failed to initialize Dirichlet model: {e}")
            self.document_stats = None
            self.collection_stats = None
    
    def _calculate_rrf_scores(self, ranked_lists: Dict[str, List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
        """Calculate RRF scores from multiple ranked lists"""
        rrf_scores = defaultdict(float)
        
        for retriever_name, ranked_list in ranked_lists.items():
            weight = config.RRF_WEIGHTS.get(retriever_name, 1.0)
            
            for rank, (doc_id, score) in enumerate(ranked_list, 1):
                # RRF formula: weight / (k + rank)
                rrf_score = weight / (config.RRF_K + rank)
                rrf_scores[doc_id] += rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def _faiss_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform FAISS semantic search"""
        if self.faiss_index is None or self.metadata_mapping is None:
            logger.error("FAISS index not loaded")
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Search index
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Collect results as (doc_id, similarity) tuples
            # CRITICAL FIX: FAISS returns distances, not similarities!
            # Lower distance = better match, so convert to similarity
            results = []
            for i in range(min(top_k, len(indices[0]))):
                idx = indices[0][i]
                distance = scores[0][i]  # This is actually a distance!
                
                # Convert distance to similarity: similarity = max(0.0, 1.0 - distance)
                similarity = max(0.0, 1.0 - distance)
                
                # Use similarity threshold (higher = better match)
                if idx in self.metadata_mapping and similarity >= 0.3:  # Proper similarity threshold
                    results.append((idx, float(similarity)))
                    logger.debug(f"Document {idx}: distance={distance:.4f}, similarity={similarity:.4f}")
            
            logger.info(f"✅ FAISS search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform BM25 keyword search"""
        if self.bm25 is None:
            logger.warning("BM25 not initialized")
            return []
        
        try:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            top_indices = bm25_scores.argsort()[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.bm25_doc_ids):
                    doc_id = self.bm25_doc_ids[idx]
                    score = bm25_scores[idx]
                    if doc_id in self.metadata_mapping and score > 0:
                        results.append((doc_id, float(score)))
            
            logger.info(f"✅ BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _colbert_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform ColBERT dense retrieval"""
        if self.colbert_model is None:
            logger.warning("ColBERT model not initialized")
            return []
        
        try:
            if RAGATOUILLE_AVAILABLE and hasattr(self.colbert_model, 'search'):
                # Use RAGatouille ColBERT
                results = self.colbert_model.search(query, k=top_k)
                
                # Convert results to (doc_id, score) format
                colbert_results = []
                for i, result in enumerate(results):
                    # Map document content back to doc_id
                    content = result['content']
                    score = result['score']
                    
                    # Find corresponding doc_id
                    for doc_id, metadata in self.metadata_mapping.items():
                        if isinstance(metadata, dict) and metadata.get('content', '') == content:
                            colbert_results.append((doc_id, float(score)))
                            break
                
                logger.info(f"✅ ColBERT search returned {len(colbert_results)} results")
                return colbert_results
            else:
                # Use sentence-transformers fallback
                if not hasattr(self, 'colbert_embeddings') or self.colbert_embeddings is None:
                    logger.warning("ColBERT fallback embeddings not available")
                    return []
                
                # Encode query
                query_embedding = self.colbert_model.encode([query], normalize_embeddings=True)
                
                # Compute similarities
                similarities = np.dot(self.colbert_embeddings, query_embedding.T).flatten()
                
                # Get top-k results
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                colbert_results = []
                for idx in top_indices:
                    if idx < len(self.colbert_doc_ids):
                        doc_id = self.colbert_doc_ids[idx]
                        score = float(similarities[idx])
                        if score > 0.1:  # Minimum threshold
                            colbert_results.append((doc_id, score))
                
                logger.info(f"✅ ColBERT fallback search returned {len(colbert_results)} results")
                return colbert_results
            
        except Exception as e:
            logger.error(f"ColBERT search failed: {e}")
            return []
    
    def _dirichlet_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform Dirichlet Query Language Model search"""
        if self.document_stats is None or self.collection_stats is None:
            logger.warning("Dirichlet model not initialized")
            return []
        
        try:
            query_terms = query.lower().split()
            doc_scores = []
            
            for doc_id, doc_stats in self.document_stats.items():
                score = 0.0
                doc_length = doc_stats['length']
                doc_terms = doc_stats['terms']
                
                for term in query_terms:
                    # Term frequency in document
                    tf = doc_terms.get(term, 0)
                    
                    # Collection frequency
                    cf = self.collection_stats['term_frequencies'].get(term, 0)
                    
                    # Dirichlet smoothing
                    # P(term|doc) = (tf + μ * P(term|collection)) / (|doc| + μ)
                    p_term_collection = cf / self.collection_stats['total_terms'] if self.collection_stats['total_terms'] > 0 else 1e-10
                    p_term_doc = (tf + config.DIRICHLET_MU * p_term_collection) / (doc_length + config.DIRICHLET_MU)
                    
                    # Add log probability (avoiding log(0))
                    if p_term_doc > 0:
                        score += math.log(p_term_doc)
                
                doc_scores.append((doc_id, score))
            
            # Sort by score (higher is better)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"✅ Dirichlet QLM search returned {len(doc_scores[:top_k])} results")
            return doc_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Dirichlet search failed: {e}")
            return []
    
    def _rrf_fusion_search(self, query: str, top_k: int = 10) -> List[Document]:
        """Perform RRF fusion search combining multiple retrievers"""
        logger.info("🔍 Performing RRF fusion search (FAISS + BM25 + ColBERT + Dirichlet)...")
        
        # Get results from all retrievers
        ranked_lists = {}
        
        # FAISS semantic search
        faiss_results = self._faiss_search(query, top_k=top_k * 2)
        if faiss_results:
            ranked_lists['faiss'] = faiss_results
            logger.info(f"FAISS: {len(faiss_results)} results")
        
        # BM25 keyword search
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        if bm25_results:
            ranked_lists['bm25'] = bm25_results
            logger.info(f"BM25: {len(bm25_results)} results")
        
        # ColBERT dense retrieval
        colbert_results = self._colbert_search(query, top_k=top_k * 2)
        if colbert_results:
            ranked_lists['colbert'] = colbert_results
            logger.info(f"ColBERT: {len(colbert_results)} results")
        
        # Dirichlet QLM search
        dirichlet_results = self._dirichlet_search(query, top_k=top_k * 2)
        if dirichlet_results:
            ranked_lists['dirichlet'] = dirichlet_results
            logger.info(f"Dirichlet QLM: {len(dirichlet_results)} results")
        
        if not ranked_lists:
            logger.warning("No results from any retriever")
            return []
        
        # Apply RRF fusion
        rrf_results = self._calculate_rrf_scores(ranked_lists)
        
        # Convert back to Document objects
        docs = []
        for doc_id, rrf_score in rrf_results[:top_k]:
            if doc_id in self.metadata_mapping:
                metadata = self.metadata_mapping[doc_id]
                if isinstance(metadata, dict):
                    content = metadata.get('content', '')
                    
                    enhanced_metadata = {
                        'source': metadata.get('source', 'Unknown'),
                        'page': metadata.get('page_number'),
                        'rrf_score': float(rrf_score),
                        'score': float(rrf_score),
                        'quality': 'high' if rrf_score >= 0.5 else 'medium' if rrf_score >= 0.3 else 'low',
                        'retrieval_method': 'rrf_fusion'
                    }
                    
                    doc = Document(
                        page_content=content,
                        metadata=enhanced_metadata
                    )
                    docs.append(doc)
        
        logger.info(f"✅ RRF fusion completed - {len(docs)} documents retrieved")
        return docs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Main retrieval method - uses RRF fusion search"""
        return self._rrf_fusion_search(query, top_k=config.MAX_DOCS_FOR_RETRIEVAL)

# =============================================================================
# 📊 CROSS-ENCODER RE-RANKING
# =============================================================================

class DocumentRanker:
    """Advanced document ranking with Cross-Encoder"""
    
    def __init__(self):
        self.reranker = None
        self.reranker_name = None
        self._init_cross_encoder()
    
    def _init_cross_encoder(self):
        """Initialize Cross-Encoder for re-ranking"""
        logger.info("Loading Cross-Encoder for document re-ranking...")
        
        try:
            self.reranker = CrossEncoder(
                config.CROSS_ENCODER_MODEL,
                max_length=512,
                trust_remote_code=True
            )
            self.reranker_name = config.CROSS_ENCODER_MODEL
            logger.info(f"✅ Cross-Encoder '{config.CROSS_ENCODER_MODEL}' loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Cross-Encoder: {e}")
            # Try fallback model
            try:
                fallback_model = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
                self.reranker = CrossEncoder(fallback_model, max_length=512)
                self.reranker_name = fallback_model
                logger.info(f"✅ Fallback Cross-Encoder '{fallback_model}' loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback Cross-Encoder: {e2}")
                self.reranker = None
    
    def _is_form_field_or_template(self, content: str) -> bool:
        """Detect form fields and templates to filter out"""
        content_lower = content.lower().strip()
        
        form_patterns = [
            r':\s*\.{3,}',                      # ": ..."
            r':\s*_{3,}',                       # ": ___"
            r':\s*\d+\.\s*',                    # ": 5."
            r'form\s*no\.',                     # "form no."
            r'application\s*form',              # "application form"
            r'checklist\s*item',                # "checklist item"
            r'\[\s*\]',                         # "[]"
            r'_{4,}',                           # "____"
            r'\(please\s+fill\)',              # "(please fill)"
            r'\(to\s+be\s+filled\)',           # "(to be filled)"
            r'\(specify\)',                     # "(specify)"
            r'\(attach\s+document\)',          # "(attach document)"
            r'signature\s*block',              # "signature block"
            r'date\s*:\s*\d{1,2}/\d{1,2}/\d{4}', # "date: 12/25/2023"
            r'reference\s*no\.',               # "reference no."
            r'^\s*(name|address|phone|email|passport|tin|nid|bin|vat)?\s*:\s*(if any)?\s*\d*\.?\s*'
        ]
        
        for pattern in form_patterns:
            if re.search(pattern, content_lower):
                return True
        
        # Check if content is too short and uninformative
        if len(content.strip()) < 30 and ':' in content:
            return True
            
        return False
    
    def _cross_encoder_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Re-rank documents using Cross-Encoder"""
        if not docs or not self.reranker:
            return docs
        
        logger.info(f"🔍 Re-ranking {len(docs)} documents with Cross-Encoder ({self.reranker_name})")
        
        try:
            pairs = []
            valid_docs = []
            filtered_count = 0
            
            for doc in docs:
                content = doc.page_content.strip()
                
                # Skip form fields and templates
                if self._is_form_field_or_template(content):
                    filtered_count += 1
                    continue
                
                # Skip very short content
                if len(content) < 50:
                    filtered_count += 1
                    continue
                
                pairs.append([query, content])
                valid_docs.append(doc)
            
            if filtered_count > 0:
                logger.info(f"⚠️ Filtered out {filtered_count} form fields/short content documents")
            
            if not pairs:
                logger.warning("No valid documents after filtering")
                return docs[:config.MAX_DOCS_FOR_CONTEXT]
            
            # Get relevance scores
            scores = self.reranker.predict(
                pairs,
                batch_size=config.CROSS_ENCODER_BATCH_SIZE,
                convert_to_numpy=True
            )
            
            # Combine and sort
            doc_scores = list(zip(valid_docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"📊 Top 5 Cross-Encoder scores: {[f'{score:.3f}' for _, score in doc_scores[:5]]}")
            
            # Filter by relevance threshold
            filtered_docs = []
            below_threshold_count = 0
            
            for doc, score in doc_scores:
                if score >= config.RELEVANCE_THRESHOLD and len(filtered_docs) < config.MAX_DOCS_FOR_CONTEXT:
                    doc.metadata['rerank_score'] = float(score)
                    doc.metadata['quality'] = 'high' if score >= 0.5 else 'medium' if score >= 0.3 else 'low'
                    filtered_docs.append(doc)
                    logger.info(f"✅ Document relevance score: {score:.3f}")
                else:
                    below_threshold_count += 1
            
            if below_threshold_count > 0:
                logger.info(f"⚠️ {below_threshold_count} documents below relevance threshold ({config.RELEVANCE_THRESHOLD})")
            
            if not filtered_docs:
                logger.warning(f"No documents above relevance threshold, using top 2 documents")
                filtered_docs = [doc for doc, _ in doc_scores[:2]]
                for doc in filtered_docs:
                    doc.metadata['quality'] = 'low_confidence'
            
            logger.info(f"✅ Cross-Encoder re-ranking completed - {len(filtered_docs)} documents selected")
            return filtered_docs
            
        except Exception as e:
            logger.warning(f"Cross-encoder re-ranking failed: {e}")
            return docs[:config.MAX_DOCS_FOR_CONTEXT]
    
    def _lexical_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Fallback lexical re-ranking when Cross-Encoder is not available"""
        if not docs:
            return docs
        
        logger.info(f"📊 Using lexical re-ranking (Cross-Encoder not available)")
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            
            # Skip form fields and templates
            if self._is_form_field_or_template(content):
                continue
            
            # Skip very short content
            if len(content) < 50:
                continue
            
            content_lower = content.lower()
            content_words = set(content_lower.split())
            
            # Calculate overlap score
            overlap = len(query_words.intersection(content_words))
            total_query_words = len(query_words)
            
            if total_query_words > 0:
                score = overlap / total_query_words
            else:
                score = 0
            
            # Boost for exact phrase matches
            if query_lower in content_lower:
                score += 0.5
            
            scored_docs.append((doc, score))
        
        # Sort and filter
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        filtered_docs = []
        min_score = 0.1
        
        for doc, score in scored_docs:
            if score >= min_score and len(filtered_docs) < config.MAX_DOCS_FOR_CONTEXT:
                doc.metadata['lexical_score'] = float(score)
                doc.metadata['quality'] = 'medium' if score >= 0.3 else 'low'
                filtered_docs.append(doc)
                logger.info(f"✅ Document lexical score: {score:.3f}")
        
        if not filtered_docs and scored_docs:
            filtered_docs = [scored_docs[0][0]]
            filtered_docs[0].metadata['quality'] = 'low_confidence'
        
        return filtered_docs
    
    def rank_and_filter(self, docs: List[Document], query: str) -> List[Document]:
        """Main ranking method"""
        if not docs:
            return docs
        
        if self.reranker is not None:
            return self._cross_encoder_rerank(docs, query)
        else:
            return self._lexical_rerank(docs, query)
    
    def prepare_docs(self, docs: List[Document]) -> List[Document]:
        """Prepare documents by truncating if needed"""
        processed = []
        for doc in docs:
            content = doc.page_content
            if len(content) > config.CONTEXT_CHUNK_SIZE:
                content = content[:config.CONTEXT_CHUNK_SIZE] + "..."
            
            processed_doc = Document(
                page_content=content,
                metadata=doc.metadata
            )
            processed.append(processed_doc)
        
        return processed

# =============================================================================
# 🤖 RESPONSE GENERATION
# =============================================================================

class ResponseGenerator:
    """Advanced response generation with language support"""
    
    def __init__(self, language_detector: LanguageDetector):
        self.language_detector = language_detector
        self._init_llm()
    
    def _init_llm(self):
        """Initialize Ollama LLM"""
        try:
            self.llm = ChatOllama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.OLLAMA_TEMPERATURE,
                num_predict=config.OLLAMA_NUM_PREDICT,
                top_p=config.OLLAMA_TOP_P,
                repeat_penalty=config.OLLAMA_REPEAT_PENALTY
            )
            logger.info(f"✅ Ollama LLM initialized with model: {config.OLLAMA_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            self.llm = None
    
    def create_language_specific_chain(self, detected_language: str):
        """Create retrieval chain with language-specific prompt"""
        language_prompt = self.language_detector.get_language_specific_prompt(detected_language)
        return create_stuff_documents_chain(self.llm, language_prompt)
    
    def generate_response(self, query: str, docs: List[Document], detected_language: str) -> str:
        """Generate response using LLM with context-aware prompts"""
        if not self.llm:
            return "LLM not available. Please check Ollama configuration."
        
        if not docs:
            if detected_language == 'bengali':
                return "আমার ডাটাবেসে আপনার প্রশ্নের জন্য প্রাসঙ্গিক তথ্য খুঁজে পাইনি।"
            else:
                return "I couldn't find relevant information in my database for your query."
        
        try:
            logger.info(f"🤖 Generating response in {detected_language} using {len(docs)} documents")
            qa_chain = self.create_language_specific_chain(detected_language)
            result = qa_chain.invoke({"input": query, "context": docs})
            
            # Post-process response for consistency
            processed_result = self._post_process_response(result, detected_language)
            
            logger.info("✅ Response generated successfully")
            return processed_result
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            if detected_language == 'bengali':
                return f"উত্তর তৈরি করতে সমস্যা হয়েছে: {e}"
            else:
                return f"Error generating response: {e}"
    
    def _post_process_response(self, response: str, language: str) -> str:
        """Post-process response for consistency and formatting"""
        if not response or not response.strip():
            return response
        
        # Clean up the response
        response = response.strip()
        
        # Standardize bullet points
        if language == 'bengali':
            # Replace various bullet point styles with standard bullet
            response = re.sub(r'^[\d]+\.\s*', '• ', response, flags=re.MULTILINE)
            response = re.sub(r'^[-*]\s*', '• ', response, flags=re.MULTILINE)
        else:
            # For English, use standard bullet points
            response = re.sub(r'^[\d]+\.\s*', '• ', response, flags=re.MULTILINE)
            response = re.sub(r'^[-]\s*', '• ', response, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = re.sub(r'[ \t]+', ' ', response)
        
        # Limit response length (backup safety)
        max_chars = 600 if language == 'bengali' else 500
        if len(response) > max_chars:
            # Find a good breaking point
            truncated = response[:max_chars]
            if '•' in truncated:
                # Break at the last complete bullet point
                last_bullet = truncated.rfind('•')
                if last_bullet > max_chars * 0.7:  # At least 70% of content
                    response = truncated[:last_bullet].strip()
            else:
                # Break at last sentence
                last_period = truncated.rfind('.')
                if last_period > max_chars * 0.7:
                    response = truncated[:last_period + 1]
                else:
                    response = truncated.rsplit(' ', 1)[0] + "..."
        
        return response.strip()

# =============================================================================
# 🚀 CORE RAG SYSTEM
# =============================================================================

class CoreRAGSystem:
    """Core RAG System with retrieval, re-ranking, and response generation"""
    
    def __init__(self):
        logger.info("🚀 Initializing Core RAG System...")
        
        # Initialize components
        self.language_detector = LanguageDetector()
        self.retriever = RRFFusionRetriever()
        self.ranker = DocumentRanker()
        self.response_generator = ResponseGenerator(self.language_detector)
        
        logger.info("✅ Core RAG System initialized successfully")
    
    async def process_query_async(self, query: str) -> Dict:
        """Async main query processing method for Telegram bot"""
        logger.info(f"🔍 Processing query: {query}")
        
        # Run the synchronous processing in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_query_sync, query)
    
    def process_query_sync(self, query: str) -> Dict:
        """Synchronous query processing method"""
        logger.info(f"🔍 Processing query: {query}")
        
        # Step 1: Detect language
        detected_language, confidence = self.language_detector.detect_language(query)
        logger.info(f"🌐 Language detected: {detected_language} (confidence: {confidence:.2f})")
        
        try:
            # Step 2: Retrieve documents using RRF fusion search
            logger.info("🔍 Retrieving documents with RRF fusion search...")
            source_docs = self.retriever.get_relevant_documents(query)
            
            if not source_docs:
                if detected_language == 'bengali':
                    response_text = "আমার ডাটাবেসে আপনার প্রশ্নের জন্য প্রাসঙ্গিক তথ্য খুঁজে পাইনি।"
                else:
                    response_text = "I couldn't find relevant information in my database."
                
                return {
                    "response": response_text,
                    "sources": [],
                    "contexts": [],
                    "detected_language": detected_language,
                    "language_confidence": confidence,
                    "retrieval_method": "rrf_fusion",
                    "documents_found": 0
                }
            
            logger.info(f"📄 Retrieved {len(source_docs)} documents")
            
            # Step 3: Rank and filter documents with Cross-Encoder
            logger.info("📊 Ranking and filtering documents...")
            filtered_docs = self.ranker.rank_and_filter(source_docs, query)
            prepared_docs = self.ranker.prepare_docs(filtered_docs)
            
            logger.info(f"✅ Final document set: {len(prepared_docs)} documents")
            
            # Step 4: Generate response
            logger.info("🤖 Generating response...")
            answer = self.response_generator.generate_response(
                query, prepared_docs, detected_language
            )
            
            # Step 5: Prepare sources and contexts
            sources = []
            contexts = []
            for doc in prepared_docs:
                source_info = {
                    "file": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0),
                    "score": doc.metadata.get("rerank_score", doc.metadata.get("rrf_score", 0.0)),
                    "quality": doc.metadata.get("quality", "unknown")
                }
                sources.append(source_info)
                contexts.append(doc.page_content)
            
            # Final response
            response = {
                "response": answer,
                "sources": sources,
                "contexts": contexts,
                "detected_language": detected_language,
                "language_confidence": confidence,
                "retrieval_method": "rrf_fusion",
                "documents_found": len(source_docs),
                "documents_used": len(prepared_docs),
                "cross_encoder_used": self.ranker.reranker is not None
            }
            
            logger.info("✅ Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            if detected_language == 'bengali':
                error_text = f"প্রশ্ন প্রক্রিয়াকরণে ত্রুটি: {e}"
            else:
                error_text = f"Error processing query: {e}"
            
            error_response = {
                "response": error_text,
                "sources": [],
                "contexts": [],
                "detected_language": detected_language,
                "language_confidence": confidence,
                "error": str(e)
            }
            return error_response
    
    def process_query(self, query: str) -> Dict:
        """Main query processing method (backward compatibility)"""
        return self.process_query_sync(query)
    
    def get_system_info(self) -> Dict:
        """Get system information and status"""
        return {
            "embedding_model_loaded": self.retriever.embedding_model is not None,
            "faiss_index_loaded": self.retriever.faiss_index is not None,
            "bm25_initialized": self.retriever.bm25 is not None,
            "colbert_initialized": self.retriever.colbert_model is not None,
            "dirichlet_initialized": self.retriever.document_stats is not None,
            "cross_encoder_loaded": self.ranker.reranker is not None,
            "cross_encoder_model": self.ranker.reranker_name,
            "llm_initialized": self.response_generator.llm is not None,
            "total_vectors": self.retriever.faiss_index.ntotal if self.retriever.faiss_index else 0,
            "metadata_entries": len(self.retriever.metadata_mapping) if self.retriever.metadata_mapping else 0,
            "config": {
                "faiss_index_path": config.FAISS_INDEX_PATH,
                "embedding_model": config.EMBEDDING_MODEL,
                "cross_encoder_model": config.CROSS_ENCODER_MODEL,
                "ollama_model": config.OLLAMA_MODEL,
                "rrf_settings": {
                    "rrf_k": config.RRF_K,
                    "rrf_weights": config.RRF_WEIGHTS
                },
                "colbert_model": config.COLBERT_MODEL,
                "dirichlet_mu": config.DIRICHLET_MU,
                "ollama_params": {
                    "temperature": config.OLLAMA_TEMPERATURE,
                    "top_p": config.OLLAMA_TOP_P,
                    "num_predict": config.OLLAMA_NUM_PREDICT,
                    "repeat_penalty": config.OLLAMA_REPEAT_PENALTY
                }
            }
        }

# =============================================================================
# 🤖 TELEGRAM BOT INTERFACE
# =============================================================================

# Global RAG system instance
rag_system = None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    welcome_message = """
🚀 **Advanced Financial RAG Assistant** 🏦

Welcome! I'm your AI-powered financial advisor assistant for Bangladesh, now enhanced with RRF Fusion technology! I can help you with:

🔍 **Services:**
• Bank loan information
• Account opening procedures
• Tax filing guidance
• Investment options
• Insurance policies
• Business registration
• Financial regulations

🌐 **Language Support:**
• English
• বাংলা (Bangla)

🔬 **Advanced Technology:**
• RRF (Reciprocal Rank Fusion) search
• FAISS semantic search
• BM25 keyword matching
• ColBERT dense retrieval
• Dirichlet Query Language Model

💡 **How to use:**
Just send me your financial question in either English or Bangla, and I'll provide detailed information using multiple advanced retrieval methods for the most accurate results.

**Example questions:**
• "How to apply for a car loan?"
• "ব্যাংক অ্যাকাউন্ট খোলার নিয়ম কি?"
• "What are the tax filing requirements?"

Type /help for more commands or just ask your question!
    """
    
    keyboard = [
        [InlineKeyboardButton("📊 System Info", callback_data="info")],
        [InlineKeyboardButton("❓ Help", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        welcome_message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command"""
    help_message = """
📚 **Available Commands:**

/start - Welcome message and introduction
/help - Show this help message
/info - Display system information and status
/stats - Show system statistics

💬 **How to ask questions:**

1. **Direct Questions:** Just type your question
   • "How to open a savings account?"
   • "ব্যাংক লোনের সুদের হার কত?"

2. **Financial Topics I can help with:**
   • Bank services and procedures
   • Loan applications and requirements
   • Tax filing and regulations
   • Investment and savings options
   • Insurance policies
   • Business registration
   • Foreign exchange
   • Remittance services

🌟 **Tips for better results:**
• Be specific in your questions
• Mention if you need information for a particular bank or service
• Ask follow-up questions for clarification

🔧 **System Features:**
• RRF fusion search (4 retrievers combined)
• FAISS semantic search
• BM25 keyword matching
• ColBERT dense retrieval
• Dirichlet Query Language Model
• Cross-encoder document re-ranking
• Multi-language support (English/Bangla)
• Real-time processing
    """
    
    await update.message.reply_text(help_message, parse_mode=ParseMode.MARKDOWN)

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /info command"""
    try:
        info = rag_system.get_system_info()
        
        info_message = f"""
📊 **System Information**

🤖 **Core Components:**
• Embedding Model: {info['config']['embedding_model']}
• Cross-Encoder: {info['config']['cross_encoder_model']}
• LLM Model: {info['config']['ollama_model']}

🎛️ **LLM Parameters:**
• Temperature: {info['config']['ollama_params']['temperature']}
• Top-P: {info['config']['ollama_params']['top_p']}
• Max Tokens: {info['config']['ollama_params']['num_predict']}
• Repeat Penalty: {info['config']['ollama_params']['repeat_penalty']}

📚 **Database Status:**
• Total Vectors: {info['total_vectors']:,}
• Metadata Entries: {info['metadata_entries']:,}
• FAISS Index: {'✅ Loaded' if info['faiss_index_loaded'] else '❌ Not Loaded'}
• BM25 Search: {'✅ Ready' if info['bm25_initialized'] else '❌ Not Ready'}
• ColBERT Model: {'✅ Ready' if info['colbert_initialized'] else '❌ Not Ready'}
• Dirichlet QLM: {'✅ Ready' if info['dirichlet_initialized'] else '❌ Not Ready'}

🔧 **Model Status:**
• Embedding Model: {'✅ Loaded' if info['embedding_model_loaded'] else '❌ Not Loaded'}
• Cross-Encoder: {'✅ Loaded' if info['cross_encoder_loaded'] else '❌ Not Loaded'}
• LLM: {'✅ Ready' if info['llm_initialized'] else '❌ Not Ready'}

⚙️ **RRF Fusion Configuration:**
• RRF K Parameter: {info['config']['rrf_settings']['rrf_k']}
• FAISS Weight: {info['config']['rrf_settings']['rrf_weights']['faiss']}
• BM25 Weight: {info['config']['rrf_settings']['rrf_weights']['bm25']}
• ColBERT Weight: {info['config']['rrf_settings']['rrf_weights']['colbert']}
• Dirichlet Weight: {info['config']['rrf_settings']['rrf_weights']['dirichlet']}
• ColBERT Model: {info['config']['colbert_model']}
• Dirichlet μ: {info['config']['dirichlet_mu']}
• Index Path: {info['config']['faiss_index_path']}

🎯 **Status:** {'🟢 All Systems Operational' if all([info['faiss_index_loaded'], info['embedding_model_loaded'], info['llm_initialized']]) else '🟡 Some Components Not Ready'}
        """
        
        await update.message.reply_text(info_message, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error getting system info: {e}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command"""
    try:
        info = rag_system.get_system_info()
        
        stats_message = f"""
📈 **System Statistics**

📊 **Document Database:**
• Total Documents: {info['total_vectors']:,}
• Metadata Records: {info['metadata_entries']:,}
• Index Size: {info['config']['faiss_index_path']}

🔍 **Search Capabilities:**
• RRF Fusion: 4 retrievers combined
• FAISS Semantic Search
• BM25 Keyword Search
• ColBERT Dense Retrieval
• Dirichlet Query Language Model
• Re-ranking: Cross-Encoder
• Languages: English, বাংলা
• Domain: Financial Services (Bangladesh)

⚡ **Performance:**
• Retrieval Method: RRF Fusion (4 retrievers)
• Max Documents Retrieved: 30
• Max Context Documents: 6
• Context Chunk Size: 1,500 chars
• RRF K Parameter: 60

🎯 **Quality Filters:**
• Relevance Threshold: 0.15
• Financial Term Density: ≥0.5%
• Form Field Filtering: Enabled
        """
        
        await update.message.reply_text(stats_message, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error getting statistics: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages (questions)"""
    try:
        user_query = update.message.text.strip()
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        if not user_query:
            await update.message.reply_text("Please send me a question about financial services.")
            return
        
        logger.info(f"Query from user {username} ({user_id}): {user_query}")
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🔍 Processing your question...\nThis may take a few moments.",
            parse_mode=None
        )
        
        # Process the query
        result = await rag_system.process_query_async(user_query)
        
        # Delete processing message
        await processing_msg.delete()
        
        # Prepare response
        response_text = result["response"]
        
        # Add language and source info
        lang_emoji = "🇧🇩" if result["detected_language"] == "bengali" else "🇺🇸"
        footer = f"\n\n{lang_emoji} Language: {result['detected_language'].title()}"
        
        if result.get("sources"):
            footer += f"\n📚 Sources: {len(result['sources'])} documents"
        
        # Split long messages
        max_length = 4000  # Telegram limit is 4096
        if len(response_text + footer) > max_length:
            # Split response
            chunks = [response_text[i:i+max_length-100] for i in range(0, len(response_text), max_length-100)]
            
            for i, chunk in enumerate(chunks):
                if i == len(chunks) - 1:  # Last chunk
                    chunk += footer
                await update.message.reply_text(chunk, parse_mode=None)
        else:
            await update.message.reply_text(response_text + footer, parse_mode=None)
        
        # Send source information if available
        if result.get("sources") and len(result["sources"]) > 0:
            sources_text = "📄 Source Documents:\n\n"
            for i, source in enumerate(result["sources"][:5], 1):  # Limit to 5 sources
                score = source.get('score', 0)
                quality = source.get('quality', 'unknown')
                sources_text += f"{i}. {source['file']} (Page {source['page']})\n"
                sources_text += f"   Score: {score:.3f}, Quality: {quality}\n\n"
            
            await update.message.reply_text(sources_text, parse_mode=None)
        
        logger.info(f"Response sent to user {username} ({user_id})")
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text(
            f"❌ Sorry, I encountered an error while processing your question: {e}\n\n"
            "Please try again or contact support if the problem persists."
        )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "info":
        await info_command(update, context)
    elif query.data == "help":
        await help_command(update, context)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")

def main():
    """Main function to run the Telegram bot"""
    global rag_system
    
    # Get bot token from configuration
    bot_token = TELEGRAM_BOT_TOKEN
    if not bot_token or bot_token == "YOUR_BOT_TOKEN_HERE":
        logger.error("TELEGRAM_BOT_TOKEN not configured")
        print("❌ Error: TELEGRAM_BOT_TOKEN not configured")
        print("Please edit the TELEGRAM_BOT_TOKEN variable at the top of this file:")
        print("TELEGRAM_BOT_TOKEN = \"your_actual_bot_token_here\"")
        print("\nTo get a bot token:")
        print("1. Message @BotFather on Telegram")
        print("2. Send /newbot command")
        print("3. Follow instructions to create your bot")
        print("4. Copy the token and paste it in the configuration section")
        return
    
    print("🚀 Initializing Financial RAG System...")
    try:
        # Initialize RAG system
        rag_system = CoreRAGSystem()
        print("✅ RAG System initialized successfully")
        
        # Get system info to verify everything is working
        info = rag_system.get_system_info()
        print(f"📊 System Status:")
        print(f"  - Documents: {info['total_vectors']:,}")
        print(f"  - FAISS Index: {'✅' if info['faiss_index_loaded'] else '❌'}")
        print(f"  - Cross-Encoder: {'✅' if info['cross_encoder_loaded'] else '❌'}")
        print(f"  - LLM: {'✅' if info['llm_initialized'] else '❌'}")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    print("🤖 Starting Telegram Bot...")
    
    # Create bot application
    application = Application.builder().token(bot_token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("info", info_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    print("✅ Telegram Bot is running!")
    print("📱 Send /start to your bot to begin")
    print("🛑 Press Ctrl+C to stop the bot")
    
    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()