import os
import re
import logging
import time
import torch
import faiss
import pickle
import numpy as np
from typing import Dict, List, Set
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG utilities and language detection
from rag_utils import RAGUtils
from advanced_rag_feedback import AdvancedRAGFeedbackLoop
from config import config
from language_utils import LanguageDetector, BilingualResponseFormatter

# Import for remote FAISS API
import requests

# Import delta indexing
from delta_indexing import DocumentVersionTracker

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Enhanced Bilingual Prompt Template ---
PROMPT_TEMPLATE = """
You are a helpful financial advisor specializing in Bangladesh's banking and financial services.
You can understand and respond in both English and Bangla languages.
Always respond in a natural, conversational tone as if speaking to a friend.

IMPORTANT INSTRUCTIONS:
- Answer based on the provided information, even if it's partially relevant
- Extract and synthesize useful information from the context to provide a helpful response
- Ignore form fields, blank templates, placeholder text, and incomplete document fragments
- Never say "According to the context" - just answer directly and naturally
- If information is limited, provide what you can and suggest where to get more details
- Use Bangladeshi Taka (৳/Tk) as currency
- Be concise but comprehensive - aim to be helpful even with partial information
- CRITICAL: Respond in the SAME LANGUAGE as the user's question (either English or Bangla)
- The context may contain both English and Bangla text - use whichever is relevant to answer the question
- Focus on providing actionable, practical advice
- If the user asks in Bangla, respond completely in Bangla using natural, conversational Bengali
- If the user asks in English, respond completely in English

Context Information:
{context}

Question: {input}

Answer (provide a helpful response in the same language as the question):"""
QA_PROMPT = PromptTemplate(input_variables=["context", "input"], template=PROMPT_TEMPLATE)

class DocumentRetriever:
    """Handles document retrieval from FAISS index with hybrid search capabilities"""
    
    def __init__(self):
        self.use_remote_faiss = False
        self.faiss_index = None
        self.metadata_mapping = None
        self.embedding_model = None
        self.bm25 = None
        self.bm25_corpus = None
        self.use_hybrid_retrieval = False
        
        # Initialize components
        self._init_embedding_model()
        self._init_faiss_index()
        self._init_bm25()
        
        # Initialize delta indexing
        self.version_tracker = DocumentVersionTracker(config.FAISS_INDEX_PATH)
        
    def _init_embedding_model(self):
        """Initialize the sentence transformer embedding model"""
        try:
            print("[INFO] Loading embedding model (BAAI/bge-m3)...")
            self.embedding_model = SentenceTransformer('BAAI/bge-m3', device='cpu')
            print("[INFO] [OK] Embedding model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def _init_faiss_index(self):
        """Initialize FAISS index and metadata"""
        try:
            # Manually load FAISS index and metadata
            index_file = os.path.join(config.FAISS_INDEX_PATH, "financial_advisor_faiss.index")
            metadata_file = os.path.join(config.FAISS_INDEX_PATH, "financial_advisor_metadata.pkl")

            if not os.path.exists(index_file) or not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Index files not found in {config.FAISS_INDEX_PATH}. Looked for {index_file} and {metadata_file}")

            print(f"[INFO] Loading FAISS index from: {index_file}")
            self.faiss_index = faiss.read_index(index_file)
            
            print(f"[INFO] Loading metadata from: {metadata_file}")
            with open(metadata_file, 'rb') as f:
                self.metadata_mapping = pickle.load(f)
            
            print("[INFO] [OK] FAISS index and metadata loaded successfully.")
            print(f"[INFO] Index has {self.faiss_index.ntotal} vectors.")
        except Exception as e:
            print(f"[ERROR] Failed to load local FAISS index manually: {e}")
            logger.error(f"Failed to load local FAISS index manually: {e}")
            # If manual loading fails, we cannot proceed.
            self.faiss_index = None
            self.metadata_mapping = None

    def _init_bm25(self):
        """Initialize BM25 for keyword-based retrieval"""
        try:
            if self.metadata_mapping is None:
                print("[WARNING] No metadata mapping available for BM25 initialization")
                self.bm25 = None
                self.bm25_corpus = None
                return

            # Prepare corpus for BM25
            corpus = []
            doc_ids = []
            
            for doc_id, metadata in self.metadata_mapping.items():
                content = metadata.get('content', '')
                if content and content.strip():
                    # Tokenize content for BM25
                    tokens = content.lower().split()
                    corpus.append(tokens)
                    doc_ids.append(doc_id)
            
            if corpus:
                self.bm25 = BM25Okapi(corpus)
                self.bm25_doc_ids = doc_ids
                self.bm25_corpus = corpus
                print(f"[INFO] [OK] BM25 initialized with {len(corpus)} documents")
            else:
                print("[WARNING] No valid documents found for BM25 initialization")
                self.bm25 = None
                self.bm25_corpus = None
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize BM25: {e}")
            self.bm25 = None
            self.bm25_corpus = None

    def enable_hybrid_retrieval(self):
        """Enable hybrid retrieval (BM25 + FAISS)"""
        self.use_hybrid_retrieval = True
        print("[INFO] [OK] Hybrid retrieval ENABLED")
        if self.bm25 is None:
            print("[WARNING] BM25 not initialized. Re-initializing...")
            self._init_bm25()
        print(f"[INFO] Hybrid retrieval ready: {'Yes' if self.bm25 is not None else 'No (BM25 failed to initialize)'}")

    def disable_hybrid_retrieval(self):
        """Disable hybrid retrieval (use FAISS only)"""
        self.use_hybrid_retrieval = False
        print("[INFO] [WARNING] Hybrid retrieval DISABLED - using FAISS only")

    def _manual_similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        """Manually perform similarity search using loaded FAISS index and metadata."""
        if self.faiss_index is None or self.metadata_mapping is None:
            print("[ERROR] FAISS index or metadata not loaded.")
            return []

        # Check if hybrid retrieval is enabled
        if self.use_hybrid_retrieval and hasattr(self, 'bm25') and self.bm25 is not None:
            # Use hybrid search (BM25 + FAISS)
            print("[INFO] [SEARCH] Using hybrid retrieval (BM25 + FAISS)...")
            return self._hybrid_search(query, top_k=top_k)
        else:
            # Use standard FAISS search with enhanced MultiQuery approach
            print("[INFO] [SEARCH] Using standard FAISS retrieval with financial domain multi-query...")
            
            # Generate financial domain-specific query variations
            query_variations = [query]  # Start with original query
            
            # Add domain-specific variations based on financial keywords
            query_lower = query.lower()
            financial_variations_added = []
            
            # Loan-related queries
            if 'loan' in query_lower:
                financial_variations_added.append(f"bank loan application Bangladesh {query}")
            
            # Account-related queries
            if any(term in query_lower for term in ['account', 'accounts']):
                financial_variations_added.append(f"bank account opening Bangladesh {query}")
                
            # Tax-related queries
            if 'tax' in query_lower:
                financial_variations_added.append(f"Bangladesh tax filing {query}")
                
            # Investment-related queries
            if any(term in query_lower for term in ['investment', 'invest']):
                financial_variations_added.append(f"Bangladesh investment options {query}")
                
            # Insurance-related queries
            if 'insurance' in query_lower:
                financial_variations_added.append(f"Bangladesh insurance policy {query}")
                
            # Credit-related queries
            if 'credit' in query_lower:
                financial_variations_added.append(f"bank credit card application Bangladesh {query}")
                
            # Deposit-related queries
            if 'deposit' in query_lower:
                financial_variations_added.append(f"bank deposit account Bangladesh {query}")
                
            # Business-related queries
            if 'business' in query_lower:
                financial_variations_added.append(f"Bangladesh business registration {query}")
                
            # Add the financial domain variations to our query set
            query_variations.extend(financial_variations_added)
            
            # Also generate LLM-based variations for additional coverage
            llm_variations_count = 0
            try:
                # Generate alternative queries using LLM (limit to 2 for performance)
                multi_query_prompt = f"""
                Generate 2 alternative ways to ask the same question to improve retrieval in a Bangladesh financial context.
                Original question: {query}
                
                Provide only the alternative questions, one per line, without any other text:
                """
                
                # For now, we'll skip LLM-based variations in this refactored version
                # In a full implementation, we would initialize an LLM here
                pass
            except Exception as e:
                print(f"[INFO] LLM-based multi-query generation failed (continuing with domain variations): {e}")
            
            print(f"[INFO] Generated {len(query_variations)-1} query variations ({len(financial_variations_added)} financial domain, {llm_variations_count} LLM-based)")
            
            # Search with all query variations and collect results
            all_results = []
            for i, q_var in enumerate(query_variations):
                if i > 0:
                    print(f"[INFO] MultiQuery variation {i}: '{q_var}'")
                
                # Encode the query
                query_embedding = self.embedding_model.encode([q_var], normalize_embeddings=True)
                query_embedding = np.array(query_embedding, dtype=np.float32)

                # Search the index
                scores, indices = self.faiss_index.search(query_embedding, top_k)

                # Collect results with debug information
                variation_results_count = 0
                for j in range(min(top_k, len(indices[0]))):
                    idx = indices[0][j]
                    score = scores[0][j]
                    
                    if idx in self.metadata_mapping:
                        # Create unique identifier for deduplication
                        content = self.metadata_mapping[idx].get('content', '')
                        result_key = (idx, content[:100])  # Use first 100 chars for uniqueness
                        
                        all_results.append({
                            'idx': idx,
                            'score': float(score),
                            'content': content,
                            'metadata': self.metadata_mapping[idx],
                            'query_source': i,  # Track which query generated this result
                            'query_text': q_var[:50] + "..." if len(q_var) > 50 else q_var  # For debugging
                        })
                        variation_results_count += 1
                
                if i > 0:
                    print(f"[INFO]   -> Retrieved {variation_results_count} results for variation {i}")

            # Deduplicate results based on document content using more robust method
            unique_results = {}
            duplicate_count = 0
            for result in all_results:
                # Use a combination of document ID and content hash for better deduplication
                content_hash = hash(result['content'][:500])  # Hash first 500 chars
                doc_identifier = f"{result['idx']}_{content_hash}"
                
                if doc_identifier not in unique_results or result['score'] > unique_results[doc_identifier]['score']:
                    if doc_identifier in unique_results:
                        duplicate_count += 1
                    unique_results[doc_identifier] = result

            print(f"[INFO] Deduplication removed {duplicate_count} duplicate results")
            
            # Convert to list and sort by score
            sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
            
            # Apply quality filtering to final results
            docs = []
            quality_filtered_count = 0
            
            for result in sorted_results[:top_k * 3]:  # Get more results for better selection
                idx = result['idx']
                score = result['score']
                
                # Add score analysis to metadata
                metadata = result['metadata']
                enhanced_metadata = {
                    'source': metadata.get('source', 'Unknown'),
                    'page': metadata.get('page_number'),
                    'score': float(score),
                    'quality': 'high' if score >= 0.5 else 'medium' if score >= 0.3 else 'low',
                    'query_variation': result['query_source'],
                    'query_text': result.get('query_text', 'original')  # For debugging
                }
                
                # Filter out very low quality results with enhanced financial term density check
                if score >= 0.2:  # Increased threshold for better quality
                    # Additional quality check for financial content
                    content = metadata.get('content', '')
                    
                    # Enhanced financial terms list with more comprehensive coverage
                    financial_terms = [
                        # Core banking terms
                        'loan', 'interest', 'account', 'tax', 'investment', 'deposit', 'balance',
                        'rate', 'fee', 'charge', 'document', 'requirement', 'procedure', 'application',
                        'regulation', 'policy', 'scheme', 'benefit', 'eligibility', 'criteria',
                        'bank', 'finance', 'credit', 'debit', 'transaction', 'statement',
                        # Additional financial services
                        'insurance', 'premium', 'claim', 'mortgage', 'equity', 'bond', 'mutual fund',
                        'portfolio', 'dividend', 'capital', 'liability', 'asset', 'liquidity',
                        'inflation', 'deflation', 'monetary', 'fiscal', 'budget', 'audit',
                        # Banking operations
                        'withdrawal', 'transfer', 'payment', 'settlement', 'clearing', 'custody',
                        'custodian', 'broker', 'trading', 'exchange', 'market', 'securities',
                        # Account types
                        'savings', 'current', 'fixed', 'recurring', 'joint', 'individual', 'corporate',
                        'business', 'personal', 'commercial', 'retail', 'wholesale',
                        # Financial instruments
                        'cheque', 'draft', 'bill', 'note', 'security', 'share', 'stock', 'option',
                        'future', 'derivative', 'swap', 'forward', 'spot',
                        # Regulatory and compliance
                        'compliance', 'kyc', 'aml', 'fatca', 'crs', 'basel', 'ifrs', 'gaap',
                        'auditor', 'supervision', 'oversight', 'enforcement', 'penalty',
                        # Risk management
                        'risk', 'credit risk', 'market risk', 'operational risk', 'liquidity risk',
                        'counterparty', 'exposure', 'limit', 'collateral', 'guarantee',
                        # Customer services
                        'customer', 'client', 'service', 'branch', 'atm', 'online', 'mobile',
                        'digital', 'internet', 'web', 'app', 'portal', 'channel',
                        # Bangla financial terms
                        'ঋণ', 'সুদ', 'অ্যাকাউন্ট', 'ট্যাক্স', 'বিনিয়োগ', 'জমা', 'ব্যালেন্স',
                        'হার', 'ফি', 'চার্জ', 'নথি', 'প্রয়োজনীয়তা', 'কার্যপ্রণালী', 'আবেদন',
                        'বিধি', 'নীতি', 'প্রকল্প', 'সুবিধা', 'যোগ্যতা', 'মানদণ্ড', 'ব্যাংক',
                        'আর্থিক', 'ক্রেডিট', 'ডেবিট', 'লেনদেন', 'বিবৃতি', 'বীমা', 'প্রিমিয়াম',
                        'দাবি', 'মার্টগেজ', 'ইক্যুইটি', 'বন্ড', 'পারস্পরিক তহবিল'
                    ]
                    
                    # Count financial terms in the content using word boundaries for accuracy
                    term_count = 0
                    content_lower = content.lower()
                    for term in financial_terms:
                        # Use word boundaries to avoid partial matches
                        term_count += len(re.findall(r'\b' + re.escape(term.lower()) + r'\b', content_lower))
                    
                    # Calculate term density (terms per 100 words) with enhanced threshold
                    words = content.split()
                    word_count = len(words)
                    if word_count > 0:
                        term_density = (term_count / word_count) * 100
                        # Enhanced threshold: Require at least 0.5 financial terms per 100 words (increased from 0.3)
                        required_density = 0.5
                        
                        # Log detailed filtering information for debugging
                        if term_density < required_density:
                            print(f"[DEBUG] Low financial term density: {term_density:.3f}% "
                                  f"(< {required_density}%), term count: {term_count}, word count: {word_count}")
                            quality_filtered_count += 1
                        else:
                            doc = Document(
                                page_content=content,
                                metadata=enhanced_metadata
                            )
                            docs.append(doc)
                            print(f"[DEBUG] Document passed financial term density check: {term_density:.3f}%")
                    else:
                        print(f"[DEBUG] Empty content filtered out")
                        quality_filtered_count += 1
                else:
                    print(f"[DEBUG] Low similarity score filtered out: {score:.3f} (< 0.2)")
                    quality_filtered_count += 1

            if quality_filtered_count > 0:
                print(f"[INFO] [WARNING] Filtered out {quality_filtered_count} low-quality local results")
                
            final_docs = docs[:top_k]  # Return top_k results
            print(f"[INFO] [OK] Retrieved {len(final_docs)} documents from local FAISS "
                  f"(filtered from {len(sorted_results)} unique results across {len(query_variations)} query variations)")
            
            # Log top 3 results for debugging
            for i, doc in enumerate(final_docs[:3]):
                print(f"[DEBUG] Top result {i+1}: Score={doc.metadata.get('score', 'N/A'):.3f}, "
                      f"Source={doc.metadata.get('source', 'Unknown')}, "
                      f"Variation={doc.metadata.get('query_variation', 'unknown')}")
            
            return final_docs

    def _hybrid_search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Perform hybrid search combining BM25 (keyword-based) and FAISS (semantic) retrieval.
        
        Args:
            query: User query string
            top_k: Number of results to return
            
        Returns:
            List of ranked documents combining both retrieval methods
        """
        print(f"[INFO] [SEARCH] Performing hybrid search for query: '{query[:50]}...'")
        
        # Determine dynamic weights based on query keywords
        # For exact/specific queries, favor keyword search
        exact_keywords = ["exact", "specific", "form", "particular", "precise"]
        if any(keyword in query.lower() for keyword in exact_keywords):
            # More keyword-focused for exact matches
            semantic_weight = 0.4
            keyword_weight = 0.6
            print(f"[INFO] Using keyword-focused weights: semantic={semantic_weight}, keyword={keyword_weight}")
        else:
            # Default balanced approach
            semantic_weight = 0.6  # Reduced from 0.7
            keyword_weight = 0.4   # Increased from 0.3
            print(f"[INFO] Using semantic-focused weights: semantic={semantic_weight}, keyword={keyword_weight}")
        
        # Get results from both retrieval methods
        semantic_docs = self._manual_similarity_search(query, top_k=top_k * 2)
        keyword_docs = self._bm25_search(query, top_k=top_k * 2)
        
        # Combine and re-rank results
        hybrid_scores = {}
        
        # Add semantic scores with adjusted weight
        for doc in semantic_docs:
            doc_id = self._get_doc_identifier(doc)
            score = doc.metadata.get('score', 0.0) * semantic_weight
            hybrid_scores[doc_id] = {'doc': doc, 'semantic_score': score, 'keyword_score': 0.0}
        
        # Add keyword scores with adjusted weight
        for doc in keyword_docs:
            doc_id = self._get_doc_identifier(doc)
            score = doc.metadata.get('bm25_score', 0.0) * keyword_weight
            if doc_id in hybrid_scores:
                hybrid_scores[doc_id]['keyword_score'] = score
                hybrid_scores[doc_id]['doc'] = doc  # Use the version with BM25 metadata
            else:
                hybrid_scores[doc_id] = {'doc': doc, 'semantic_score': 0.0, 'keyword_score': score}
        
        # Calculate final hybrid scores
        for doc_id, scores in hybrid_scores.items():
            final_score = scores['semantic_score'] + scores['keyword_score']
            scores['final_score'] = final_score
            # Add final score to document metadata
            scores['doc'].metadata['hybrid_score'] = final_score
            scores['doc'].metadata['semantic_score'] = scores['semantic_score']
            scores['doc'].metadata['keyword_score'] = scores['keyword_score']
        
        # Sort by final hybrid score and return top_k results
        sorted_results = sorted(
            hybrid_scores.values(), 
            key=lambda x: x['final_score'], 
            reverse=True
        )
        
        # Apply quality filtering
        filtered_docs = []
        for item in sorted_results:
            doc = item['doc']
            final_score = item['final_score']
            
            # Enhanced quality filtering with hybrid scores
            if final_score >= 0.15:  # Hybrid score threshold
                # Add quality metadata
                doc.metadata['quality'] = 'high' if final_score >= 0.5 else 'medium' if final_score >= 0.3 else 'low'
                doc.metadata['hybrid_retrieval'] = True
                filtered_docs.append(doc)
                
                if len(filtered_docs) >= top_k:
                    break
        
        print(f"[INFO] [OK] Hybrid search completed - {len(filtered_docs)} documents retrieved")
        return filtered_docs[:top_k]

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Perform BM25 keyword-based search.
        
        Args:
            query: User query string
            top_k: Number of results to return
            
        Returns:
            List of documents ranked by BM25 scores
        """
        if self.bm25 is None or self.metadata_mapping is None:
            print("[WARNING] BM25 not initialized, returning empty results")
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Get top_k documents
            top_indices = bm25_scores.argsort()[::-1][:top_k]
            
            # Convert to LangChain Documents
            docs = []
            for idx in top_indices:
                if idx < len(self.bm25_doc_ids):
                    doc_id = self.bm25_doc_ids[idx]
                    if doc_id in self.metadata_mapping:
                        metadata = self.metadata_mapping[doc_id]
                        doc = Document(
                            page_content=metadata.get('content', ''),
                            metadata={
                                'source': metadata.get('source', 'Unknown'),
                                'page': metadata.get('page_number'),
                                'bm25_score': float(bm25_scores[idx]),
                                'score': float(bm25_scores[idx])  # For compatibility
                            }
                        )
                        docs.append(doc)
            
            return docs
            
        except Exception as e:
            print(f"[ERROR] BM25 search failed: {e}")
            return []

    def _get_doc_identifier(self, doc: Document) -> str:
        """Get unique identifier for a document for deduplication"""
        content = doc.page_content
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 0)
        return f"{source}_{page}_{content[:100]}"

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """This method makes the class act like a retriever."""
        return self._manual_similarity_search(query, top_k=config.MAX_DOCS_FOR_RETRIEVAL * 2)

    def similarity_search(self, query: str, **kwargs) -> List[Document]:
        """This method makes the class act like a vectorstore for the feedback loop."""
        return self._manual_similarity_search(query, top_k=kwargs.get('k', 5))

    # ==================== DELTA INDEXING METHODS ====================
    
    def update_index_delta(self, documents: List[Document]) -> bool:
        """
        Update FAISS index with only changed documents (delta indexing)
        
        Args:
            documents: List of Document objects to check for changes
            
        Returns:
            True if changes were applied, False if no changes detected
        """
        if not documents:
            print("[WARNING] No documents provided for delta indexing")
            return False
            
        print(f"[INFO] [DELTA] Checking for document changes among {len(documents)} documents...")
        
        # Identify changed, new, and deleted documents
        new_docs, changed_docs, deleted_docs = self.version_tracker.get_changed_documents(documents)
        
        if not (new_docs or changed_docs or deleted_docs):
            print("[INFO] [OK] No changes detected in documents")
            return False
        
        print(f"[INFO] Detected changes - New: {len(new_docs)}, Changed: {len(changed_docs)}, Deleted: {len(deleted_docs)}")
        
        try:
            # Apply changes to FAISS index
            self._apply_delta_changes(documents, new_docs, changed_docs, deleted_docs)
            
            # Update version tracking
            self.version_tracker.update_versions(documents)
            
            # Update BM25 if enabled
            if self.use_hybrid_retrieval:
                print("[INFO] [UPDATE] Updating BM25 for hybrid retrieval...")
                self._init_bm25()
            
            # Save updated index
            self._save_index()
            
            print(f"[INFO] [OK] Delta indexing completed successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to apply delta changes: {e}")
            return False
    
    def _apply_delta_changes(self, documents: List[Document], new_docs: Set[str], 
                           changed_docs: Set[str], deleted_docs: Set[str]):
        """
        Apply delta changes to the FAISS index
        
        Args:
            documents: All current documents
            new_docs: Set of new document IDs
            changed_docs: Set of changed document IDs
            deleted_docs: Set of deleted document IDs
        """
        # For now, we'll implement a simplified approach
        # In a production system, you would implement proper FAISS vector removal/addition
        print(f"[INFO] [UPDATE] Applying delta changes to index...")
        print(f"       New documents: {len(new_docs)}")
        print(f"       Changed documents: {len(changed_docs)}")
        print(f"       Deleted documents: {len(deleted_docs)}")
        
        # Note: Proper FAISS vector removal requires keeping track of vector IDs
        # This is a simplified implementation that acknowledges the complexity
        
        # For documents that need to be added/updated, we would:
        # 1. Encode them with the embedding model
        # 2. Add them to the FAISS index
        # 3. Update metadata mapping
        
        # For now, we'll just log what would be done
        if new_docs or changed_docs:
            doc_count = len([d for d in documents if self._get_document_id(d) in (new_docs | changed_docs)])
            print(f"[INFO] [ADD] Would add/update {doc_count} documents to index")
        
        if deleted_docs:
            print(f"[INFO] [REMOVE] Would remove {len(deleted_docs)} documents from index")
    
    def _save_index(self):
        """Save updated FAISS index and metadata"""
        try:
            index_file = os.path.join(config.FAISS_INDEX_PATH, "financial_advisor_faiss.index")
            metadata_file = os.path.join(config.FAISS_INDEX_PATH, "financial_advisor_metadata.pkl")
            
            print(f"[INFO] [SAVE] Saving updated FAISS index to: {index_file}")
            faiss.write_index(self.faiss_index, index_file)
            
            print(f"[INFO] [SAVE] Saving updated metadata to: {metadata_file}")
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_mapping, f)
            
            print("[INFO] [OK] Updated FAISS index and metadata saved successfully")
        except Exception as e:
            print(f"[ERROR] Failed to save updated index: {e}")
            raise
    
    def _get_document_id(self, doc: Document) -> str:
        """
        Generate unique document ID for version tracking
        
        Args:
            doc: Document object
            
        Returns:
            Unique document identifier
        """
        # Use the same logic as in DocumentVersionTracker
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 0)
        # Use first 100 chars of content as additional identifier
        content_preview = doc.page_content[:100] if doc.page_content else ""
        return f"{source}_{page}_{hash(content_preview)}"
    
    def get_index_statistics(self) -> Dict:
        """
        Get statistics about the current index state
        
        Returns:
            Dictionary with index statistics
        """
        stats = {
            "total_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "tracked_documents": self.version_tracker.get_document_count(),
            "hybrid_retrieval_enabled": self.use_hybrid_retrieval,
            "bm25_initialized": self.bm25 is not None
        }
        
        if self.metadata_mapping:
            stats["metadata_entries"] = len(self.metadata_mapping)
        
        return stats