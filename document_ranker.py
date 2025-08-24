import time
import re
from typing import List, Dict
from langchain.schema import Document
from sentence_transformers import CrossEncoder

from config import config

class DocumentRanker:
    """Handles document ranking and re-ranking using various methods"""
    
    def __init__(self):
        self.reranker = None
        self.reranker_name = None
        self._init_cross_encoder()
    
    def _init_cross_encoder(self):
        """Initialize Cross-Encoder for advanced re-ranking"""
        print("[INFO] Loading Cross-Encoder for document re-ranking...")
        self.reranker = None
        self.reranker_name = None
        
        # Try to load the primary cross-encoder model
        primary_model = config.CROSS_ENCODER_MODEL
        print(f"[INFO] Attempting to load primary cross-encoder model: {primary_model}")
        
        try:
            # Load with timeout and performance settings
            start_time = time.time()
            self.reranker = CrossEncoder(
                primary_model,
                max_length=512,  # Set appropriate max length for financial documents
                trust_remote_code=True  # Required for some models
            )
            load_time = time.time() - start_time
            self.reranker_name = primary_model
            print(f"[INFO] ✅ Cross-Encoder '{primary_model}' loaded successfully in {load_time:.2f}s")
        except Exception as e:
            print(f"[WARNING] Failed to load primary Cross-Encoder '{primary_model}': {e}")
            
            # If fallback is enabled, try alternative models
            if config.ENABLE_CROSS_ENCODER_FALLBACK:
                print("[INFO] Attempting to load fallback cross-encoder models...")
                for model_name, model_path in config.CROSS_ENCODER_MODELS.items():
                    if model_path == primary_model:
                        continue  # Skip the one we already tried
                        
                    try:
                        print(f"[INFO] Attempting to load fallback model: {model_path}")
                        start_time = time.time()
                        self.reranker = CrossEncoder(
                            model_path,
                            max_length=512,
                            trust_remote_code=True
                        )
                        load_time = time.time() - start_time
                        self.reranker_name = model_path
                        print(f"[INFO] ✅ Fallback Cross-Encoder '{model_path}' loaded successfully in {load_time:.2f}s")
                        break
                    except Exception as fallback_e:
                        print(f"[WARNING] Failed to load fallback Cross-Encoder '{model_path}': {fallback_e}")
                        continue
        
        if self.reranker is not None:
            print(f"[INFO] ✅ Cross-Encoder re-ranking enabled with model: {self.reranker_name}")
        else:
            print("[INFO] ⚠️ Cross-Encoder re-ranking disabled - falling back to simple re-ranking...")

    def rank_and_filter(self, docs: List[Document], query: str) -> List[Document]:
        """Advanced re-ranking using Cross-Encoder for semantic relevance"""
        if not docs:
            return docs
        
        # Use Cross-Encoder if available
        if self.reranker is not None:
            return self._cross_encoder_rerank(docs, query)
        else:
            # Fallback to improved lexical matching
            return self._lexical_rerank(docs, query)
    
    def _is_form_field_or_template(self, content: str) -> bool:
        """Detect if content is just a form field or template rather than informative text"""
        content_lower = content.lower().strip()
        
        # Check for common form field patterns with enhanced financial-specific patterns
        form_patterns = [
            r':\s*\.{3,}',                      # ": ..."
            r':\s*_{3,}',                       # ": ___"
            r':\s*\d+\.\s*',
            # Financial form-specific patterns
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
            # "Passport No."
            r'^\s*(name|address|phone|email|passport|tin|nid|bin|vat)?\s*:\s*(if any)?\s*\d*\.?\s*'
        ]
        
        for pattern in form_patterns:
            if re.search(pattern, content_lower):
                print(f"[DEBUG] Form pattern matched: {pattern}")
                return True
        
        # Check if content is too short and uninformative
        if len(content.strip()) < 30 and ':' in content:
            print(f"[DEBUG] Content too short and contains colon: {len(content.strip())} chars")
            return True
            
        return False

    def _cross_encoder_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Re-rank documents using Cross-Encoder model with timing and debug logging"""
        if not docs or not self.reranker:
            return docs
        
        print(f"[INFO] 🔍 Re-ranking {len(docs)} documents with Cross-Encoder ({self.reranker_name})")
        start_time = time.time()
        
        try:
            # Prepare query-document pairs for scoring
            pairs = []
            valid_docs = []
            invalid_docs_count = 0
            
            for doc in docs:
                content = doc.page_content.strip()
                
                # Skip form fields and templates
                if self._is_form_field_or_template(content):
                    invalid_docs_count += 1
                    continue
                
                # Skip very short or empty content
                if len(content) < 50:
                    invalid_docs_count += 1
                    continue
                
                pairs.append([query, content])
                valid_docs.append(doc)
            
            if invalid_docs_count > 0:
                print(f"[INFO] ⚠️  Skipped {invalid_docs_count} invalid documents (form fields/short content)")
            
            if not pairs:
                print("[INFO] ⚠️ No valid documents after filtering form fields")
                return docs[:config.MAX_DOCS_FOR_CONTEXT]  # Return original docs as fallback
            
            # Get relevance scores with timing
            predict_start = time.time()
            scores = self.reranker.predict(
                pairs,
                batch_size=config.CROSS_ENCODER_BATCH_SIZE,
                convert_to_numpy=True
            )
            predict_time = time.time() - predict_start
            
            # Combine documents with scores and sort
            doc_scores = list(zip(valid_docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Log top scores for debugging
            print(f"[DEBUG] Top 5 Cross-Encoder scores: {[f'{score:.3f}' for _, score in doc_scores[:5]][:5]}")
            
            # Filter by relevance threshold and limit
            filtered_docs = []
            below_threshold_count = 0
            
            for doc, score in doc_scores:
                if score >= config.RELEVANCE_THRESHOLD and len(filtered_docs) < config.MAX_DOCS_FOR_CONTEXT:
                    filtered_docs.append(doc)
                    print(f"[INFO] ✅ Document relevance score: {score:.3f}")
                else:
                    below_threshold_count += 1
            
            if below_threshold_count > 0:
                print(f"[INFO] ⚠️  {below_threshold_count} documents below relevance threshold ({config.RELEVANCE_THRESHOLD})")
            
            if not filtered_docs:
                print(f"[INFO] ⚠️ No documents above relevance threshold ({config.RELEVANCE_THRESHOLD})")
                # Return top documents even if below threshold, but limit to 2
                filtered_docs = [doc for doc, _ in doc_scores[:2]]
            
            rerank_time = time.time() - start_time
            print(f"[INFO] ✅ Cross-Encoder re-ranking completed in {rerank_time:.2f}s "
                  f"(prediction: {predict_time:.2f}s)")
            
            return filtered_docs
            
        except Exception as e:
            print(f"[WARNING] Cross-encoder re-ranking failed: {e}")
            rerank_time = time.time() - start_time
            print(f"[INFO] Cross-encoder re-ranking failed after {rerank_time:.2f}s")
            return docs[:config.MAX_DOCS_FOR_CONTEXT]

    def _lexical_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Fallback lexical re-ranking when Cross-Encoder is not available"""
        if not docs:
            return docs
        
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
            
            # Calculate simple overlap score
            overlap = len(query_words.intersection(content_words))
            total_query_words = len(query_words)
            
            if total_query_words > 0:
                score = overlap / total_query_words
            else:
                score = 0
            
            # Boost score for exact phrase matches
            if query_lower in content_lower:
                score += 0.5
            
            scored_docs.append((doc, score))
        
        # Sort by score and return top documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by a minimum score and limit
        min_score = 0.1  # Lower threshold for lexical matching
        filtered_docs = []
        
        for doc, score in scored_docs:
            if score >= min_score and len(filtered_docs) < config.MAX_DOCS_FOR_CONTEXT:
                filtered_docs.append(doc)
                print(f"[INFO] ✅ Document lexical score: {score:.3f}")
        
        if not filtered_docs and scored_docs:
            # If no docs meet threshold, return the best one
            filtered_docs = [scored_docs[0][0]]
            print(f"[INFO] ⚠️ Using best document despite low score: {scored_docs[0][1]:.3f}")
        
        return filtered_docs

    def prepare_docs(self, docs: List[Document]) -> List[Document]:
        """Prepare documents by truncating content if needed"""
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