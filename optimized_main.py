import os
import re
import logging
import time
import gc
import torch
from typing import Dict, List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from sentence_transformers import CrossEncoder
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache

# Import enhanced RAG utilities
from enhanced_rag_utils import EnhancedRAGUtils

# --- M1 Optimized Configuration ---
# Force MPS (Metal Performance Shaders) for M1 if available
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("[INFO] 🚀 Using Apple Silicon MPS acceleration")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("[INFO] 🚀 Using CUDA acceleration")
else:
    DEVICE = "cpu"
    print("[INFO] 💻 Using CPU")

# Memory-optimized settings for 8GB RAM
torch.set_num_threads(4)  # Limit CPU threads for M1
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
os.environ["OMP_NUM_THREADS"] = "4"

# --- Logging ---
logging.basicConfig(
    filename='logs/telegram_financial_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Optimized Config ---
FAISS_INDEX_PATH = "faiss_index"
# Use smaller, faster embedding model for M1
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller, faster model
GROQ_MODEL = "llama3-8b-8192"  # Fast Groq model
GROQ_API_KEY = "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"
CACHE_TTL = 86400  # 24 hours

# Memory-optimized retrieval settings
MAX_DOCS_FOR_RETRIEVAL = 8  # Reduced for memory efficiency
MAX_DOCS_FOR_CONTEXT = 4   # Reduced for memory efficiency
CONTEXT_CHUNK_SIZE = 1200   # Reduced chunk size

# Enhanced Cross-Encoder Configuration
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-2-v1'  # Smaller, faster model
RELEVANCE_THRESHOLD = 0.15  # Adjusted threshold
SEMANTIC_SIMILARITY_THRESHOLD = 0.3  # For semantic filtering

# Query refinement settings
MAX_QUERY_REFINEMENT_ATTEMPTS = 2
QUERY_EXPANSION_TERMS = 3

# --- Enhanced Prompt Template ---
PROMPT_TEMPLATE = """
You are an expert financial advisor specializing in Bangladesh's banking and financial services.
Provide accurate, practical advice based on the given context.

CRITICAL INSTRUCTIONS:
- Answer directly and conversationally
- Focus on actionable information
- Use Bangladeshi Taka (৳/Tk) for currency
- If information is insufficient, clearly state limitations
- Prioritize recent and official information
- Be concise but comprehensive

Context Information:
{context}

User Question: {input}

Expert Answer:"""

QA_PROMPT = PromptTemplate(input_variables=["context", "input"], template=PROMPT_TEMPLATE)

# --- Memory-Efficient Cache ---
class OptimizedResponseCache:
    def __init__(self, ttl=CACHE_TTL, max_size=100):
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}

    def get(self, query):
        entry = self.cache.get(query)
        if entry and time.time() - entry["time"] < self.ttl:
            self.access_times[query] = time.time()
            return entry["response"]
        elif entry:
            # Remove expired entry
            del self.cache[query]
            if query in self.access_times:
                del self.access_times[query]
        return None

    def set(self, query, response):
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_query = min(self.access_times.keys(), 
                             key=lambda k: self.access_times[k])
            del self.cache[oldest_query]
            del self.access_times[oldest_query]
            gc.collect()  # Force garbage collection
        
        self.cache[query] = {"response": response, "time": time.time()}
        self.access_times[query] = time.time()

# --- Enhanced Query Processor ---
class EnhancedQueryProcessor:
    def __init__(self):
        # Financial domain keywords for better categorization
        self.categories = {
            "taxation": ["tax", "vat", "income tax", "return", "nbr", "tin", "withholding"],
            "loans": ["loan", "credit", "mortgage", "financing", "installment", "emi", "interest rate"],
            "investment": ["investment", "bond", "savings", "certificate", "profit", "dividend", "portfolio"],
            "banking": ["bank", "account", "deposit", "withdrawal", "transfer", "atm", "card", "branch"],
            "insurance": ["insurance", "policy", "premium", "claim", "coverage", "life insurance"],
            "business": ["business", "startup", "license", "registration", "trade", "commerce", "sme"],
            "foreign_exchange": ["forex", "foreign exchange", "remittance", "dollar", "currency", "export", "import"]
        }
    
    def process(self, query: str) -> str:
        """Enhanced query categorization with confidence scoring"""
        q = query.lower()
        category_scores = {}
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in q)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.keys(), key=lambda k: category_scores[k])
        return "general"

    @lru_cache(maxsize=128)
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key financial terms from query"""
        # Remove common stop words and extract meaningful terms
        stop_words = {"what", "how", "when", "where", "why", "is", "are", "can", "do", "does", "the", "a", "an"}
        terms = [word.lower() for word in query.split() if word.lower() not in stop_words and len(word) > 2]
        return terms[:5]  # Limit to top 5 terms

# --- Optimized Financial Advisor Bot ---
class OptimizedFinancialAdvisorBot:
    def __init__(self):
        self.cache = OptimizedResponseCache()
        self.processor = EnhancedQueryProcessor()
        self.rag_utils = EnhancedRAGUtils()
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit threads for memory
        self._init_rag()

    def _init_rag(self):
        print("[INFO] 🚀 Initializing optimized RAG system for M1...")
        logger.info("Initializing optimized FAISS + LLM...")

        # Initialize embeddings with M1 optimization
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": DEVICE,
                "trust_remote_code": True
            },
            encode_kwargs={
                "normalize_embeddings": True,  # Better for similarity search
                "batch_size": 16  # Smaller batch size for memory efficiency
            }
        )
        
        print(f"[INFO] Loading FAISS index from: {FAISS_INDEX_PATH}")
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("[INFO] ✅ FAISS index loaded successfully.")

        # Initialize Groq LLM with optimized settings
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.4,  # Slightly more creative
            max_tokens=1000,  # Reduced for memory
            model_kwargs={
                "top_p": 0.85,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        )
        print(f"[INFO] ✅ Groq LLM initialized with model: {GROQ_MODEL}")

        # Initialize lightweight Cross-Encoder
        print("[INFO] Loading optimized Cross-Encoder...")
        try:
            self.reranker = CrossEncoder(
                CROSS_ENCODER_MODEL,
                max_length=256,  # Reduced for memory efficiency
                device=DEVICE if DEVICE != "mps" else "cpu"  # MPS not supported by all models
            )
            print("[INFO] ✅ Cross-Encoder loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load Cross-Encoder: {e}")
            self.reranker = None

        # Create optimized chains
        self.doc_chain = create_stuff_documents_chain(self.llm, QA_PROMPT)
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": MAX_DOCS_FOR_RETRIEVAL,
                "score_threshold": 0.1  # Filter out very irrelevant docs
            }
        )
        self.qa_chain = create_retrieval_chain(retriever, self.doc_chain)

    def _enhanced_semantic_filter(self, docs: List[Document], query: str) -> List[Document]:
        """Enhanced semantic filtering using embeddings"""
        if not docs or len(docs) <= 2:
            return docs
        
        try:
            # Get query embedding
            query_embedding = self.vectorstore.embeddings.embed_query(query)
            
            # Calculate semantic similarities
            filtered_docs = []
            for doc in docs:
                # Get document embedding (approximate using first 200 chars)
                doc_text = doc.page_content[:200]
                doc_embedding = self.vectorstore.embeddings.embed_query(doc_text)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                if similarity > SEMANTIC_SIMILARITY_THRESHOLD:
                    filtered_docs.append(doc)
            
            return filtered_docs if filtered_docs else docs[:2]  # Fallback to top 2
            
        except Exception as e:
            print(f"[WARNING] Semantic filtering failed: {e}")
            return docs

    def _advanced_rerank_and_filter(self, docs: List[Document], query: str) -> List[Document]:
        """Advanced re-ranking with multiple strategies"""
        if not docs:
            return docs
        
        # Step 1: Filter out form fields and templates
        informative_docs = self._filter_form_fields(docs)
        
        # Step 2: Apply semantic filtering
        semantic_filtered = self._enhanced_semantic_filter(informative_docs, query)
        
        # Step 3: Cross-encoder re-ranking if available
        if self.reranker is not None:
            return self._cross_encoder_rerank(semantic_filtered, query)
        else:
            return self._enhanced_lexical_rerank(semantic_filtered, query)

    def _filter_form_fields(self, docs: List[Document]) -> List[Document]:
        """Enhanced form field detection and filtering"""
        informative_docs = []
        
        for doc in docs:
            content = doc.page_content.strip()
            
            # Enhanced form field patterns
            form_patterns = [
                r':\s*\.{3,}',  # ": ..."
                r':\s*_{3,}',   # ": ___"
                r':\s*\d+\.\s*$',  # ": 5."
                r'\(if any\):\s*\d*',
                r'^\w+\s+(no|number)\.?\s*:?\s*$',
                r'signature\s*:?\s*$',
                r'date\s*:?\s*$',
                r'name\s*:?\s*$',
                r'^[A-Z\s]+:\s*$'  # All caps labels
            ]
            
            is_form_field = False
            for pattern in form_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    is_form_field = True
                    break
            
            # Additional checks
            if not is_form_field:
                # Check content quality
                words = content.split()
                if len(words) < 10:  # Too short
                    is_form_field = True
                elif content.count(':') > len(words) * 0.3:  # Too many colons
                    is_form_field = True
                elif len([w for w in words if w.isalpha()]) < len(words) * 0.5:  # Too few alphabetic words
                    is_form_field = True
            
            if not is_form_field:
                informative_docs.append(doc)
        
        return informative_docs if informative_docs else docs[:2]  # Keep at least 2 docs

    def _cross_encoder_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Optimized cross-encoder re-ranking"""
        if not docs:
            return docs
        
        try:
            print(f"[INFO] 🔄 Re-ranking {len(docs)} documents with Cross-Encoder...")
            
            # Prepare pairs with truncated content for efficiency
            pairs = []
            for doc in docs:
                content = doc.page_content[:500]  # Truncate for efficiency
                pairs.append([query, content])
            
            # Get relevance scores
            scores = self.reranker.predict(pairs)
            
            # Combine and sort
            scored_docs = list(zip(docs, scores))
            ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
            # Filter by relevance threshold
            filtered_docs = [doc for doc, score in ranked_docs if score > RELEVANCE_THRESHOLD]
            
            result = filtered_docs[:MAX_DOCS_FOR_CONTEXT]
            print(f"[INFO] ✅ Kept {len(result)} highly relevant documents")
            return result
            
        except Exception as e:
            print(f"[WARNING] Cross-encoder re-ranking failed: {e}")
            return self._enhanced_lexical_rerank(docs, query)

    def _enhanced_lexical_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Enhanced lexical re-ranking with TF-IDF-like scoring"""
        if not docs:
            return docs
        
        query_terms = set(self.processor.extract_key_terms(query))
        scored_docs = []
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            words = content_lower.split()
            
            # Calculate various relevance signals
            exact_matches = sum(1 for term in query_terms if term in content_lower)
            partial_matches = sum(1 for term in query_terms 
                                if any(term in word for word in words))
            
            # Position-based scoring (earlier mentions are more important)
            position_score = 0
            for i, word in enumerate(words[:50]):  # Check first 50 words
                if any(term in word for term in query_terms):
                    position_score += 1 / (i + 1)  # Higher score for earlier positions
            
            # Length normalization
            length_penalty = min(1.0, len(words) / 100)  # Prefer moderate length docs
            
            # Final score
            total_score = (exact_matches * 3 + partial_matches * 1.5 + 
                          position_score * 2) * length_penalty
            
            if total_score > 0:
                scored_docs.append((doc, total_score))
        
        # Sort and return top documents
        ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:MAX_DOCS_FOR_CONTEXT]]

    def _prepare_optimized_context(self, docs: List[Document]) -> List[Document]:
        """Prepare context with intelligent truncation"""
        processed = []
        total_length = 0
        max_total_length = CONTEXT_CHUNK_SIZE * MAX_DOCS_FOR_CONTEXT
        
        for doc in docs:
            content = doc.page_content
            
            # Intelligent truncation - keep important parts
            if len(content) > CONTEXT_CHUNK_SIZE:
                # Try to keep complete sentences
                sentences = content.split('. ')
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence) <= CONTEXT_CHUNK_SIZE:
                        truncated += sentence + ". "
                    else:
                        break
                content = truncated.strip() + "...[truncated]"
            
            if total_length + len(content) <= max_total_length:
                processed.append(Document(page_content=content, metadata=doc.metadata))
                total_length += len(content)
            else:
                break
        
        return processed

    async def process_query_async(self, query: str) -> Dict:
        """Async query processing for better performance"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.process_query, query)

    def process_query(self, query: str) -> Dict:
        """Enhanced query processing with multiple optimization strategies"""
        category = self.processor.process(query)
        logger.info(f"Processing query (category={category}): {query}")
        print(f"[INFO] 🔍 Query: {query} | Category: {category}")

        # Check cache first
        cached = self.cache.get(query)
        if cached:
            print("[INFO] ⚡ Cache hit - returning stored response")
            return cached

        try:
            # Step 1: Multi-stage query refinement
            refined_queries = self.rag_utils.multi_stage_query_refinement(query, category)
            print(f"[INFO] 🔧 Generated {len(refined_queries)} refined queries")

            # Step 2: Retrieve documents using best refined query
            best_docs = []
            best_relevance = 0
            
            for refined_query in refined_queries[:2]:  # Try top 2 refined queries
                retrieved = self.vectorstore.similarity_search(refined_query, k=MAX_DOCS_FOR_RETRIEVAL)
                
                # Check relevance
                is_relevant, confidence = self.rag_utils.enhanced_relevance_check(
                    refined_query, retrieved, category
                )
                
                if confidence > best_relevance:
                    best_relevance = confidence
                    best_docs = retrieved
                    best_query = refined_query
            
            print(f"[INFO] 📊 Best relevance score: {best_relevance:.3f}")
            
            # Step 3: Enhanced relevance filtering
            if best_relevance < 0.15:  # Very low relevance threshold
                return {
                    "response": "I couldn't find relevant information in my database for your query. Please try rephrasing your question or ask about a different financial topic.",
                    "sources": [],
                    "contexts": []
                }

            # Step 4: Advanced document filtering and re-ranking
            filtered_docs = self._advanced_rerank_and_filter(best_docs, best_query)
            
            if not filtered_docs:
                return {
                    "response": "No relevant documents found after filtering. Please try a more specific question.",
                    "sources": [],
                    "contexts": []
                }

            # Step 5: Prepare optimized context
            context_docs = self._prepare_optimized_context(filtered_docs)

            # Step 6: Generate answer with enhanced validation
            print("[INFO] 🤖 Generating answer with LLM...")
            result = self.qa_chain.invoke({"input": best_query})
            answer = result.get("answer") or result.get("result") or str(result)

            # Step 7: Enhanced answer validation
            context_texts = [d.page_content for d in context_docs]
            validation = self.rag_utils.comprehensive_answer_validation(
                query, answer, context_texts, category
            )
            
            print(f"[INFO] ✅ Answer validation - Valid: {validation['valid']}, "
                  f"Confidence: {validation['confidence']:.3f}")

            # Step 8: Apply answer enhancement if needed
            if validation['confidence'] < 0.4:
                answer = self.rag_utils.enhance_answer(answer, context_texts, query)

            # Step 9: Prepare response
            response = {
                "response": answer,
                "sources": [{"file": d.metadata.get("source", "Unknown")} for d in context_docs],
                "contexts": context_texts,
                "relevance_score": best_relevance,
                "validation": validation
            }

            # Cache the response
            self.cache.set(query, response)
            
            # Memory cleanup
            gc.collect()
            
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"[ERROR] {e}")
            return {
                "response": f"I encountered an error while processing your query. Please try again or rephrase your question.",
                "sources": [],
                "contexts": []
            }

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# --- Telegram Handlers ---
bot_instance = OptimizedFinancialAdvisorBot()

async def send_in_chunks(update: Update, text: str, max_length: int = 4000):
    """Send long messages in chunks"""
    for i in range(0, len(text), max_length):
        chunk = text[i:i+max_length]
        await update.message.reply_text(chunk)
        await asyncio.sleep(0.1)  # Small delay to avoid rate limits

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = """
🏦 **Bangladesh Financial Advisor Bot** 🇧🇩

I'm your AI financial advisor specializing in Bangladesh's banking and financial services.

**What I can help with:**
• Banking services and account opening
• Loans and credit information
• Investment opportunities
• Tax and VAT guidance
• Insurance policies
• Business startup guidance
• Foreign exchange regulations

**Tips for better results:**
• Be specific in your questions
• Mention relevant amounts or timeframes
• Ask about particular banks or services

Type your financial question to get started! 💰
    """
    await update.message.reply_text(welcome_msg)

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text.strip()
    if not user_query:
        await update.message.reply_text("Please enter a valid financial question.")
        return

    print(f"[INFO] 👤 User query: {user_query}")
    
    # Send processing message
    processing_msg = await update.message.reply_text("🔍 Analyzing your question...")
    
    try:
        # Process query asynchronously
        response = await bot_instance.process_query_async(user_query)
        
        # Delete processing message
        await processing_msg.delete()
        
        # Send main answer
        answer = response.get("response", "No response generated")
        await send_in_chunks(update, f"💡 **Answer:**\n{answer}")
        
        # Send additional info if available
        if isinstance(response, dict) and response.get("sources"):
            relevance_score = response.get("relevance_score", 0)
            validation = response.get("validation", {})
            
            # Send metadata
            metadata_msg = f"""
📊 **Query Analysis:**
• Relevance Score: {relevance_score:.2f}
• Answer Confidence: {validation.get('confidence', 0):.2f}
• Sources Found: {len(response['sources'])}
            """
            await update.message.reply_text(metadata_msg)
            
            # Send source information
            if response.get("contexts"):
                sources_msg = "📚 **Source Documents:**\n"
                for i, (source, context) in enumerate(zip(response["sources"], response["contexts"]), 1):
                    filename = source.get("file", "Unknown")
                    preview = context[:150] + "..." if len(context) > 150 else context
                    sources_msg += f"\n**{i}. {filename}**\n{preview}\n"
                
                await send_in_chunks(update, sources_msg)

    except Exception as e:
        await processing_msg.delete()
        await update.message.reply_text(f"❌ Error processing your query: {str(e)}")
        logger.error(f"Handler error: {e}")

    print("[INFO] ✅ Response sent to user")

# --- Main Execution ---
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN", "7596897324:AAG3TsT18amwRF2nRBcr1JS6NdGs96Ie-D0")
    
    print("[INFO] 🚀 Starting Optimized Financial Advisor Bot for M1...")
    print(f"[INFO] 💻 Device: {DEVICE}")
    print(f"[INFO] 🧠 Embedding Model: {EMBEDDING_MODEL}")
    print(f"[INFO] 🤖 LLM Model: {GROQ_MODEL}")
    
    app = ApplicationBuilder().token(token).build()
    logger.info("Optimized bot started successfully")
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    
    print("[INFO] ✅ Bot is ready and polling for messages...")
    app.run_polling()
