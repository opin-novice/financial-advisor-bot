import os
import re
import logging
import time
from typing import Dict, List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from sentence_transformers import CrossEncoder

# --- Logging ---
logging.basicConfig(
    filename='logs/telegram_financial_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Config ---
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_MODEL = "gemma3n:e4b"
CACHE_TTL = 86400  # 24 hours

# Retrieval Settings
MAX_DOCS_FOR_RETRIEVAL = 12
MAX_DOCS_FOR_CONTEXT = 5
CONTEXT_CHUNK_SIZE = 1500

# Cross-Encoder Re-ranking Configuration
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RELEVANCE_THRESHOLD = 0.2     # Minimum relevance score to keep documents (adjusted for BGE-M3)

# Hybrid Re-ranking Configuration
SEMANTIC_WEIGHT = 0.7         # Weight for Cross-Encoder semantic scoring
LEXICAL_WEIGHT = 0.3          # Weight for lexical scoring
PHRASE_BONUS_MULTIPLIER = 2   # Multiplier for consecutive word matches
LENGTH_BONUS_MULTIPLIER = 0.5 # Multiplier for document length bonus

# --- Prompt (Bangladesh Context) ---
PROMPT_TEMPLATE = """
You are a helpful financial advisor specializing in Bangladesh's banking and financial services.
Always respond in a natural, conversational tone as if speaking to a friend.

IMPORTANT INSTRUCTIONS:
- Answer based on the provided information
- Ignore form fields, blank templates, placeholder text, and incomplete document fragments
- Never say "According to the context" - just answer directly
- If you don't have enough information, say "I don't have specific information about that"
- Use Bangladeshi Taka (৳/Tk) as currency
- Be concise and practical

Context Information:
{context}

Question: {input}

Answer:"""
QA_PROMPT = PromptTemplate(input_variables=["context", "input"], template=PROMPT_TEMPLATE)

# --- Cache ---
class ResponseCache:
    def __init__(self, ttl=CACHE_TTL):
        self.ttl = ttl
        self.cache = {}

    def get(self, query):
        entry = self.cache.get(query)
        if entry and time.time() - entry["time"] < self.ttl:
            return entry["response"]
        return None

    def set(self, query, response):
        self.cache[query] = {"response": response, "time": time.time()}

# --- Query Categorizer ---
class QueryProcessor:
    def process(self, query: str) -> str:
        q = query.lower()
        if "tax" in q: return "taxation"
        if "loan" in q: return "loans"
        if "investment" in q: return "investment"
        if "bank" in q: return "banking"
        return "general"

# --- Financial Advisor Bot ---
class FinancialAdvisorTelegramBot:
    def __init__(self):
        self.cache = ResponseCache()
        self.processor = QueryProcessor()
        self._init_rag()

    def _init_rag(self):
        print("[INFO] Initializing FAISS and LLM...")
        logger.info("Initializing FAISS + LLM...")

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        print(f"[INFO] Loading FAISS index from: {FAISS_INDEX_PATH}")
        self.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("[INFO] ✅ FAISS index loaded successfully.")

        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            temperature=0.5,
            max_tokens=1200,
            top_p=0.9,
            repeat_penalty=1.1
        )
        print(f"[INFO] ✅ Ollama LLM initialized with model: {OLLAMA_MODEL}")

        # Initialize Cross-Encoder for advanced re-ranking
        print("[INFO] Loading Cross-Encoder for document re-ranking...")
        try:
            self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
            print("[INFO] ✅ Cross-Encoder loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load Cross-Encoder: {e}")
            print("[INFO] Falling back to simple re-ranking...")
            self.reranker = None

        self.doc_chain = create_stuff_documents_chain(self.llm, QA_PROMPT)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": MAX_DOCS_FOR_RETRIEVAL})
        self.qa_chain = create_retrieval_chain(retriever, self.doc_chain)

    def _rank_and_filter(self, docs: List[Document], query: str) -> List[Document]:
        """Advanced re-ranking using hybrid approach combining Cross-Encoder and lexical scoring"""
        if not docs:
            return docs
        
        # Use hybrid re-ranking if Cross-Encoder is available
        if self.reranker is not None:
            return self._hybrid_rerank(docs, query)
        else:
            # Fallback to improved lexical matching
            return self._lexical_rerank(docs, query)
    
    def _hybrid_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Hybrid re-ranking combining Cross-Encoder semantic scoring with lexical features"""
        if not docs:
            return docs
        
        try:
            # First, filter out form fields and templates
            informative_docs = []
            for doc in docs:
                if not self._is_form_field_or_template(doc.page_content):
                    informative_docs.append(doc)
                else:
                    print(f"[INFO] 🚫 Filtered out form field: {doc.page_content[:50]}...")
            
            if not informative_docs:
                print("[WARNING] All documents were filtered as form fields. Using original documents.")
                informative_docs = docs
            
            # Log filtering statistics
            self._log_reranking_stats(len(docs), len(informative_docs), "Form Field Filtering")
            
            print(f"[INFO] Re-ranking {len(informative_docs)} informative documents using Hybrid approach...")
            
            # Get Cross-Encoder scores
            pairs = []
            for doc in informative_docs:
                content = doc.page_content[:1000]
                pairs.append([query, content])
            
            cross_encoder_scores = self.reranker.predict(pairs)
            
            # Get lexical scores
            lexical_scores = self._get_lexical_scores(informative_docs, query)
            
            # Combine scores with weights
            combined_scores = []
            for i, doc in enumerate(informative_docs):
                # Normalize scores to 0-1 range
                ce_score = max(0, cross_encoder_scores[i])  # Cross-encoder scores are usually positive
                lex_score = lexical_scores[i] / max(lexical_scores) if max(lexical_scores) > 0 else 0
                
                # Weighted combination (70% semantic, 30% lexical)
                combined_score = SEMANTIC_WEIGHT * ce_score + LEXICAL_WEIGHT * lex_score
                combined_scores.append((doc, combined_score))
            
            # Sort by combined score
            ranked_docs = [doc for doc, _ in sorted(combined_scores, key=lambda x: x[1], reverse=True)]
            
            # Filter by minimum relevance threshold
            filtered_docs = [doc for doc, score in combined_scores if score > RELEVANCE_THRESHOLD]
            
            # Log final statistics
            self._log_reranking_stats(len(informative_docs), len(filtered_docs), "Hybrid Re-ranking")
            
            print(f"[INFO] ✅ Hybrid re-ranking completed. Kept {len(filtered_docs)} relevant documents.")
            return filtered_docs[:MAX_DOCS_FOR_CONTEXT]
            
        except Exception as e:
            print(f"[WARNING] Hybrid re-ranking failed: {e}")
            return self._lexical_rerank(docs, query)
    
    def _get_lexical_scores(self, docs: List[Document], query: str) -> List[float]:
        """Calculate lexical similarity scores for documents"""
        query_terms = set(query.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        query_terms = query_terms - stop_words
        
        scores = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            exact_matches = sum(1 for t in query_terms if t in content_lower)
            partial_matches = sum(1 for t in query_terms if any(t in word for word in content_lower.split()))
            
            # Phrase matching
            query_words = query.lower().split()
            phrase_bonus = 0
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase in content_lower:
                    phrase_bonus += PHRASE_BONUS_MULTIPLIER
            
            # Length bonus
            length_bonus = min(len(content_lower) / 1000, 1.0) * LENGTH_BONUS_MULTIPLIER
            
            score = (exact_matches * 3 + 
                    partial_matches * 1 + 
                    phrase_bonus + 
                    length_bonus)
            
            scores.append(score)
        
        return scores

    def _is_form_field_or_template(self, content: str) -> bool:
        """Detect if content is just a form field or template rather than informative text"""
        import re
        content_lower = content.lower().strip()
        
        # Check for common form field patterns
        form_patterns = [
            r':\s*\.{3,}',  # ": ..."
            r':\s*_{3,}',   # ": ___"
            r':\s*\d+\.\s*$',  # ": 5."
            r'\(if any\):\s*\d+',  # "(if any): 5"
            r'^\w+\s+no\.\s*$',  # "Passport No."
            r'^\s*(name|address|phone|email|passport|tin|nid|bin|vat)?\s*:\s*(if any)?\s*\d*\.?\s*$',
            r'^\s*[a-z\s]+\s*:\s*\.{3,}',  # "Field Name: ..."
            r'^\s*[a-z\s]+\s*:\s*_{3,}',   # "Field Name: ___"
            r'^\s*\d+\.\s*[a-z\s]*\s*:\s*\.{3,}',  # "1. Field: ..."
            r'^\s*\([^)]*\)\s*:\s*\.{3,}',  # "(Optional): ..."
            r'^\s*[a-z\s]+\s*\([^)]*\)\s*:\s*\.{3,}',  # "Field (if any): ..."
        ]
        
        for pattern in form_patterns:
            if re.search(pattern, content_lower):
                return True
        
        # Check if content is too short and uninformative
        if len(content.strip()) < 30 and ':' in content:
            return True
            
        # Check for repetitive dots, underscores, or numbers
        if content.count('.') > len(content) * 0.3 or content.count('_') > 5:
            return True
        
        # Check for common form field keywords
        form_keywords = ['signature', 'date', 'seal', 'stamp', 'official use only', 'for office use']
        if any(keyword in content_lower for keyword in form_keywords):
            return True
            
        return False

    def _log_reranking_stats(self, original_count: int, filtered_count: int, method: str):
        """Log re-ranking statistics for monitoring"""
        print(f"[INFO] 📊 Re-ranking Stats ({method}):")
        print(f"   - Original documents: {original_count}")
        print(f"   - After filtering: {filtered_count}")
        print(f"   - Filtered out: {original_count - filtered_count}")
        if original_count > 0:
            retention_rate = (filtered_count / original_count) * 100
            print(f"   - Retention rate: {retention_rate:.1f}%")

    def _cross_encoder_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Advanced semantic re-ranking using cross-encoder with form field filtering"""
        if not docs:
            return docs
        
        try:
            # First, filter out form fields and templates
            informative_docs = []
            for doc in docs:
                if not self._is_form_field_or_template(doc.page_content):
                    informative_docs.append(doc)
                else:
                    print(f"[INFO] 🚫 Filtered out form field: {doc.page_content[:50]}...")
            
            if not informative_docs:
                print("[WARNING] All documents were filtered as form fields. Using original documents.")
                informative_docs = docs
            
            print(f"[INFO] Re-ranking {len(informative_docs)} informative documents using Cross-Encoder...")
            
            # Create query-document pairs for cross-encoder
            pairs = []
            for doc in informative_docs:
                # Truncate document content to reasonable length for cross-encoder
                content = doc.page_content[:1000]  # Keep first 1000 chars for relevance scoring
                pairs.append([query, content])
            
            # Get relevance scores from cross-encoder
            scores = self.reranker.predict(pairs)
            
            # Combine documents with scores and sort by relevance
            scored_docs = list(zip(informative_docs, scores))
            ranked_docs = [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)]
            
            # Filter out documents with very low relevance scores
            filtered_docs = [doc for doc, score in scored_docs if score > RELEVANCE_THRESHOLD]
            
            print(f"[INFO] ✅ Cross-Encoder re-ranking completed. Kept {len(filtered_docs)} relevant documents.")
            return filtered_docs[:MAX_DOCS_FOR_CONTEXT]  # Return top documents
            
        except Exception as e:
            print(f"[WARNING] Cross-Encoder re-ranking failed: {e}")
            return self._lexical_rerank(docs, query)
    
    def _lexical_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Improved lexical re-ranking as fallback with enhanced scoring"""
        if not docs:
            return docs
        
        # Enhanced query preprocessing
        query_terms = set(query.lower().split())
        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        query_terms = query_terms - stop_words
        
        scored = []
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            # Skip form fields and templates
            if self._is_form_field_or_template(doc.page_content):
                continue
            
            # Enhanced scoring system
            exact_matches = sum(1 for t in query_terms if t in content_lower)
            partial_matches = sum(1 for t in query_terms if any(t in word for word in content_lower.split()))
            
            # Bonus for consecutive word matches (phrases)
            query_words = query.lower().split()
            phrase_bonus = 0
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase in content_lower:
                    phrase_bonus += PHRASE_BONUS_MULTIPLIER
            
            # Bonus for document length (prefer longer, more informative documents)
            length_bonus = min(len(content_lower) / 1000, 1.0) * LENGTH_BONUS_MULTIPLIER # Cap at 1.0
            
            # Calculate final score with weights
            score = (exact_matches * 3 + 
                    partial_matches * 1 + 
                    phrase_bonus + 
                    length_bonus)
            
            if score > 0:  # Only include documents with some relevance
                scored.append((doc, score))
        
        # Sort by score and return top documents
        ranked = [d for d, s in sorted(scored, key=lambda x: x[1], reverse=True)]
        return ranked[:MAX_DOCS_FOR_CONTEXT]

    def _prepare_docs(self, docs: List[Document]) -> List[Document]:
        processed = []
        for d in docs[:MAX_DOCS_FOR_CONTEXT]:
            content = d.page_content[:CONTEXT_CHUNK_SIZE] + ("...[truncated]" if len(d.page_content) > CONTEXT_CHUNK_SIZE else "")
            processed.append(Document(page_content=content, metadata=d.metadata))
        return processed

    def process_query(self, query: str) -> Dict:
        category = self.processor.process(query)
        logger.info(f"Processing query (category={category}): {query}")
        print(f"[INFO] 🔍 Received query: {query}")

        cached = self.cache.get(query)
        if cached:
            print("[INFO] ✅ Cache hit - returning stored response.")
            return cached

        try:
            retrieved = self.vectorstore.similarity_search(query, k=MAX_DOCS_FOR_RETRIEVAL)
            filtered = self._rank_and_filter(retrieved, query)
            if not filtered:
                print("[INFO] ❌ No relevant documents found.")
                return {"response": "I could not find relevant information in my database.", "sources": [], "contexts": []}

            docs = self._prepare_docs(filtered)

            print("[INFO] ✅ Running LLM to generate answer...")
            result = self.qa_chain.invoke({"input": query})

            answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
            print("[INFO] ✅ Answer generated successfully.")

            context_texts = [d.page_content for d in docs]

            response = {
                "response": answer,
                "sources": [{"file": d.metadata.get("source", "Unknown")} for d in docs],
                "contexts": context_texts
            }
            self.cache.set(query, response)
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"[ERROR] {e}")
            return {"response": f"Error: {e}", "sources": [], "contexts": []}

# --- Telegram Handlers ---
bot_instance = FinancialAdvisorTelegramBot()

async def send_in_chunks(update: Update, text: str):
    MAX_LEN = 4000
    for i in range(0, len(text), MAX_LEN):
        await update.message.reply_text(text[i:i+MAX_LEN])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Ask me any financial question.")

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text.strip()
    if not user_query:
        await update.message.reply_text("Please enter a valid question.")
        return

    print(f"[INFO] 👤 User asked: {user_query}")
    await update.message.reply_text("Processing your question...")
    response = bot_instance.process_query(user_query)

    # ✅ Send the final answer
    answer = response.get("response") if isinstance(response, dict) else str(response)
    await send_in_chunks(update, answer)

    # ✅ Organize chunks by source file
    if isinstance(response, dict) and response.get("sources") and response.get("contexts"):
        grouped = {}
        for i, src in enumerate(response["sources"]):
            filename = src["file"]
            grouped.setdefault(filename, []).append(response["contexts"][i])

        # ✅ Build organized output
        organized_output = "📄 Retrieved Documents:\n"
        for file, chunks in grouped.items():
            organized_output += f"\n📂 **{file}**\n"
            for idx, chunk in enumerate(chunks, 1):
                organized_output += f"\n🔹 Chunk {idx}:\n{chunk}\n"

        await send_in_chunks(update, organized_output)

    print("[INFO] ✅ Response sent to user.")

# --- Run Bot ---
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN", "7596897324:AAG3TsT18amwRF2nRBcr1JS6NdGs96Ie-D0")
    print("[INFO] 🚀 Starting Telegram Financial Advisor Bot...")
    app = ApplicationBuilder().token(token).build()
    logger.info("Bot started successfully.")
    print("[INFO] ✅ Telegram Bot is now polling for messages...")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    app.run_polling()
