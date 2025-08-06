import os
import re
import logging
import time
from typing import Dict, List, Tuple
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
import langdetect
from langdetect import detect
import warnings
warnings.filterwarnings("ignore")

# Import Spanish translator
from spanish_translator import SpanishTranslator

# --- Logging ---
logging.basicConfig(
    filename='logs/telegram_multilingual_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Config ---
FAISS_INDEX_PATH = "faiss_index_multilingual"
# Using multilingual embedding model that supports Bangla and English
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Alternative: "BAAI/bge-m3" (better for multilingual)
OLLAMA_MODEL = "gemma3n:e2b" 
CACHE_TTL = 86400  # 24 hours

# Retrieval Settings
MAX_DOCS_FOR_RETRIEVAL = 15  # Increased for multilingual
MAX_DOCS_FOR_CONTEXT = 6     # Increased for better context
CONTEXT_CHUNK_SIZE = 1800    # Increased for multilingual content

# Cross-Encoder Re-ranking Configuration (multilingual)
CROSS_ENCODER_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'  # Multilingual cross-encoder
RELEVANCE_THRESHOLD = 0.15   # Lowered for multilingual (different score ranges)

# Hybrid Re-ranking Configuration
SEMANTIC_WEIGHT = 0.75       # Increased semantic weight for multilingual
LEXICAL_WEIGHT = 0.25        # Decreased lexical weight
PHRASE_BONUS_MULTIPLIER = 1.5
LENGTH_BONUS_MULTIPLIER = 0.3

# Language Detection Settings
BANGLA_KEYWORDS = ['ব্যাংক', 'টাকা', 'ঋণ', 'হিসাব', 'সুদ', 'বিনিয়োগ', 'কর', 'আয়কর', 'সঞ্চয়', 'আবেদন']
ENGLISH_KEYWORDS = ['bank', 'loan', 'account', 'interest', 'investment', 'tax', 'income', 'savings', 'application']
SPANISH_KEYWORDS = ['banco', 'cuenta', 'préstamo', 'crédito', 'dinero', 'inversión', 'impuesto', 'ahorro', 'tarjeta', 'financiero']

# --- Multilingual Prompts ---
BANGLA_PROMPT_TEMPLATE = """
আপনি বাংলাদেশের ব্যাংকিং এবং আর্থিক সেবা বিষয়ে বিশেষজ্ঞ একজন সহায়ক আর্থিক পরামর্শদাতা।
সর্বদা বন্ধুত্বপূর্ণ এবং স্বাভাবিক ভাষায় উত্তর দিন।

গুরুত্বপূর্ণ নির্দেশনা:
- প্রদত্ত তথ্যের ভিত্তিতে উত্তর দিন
- ফর্ম ফিল্ড, খালি টেমপ্লেট, এবং অসম্পূর্ণ নথির অংশ উপেক্ষা করুন
- "প্রসঙ্গ অনুযায়ী" বলবেন না - সরাসরি উত্তর দিন
- যথেষ্ট তথ্য না থাকলে বলুন "এ বিষয়ে আমার কাছে নির্দিষ্ট তথ্য নেই"
- মুদ্রা হিসেবে বাংলাদেশী টাকা (৳/টক) ব্যবহার করুন
- সংক্ষিপ্ত এবং ব্যবহারিক উত্তর দিন

প্রসঙ্গ তথ্য:
{context}

প্রশ্ন: {input}

উত্তর:"""

ENGLISH_PROMPT_TEMPLATE = """
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

# Spanish uses English prompt since we translate the query to English first
SPANISH_PROMPT_TEMPLATE = ENGLISH_PROMPT_TEMPLATE

# --- Language Detection and Processing ---
class LanguageProcessor:
    def __init__(self):
        self.bangla_keywords = set(BANGLA_KEYWORDS)
        self.english_keywords = set(ENGLISH_KEYWORDS)
        self.spanish_keywords = set(SPANISH_KEYWORDS)
    
    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Bangla, English, or Spanish"""
        try:
            # First try langdetect
            detected = detect(text.lower())
            if detected == 'bn':
                return 'bangla'
            elif detected == 'en':
                return 'english'
            elif detected == 'es':
                return 'spanish'
        except:
            pass
        
        # Fallback to keyword-based detection
        text_lower = text.lower()
        bangla_count = sum(1 for word in self.bangla_keywords if word in text)
        english_count = sum(1 for word in self.english_keywords if word in text_lower)
        spanish_count = sum(1 for word in self.spanish_keywords if word in text_lower)
        
        # Check for Bangla Unicode characters
        bangla_chars = len([c for c in text if '\u0980' <= c <= '\u09FF'])
        
        if bangla_chars > 0 or bangla_count > max(english_count, spanish_count):
            return 'bangla'
        elif spanish_count > english_count:
            return 'spanish'
        else:
            return 'english'
    
    def get_prompt_template(self, language: str) -> PromptTemplate:
        """Get appropriate prompt template based on language"""
        if language == 'bangla':
            return PromptTemplate(
                input_variables=["context", "input"], 
                template=BANGLA_PROMPT_TEMPLATE
            )
        elif language == 'spanish':
            # Spanish uses English prompt since query is translated to English
            return PromptTemplate(
                input_variables=["context", "input"], 
                template=SPANISH_PROMPT_TEMPLATE
            )
        else:
            return PromptTemplate(
                input_variables=["context", "input"], 
                template=ENGLISH_PROMPT_TEMPLATE
            )

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

# --- Multilingual Query Processor ---
class MultilingualQueryProcessor:
    def __init__(self):
        self.lang_processor = LanguageProcessor()
    
    def process(self, query: str) -> Tuple[str, str]:
        """Process query and return (category, language)"""
        language = self.lang_processor.detect_language(query)
        
        q = query.lower()
        # Multilingual category detection
        if any(word in q for word in ['tax', 'কর', 'আয়কর', 'income tax', 'impuesto', 'fiscal']):
            return "taxation", language
        elif any(word in q for word in ['loan', 'ঋণ', 'লোন', 'credit', 'préstamo', 'crédito']):
            return "loans", language
        elif any(word in q for word in ['investment', 'বিনিয়োগ', 'সঞ্চয়', 'savings', 'inversión', 'ahorro']):
            return "investment", language
        elif any(word in q for word in ['bank', 'ব্যাংক', 'account', 'হিসাব', 'banco', 'cuenta']):
            return "banking", language
        else:
            return "general", language

# --- Multilingual Financial Advisor Bot ---
class MultilingualFinancialAdvisorBot:
    def __init__(self):
        self.cache = ResponseCache()
        self.processor = MultilingualQueryProcessor()
        self.lang_processor = LanguageProcessor()
        self.spanish_translator = SpanishTranslator(OLLAMA_MODEL)  # Initialize Spanish translator
        self._init_rag()

    def _init_rag(self):
        print("[INFO] Initializing Multilingual FAISS and LLM...")
        logger.info("Initializing Multilingual FAISS + LLM...")

        # Use multilingual embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, 
            model_kwargs={"device": "cpu"}
        )
        print(f"[INFO] Loading multilingual FAISS index from: {FAISS_INDEX_PATH}")
        
        try:
            self.vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("[INFO] ✅ Multilingual FAISS index loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Could not load multilingual index: {e}")
            print("[INFO] Please run multilingual_semantic_chunking.py first")
            raise e

        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            temperature=0.5,
            max_tokens=1500,  # Increased for multilingual responses
            top_p=0.9,
            repeat_penalty=1.1
        )
        print(f"[INFO] ✅ Ollama LLM initialized with model: {OLLAMA_MODEL}")

        # Initialize Multilingual Cross-Encoder
        print("[INFO] Loading Multilingual Cross-Encoder for document re-ranking...")
        try:
            self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
            print("[INFO] ✅ Multilingual Cross-Encoder loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load Cross-Encoder: {e}")
            print("[INFO] Falling back to simple re-ranking...")
            self.reranker = None

    def _is_form_field_or_template(self, content: str) -> bool:
        """Enhanced form field detection for both Bangla and English"""
        content_lower = content.lower().strip()
        
        # English form patterns
        english_patterns = [
            r':\s*\.{3,}', r':\s*_{3,}', r':\s*\d+\.\s*$',
            r'\(if any\):\s*\d+', r'^\w+\s+no\.\s*$',
            r'^\s*(name|address|phone|email|passport|tin|nid|bin|vat)?\s*:\s*(if any)?\s*\d*\.?\s*$',
        ]
        
        # Bangla form patterns
        bangla_patterns = [
            r'নাম\s*:\s*\.{3,}', r'ঠিকানা\s*:\s*\.{3,}', r'ফোন\s*:\s*\.{3,}',
            r'জন্ম তারিখ\s*:\s*\.{3,}', r'পিতার নাম\s*:\s*\.{3,}',
            r'মাতার নাম\s*:\s*\.{3,}', r'স্বাক্ষর\s*:\s*\.{3,}',
        ]
        
        all_patterns = english_patterns + bangla_patterns
        
        for pattern in all_patterns:
            if re.search(pattern, content_lower):
                return True
        
        # Check for common form keywords in both languages
        form_keywords = [
            'signature', 'date', 'seal', 'stamp', 'official use only', 'for office use',
            'স্বাক্ষর', 'তারিখ', 'সিল', 'স্ট্যাম্প', 'অফিসিয়াল ব্যবহার'
        ]
        
        if any(keyword in content_lower for keyword in form_keywords):
            return True
            
        return False

    def _rank_and_filter(self, docs: List[Document], query: str) -> List[Document]:
        """Enhanced multilingual re-ranking"""
        if not docs:
            return docs
        
        if self.reranker is not None:
            return self._hybrid_rerank(docs, query)
        else:
            return self._multilingual_lexical_rerank(docs, query)
    
    def _multilingual_lexical_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Multilingual lexical re-ranking"""
        if not docs:
            return docs
        
        # Detect query language
        query_lang = self.lang_processor.detect_language(query)
        
        # Enhanced query preprocessing for multilingual
        query_terms = set(query.lower().split())
        
        # Language-specific stop words
        if query_lang == 'bangla':
            stop_words = {'এর', 'এবং', 'বা', 'কিন্তু', 'যে', 'যা', 'কি', 'কে', 'কোন', 'এই', 'সেই', 'তার', 'তাদের'}
        else:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        query_terms = query_terms - stop_words
        
        scored = []
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            # Skip form fields
            if self._is_form_field_or_template(doc.page_content):
                continue
            
            # Enhanced multilingual scoring
            exact_matches = sum(1 for t in query_terms if t in content_lower)
            partial_matches = sum(1 for t in query_terms if any(t in word for word in content_lower.split()))
            
            # Phrase matching
            query_words = query.lower().split()
            phrase_bonus = 0
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase in content_lower:
                    phrase_bonus += PHRASE_BONUS_MULTIPLIER
            
            # Language preference bonus
            doc_lang = self.lang_processor.detect_language(doc.page_content)
            lang_bonus = 1.2 if doc_lang == query_lang else 1.0
            
            # Length bonus
            length_bonus = min(len(content_lower) / 1000, 1.0) * LENGTH_BONUS_MULTIPLIER
            
            # Calculate final score
            score = (exact_matches * 3 + 
                    partial_matches * 1 + 
                    phrase_bonus + 
                    length_bonus) * lang_bonus
            
            if score > 0:
                scored.append((doc, score))
        
        # Sort by score and return top documents
        ranked = [d for d, s in sorted(scored, key=lambda x: x[1], reverse=True)]
        return ranked[:MAX_DOCS_FOR_CONTEXT]

    def _hybrid_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Multilingual hybrid re-ranking"""
        if not docs:
            return docs
        
        try:
            # Filter form fields
            informative_docs = [doc for doc in docs if not self._is_form_field_or_template(doc.page_content)]
            
            if not informative_docs:
                informative_docs = docs
            
            print(f"[INFO] Re-ranking {len(informative_docs)} documents using Multilingual Hybrid approach...")
            
            # Cross-encoder scoring
            pairs = [[query, doc.page_content[:1000]] for doc in informative_docs]
            cross_encoder_scores = self.reranker.predict(pairs)
            
            # Lexical scores
            lexical_scores = self._get_multilingual_lexical_scores(informative_docs, query)
            
            # Combine scores
            combined_scores = []
            max_lex = max(lexical_scores) if lexical_scores else 1
            
            for i, doc in enumerate(informative_docs):
                ce_score = max(0, cross_encoder_scores[i])
                lex_score = lexical_scores[i] / max_lex if max_lex > 0 else 0
                
                combined_score = SEMANTIC_WEIGHT * ce_score + LEXICAL_WEIGHT * lex_score
                combined_scores.append((doc, combined_score))
            
            # Filter by relevance threshold
            filtered_docs = [doc for doc, score in combined_scores if score > RELEVANCE_THRESHOLD]
            
            print(f"[INFO] ✅ Multilingual hybrid re-ranking completed. Kept {len(filtered_docs)} relevant documents.")
            return filtered_docs[:MAX_DOCS_FOR_CONTEXT]
            
        except Exception as e:
            print(f"[WARNING] Multilingual hybrid re-ranking failed: {e}")
            return self._multilingual_lexical_rerank(docs, query)

    def _get_multilingual_lexical_scores(self, docs: List[Document], query: str) -> List[float]:
        """Calculate lexical scores for multilingual documents"""
        query_lang = self.lang_processor.detect_language(query)
        query_terms = set(query.lower().split())
        
        # Remove stop words based on language
        if query_lang == 'bangla':
            stop_words = {'এর', 'এবং', 'বা', 'কিন্তু', 'যে', 'যা', 'কি', 'কে', 'কোন'}
        else:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        query_terms = query_terms - stop_words
        
        scores = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            exact_matches = sum(1 for t in query_terms if t in content_lower)
            partial_matches = sum(1 for t in query_terms if any(t in word for word in content_lower.split()))
            
            # Language matching bonus
            doc_lang = self.lang_processor.detect_language(doc.page_content)
            lang_bonus = 1.3 if doc_lang == query_lang else 1.0
            
            score = (exact_matches * 3 + partial_matches * 1) * lang_bonus
            scores.append(score)
        
        return scores

    def _prepare_docs(self, docs: List[Document]) -> List[Document]:
        processed = []
        for d in docs[:MAX_DOCS_FOR_CONTEXT]:
            content = d.page_content[:CONTEXT_CHUNK_SIZE] + ("...[truncated]" if len(d.page_content) > CONTEXT_CHUNK_SIZE else "")
            processed.append(Document(page_content=content, metadata=d.metadata))
        return processed

    def process_query(self, query: str) -> Dict:
        category, language = self.processor.process(query)
        logger.info(f"Processing multilingual query (category={category}, language={language}): {query}")
        print(f"[INFO] 🔍 Received {language} query: {query}")

        # Handle Spanish translation workflow
        original_query = query
        original_language = language
        
        if language == 'spanish':
            print("[INFO] 🔄 Spanish detected - translating to English for search...")
            # Translate Spanish query to English
            english_query, _ = self.spanish_translator.process_spanish_query(query)
            query = english_query  # Use English query for search
            language = 'english'   # Process as English internally
            print(f"[INFO] 📝 Translated query: {english_query}")

        cached = self.cache.get(original_query)  # Cache with original query
        if cached:
            print("[INFO] ✅ Cache hit - returning stored response.")
            return cached

        try:
            # Retrieve documents (always in English since our index is English/Bangla)
            retrieved = self.vectorstore.similarity_search(query, k=MAX_DOCS_FOR_RETRIEVAL)
            filtered = self._rank_and_filter(retrieved, query)
            
            if not filtered:
                if original_language == 'spanish':
                    no_info_msg = "No pude encontrar información relevante en mi base de datos."
                elif original_language == 'bangla':
                    no_info_msg = "এ বিষয়ে আমার কাছে প্রাসঙ্গিক তথ্য নেই।"
                else:
                    no_info_msg = "I could not find relevant information in my database."
                return {"response": no_info_msg, "sources": [], "contexts": [], "language": original_language}

            docs = self._prepare_docs(filtered)

            # Get appropriate prompt template (use English for Spanish since query was translated)
            prompt_template = self.lang_processor.get_prompt_template(language)
            
            # Create chain with language-specific prompt
            doc_chain = create_stuff_documents_chain(self.llm, prompt_template)
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": MAX_DOCS_FOR_RETRIEVAL})
            qa_chain = create_retrieval_chain(retriever, doc_chain)

            print(f"[INFO] ✅ Running LLM to generate {language} answer...")
            result = qa_chain.invoke({"input": query})

            answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
            
            # Handle Spanish response translation
            if original_language == 'spanish':
                print("[INFO] 🔄 Translating English response back to Spanish...")
                answer = self.spanish_translator.process_english_response(answer, original_query)
                print(f"[INFO] ✅ Spanish response generated successfully.")
            else:
                print(f"[INFO] ✅ {language.title()} answer generated successfully.")

            context_texts = [d.page_content for d in docs]

            response = {
                "response": answer,
                "sources": [{"file": d.metadata.get("source", "Unknown")} for d in docs],
                "contexts": context_texts,
                "language": original_language,  # Return original language
                "translated_query": query if original_language == 'spanish' else None
            }
            self.cache.set(original_query, response)  # Cache with original query
            return response

        except Exception as e:
            logger.error(f"Error processing multilingual query: {e}")
            print(f"[ERROR] {e}")
            if original_language == 'spanish':
                error_msg = f"Error: {e}"
            elif original_language == 'bangla':
                error_msg = f"ত্রুটি: {e}"
            else:
                error_msg = f"Error: {e}"
            return {"response": error_msg, "sources": [], "contexts": [], "language": original_language}

# --- Telegram Handlers ---
bot_instance = MultilingualFinancialAdvisorBot()

async def send_in_chunks(update: Update, text: str):
    MAX_LEN = 4000
    for i in range(0, len(text), MAX_LEN):
        await update.message.reply_text(text[i:i+MAX_LEN])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = """
🇧🇩 স্বাগতম! আমি আপনার আর্থিক পরামর্শদাতা।
🇬🇧 Welcome! I'm your financial advisor.
🇪🇸 ¡Bienvenido! Soy tu asesor financiero.

আপনি বাংলা, ইংরেজি বা স্প্যানিশ ভাষায় যেকোনো আর্থিক প্রশ্ন করতে পারেন।
You can ask any financial question in Bangla, English, or Spanish.
Puedes hacer cualquier pregunta financiera en bengalí, inglés o español.
    """
    await update.message.reply_text(welcome_msg)

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text.strip()
    if not user_query:
        await update.message.reply_text("Please enter a valid question. / দয়া করে একটি বৈধ প্রশ্ন লিখুন। / Por favor ingrese una pregunta válida.")
        return

    print(f"[INFO] 👤 User asked: {user_query}")
    
    # Detect language for processing message
    lang_processor = LanguageProcessor()
    detected_lang = lang_processor.detect_language(user_query)
    
    if detected_lang == 'bangla':
        processing_msg = "আপনার প্রশ্ন প্রক্রিয়া করা হচ্ছে..."
    elif detected_lang == 'spanish':
        processing_msg = "Procesando tu pregunta..."
    else:
        processing_msg = "Processing your question..."
        
    await update.message.reply_text(processing_msg)
    
    response = bot_instance.process_query(user_query)

    # Send the final answer
    answer = response.get("response") if isinstance(response, dict) else str(response)
    await send_in_chunks(update, answer)

    # Send source information
    if isinstance(response, dict) and response.get("sources") and response.get("contexts"):
        grouped = {}
        for i, src in enumerate(response["sources"]):
            filename = src["file"]
            grouped.setdefault(filename, []).append(response["contexts"][i])

        # Language-specific source header
        response_lang = response.get("language", "english")
        if response_lang == 'bangla':
            source_header = "📄 উৎস নথিসমূহ:"
        elif response_lang == 'spanish':
            source_header = "📄 Documentos Fuente:"
        else:
            source_header = "📄 Retrieved Documents:"
        
        organized_output = source_header + "\n"
        for file, chunks in grouped.items():
            organized_output += f"\n📂 **{file}**\n"
            for idx, chunk in enumerate(chunks, 1):
                if response_lang == 'bangla':
                    chunk_label = f"অংশ {idx}"
                elif response_lang == 'spanish':
                    chunk_label = f"Fragmento {idx}"
                else:
                    chunk_label = f"Chunk {idx}"
                organized_output += f"\n🔹 {chunk_label}:\n{chunk}\n"

        await send_in_chunks(update, organized_output)

    print("[INFO] ✅ Multilingual response sent to user.")

# --- Run Bot ---
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN", "7596897324:AAG3TsT18amwRF2nRBcr1JS6NdGs96Ie-D0")
    print("[INFO] 🚀 Starting Multilingual Telegram Financial Advisor Bot...")
    app = ApplicationBuilder().token(token).build()
    logger.info("Multilingual Bot started successfully.")
    print("[INFO] ✅ Multilingual Telegram Bot is now polling for messages...")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    app.run_polling()
