import os
import re
import logging
import time
import torch
from typing import Dict, List
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG utilities and language detection
from rag_utils import RAGUtils
from advanced_rag_feedback import AdvancedRAGFeedbackLoop
from config import config
from language_utils import LanguageDetector, BilingualResponseFormatter

# --- Logging ---
logging.basicConfig(
    filename='logs/telegram_financial_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Config (Using centralized configuration) ---
FAISS_INDEX_PATH = config.FAISS_INDEX_PATH
EMBEDDING_MODEL = config.EMBEDDING_MODEL
GROQ_MODEL = config.GROQ_MODEL
GROQ_API_KEY = config.GROQ_API_KEY
CACHE_TTL = config.CACHE_TTL

# Retrieval Settings
MAX_DOCS_FOR_RETRIEVAL = config.MAX_DOCS_FOR_RETRIEVAL
MAX_DOCS_FOR_CONTEXT = config.MAX_DOCS_FOR_CONTEXT
CONTEXT_CHUNK_SIZE = config.CONTEXT_CHUNK_SIZE

# Cross-Encoder Re-ranking Configuration
CROSS_ENCODER_MODEL = config.CROSS_ENCODER_MODEL
RELEVANCE_THRESHOLD = config.RELEVANCE_THRESHOLD

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

# --- Cache ---
class ResponseCache:
    def __init__(self, ttl: int = CACHE_TTL):
        self.cache = {}
        self.ttl = ttl

    def get(self, key: str):
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value):
        self.cache[key] = (time.time(), value)

# --- Query Processing ---
class QueryProcessor:
    def process(self, query: str) -> str:
        return "general"

# --- Financial Advisor Bot with Language Detection ---
class FinancialAdvisorTelegramBot:
    def __init__(self):
        self.cache = ResponseCache()
        self.processor = QueryProcessor()
        self.rag_utils = RAGUtils()  # Initialize RAG utilities
        self.feedback_loop = None  # Will be initialized after RAG setup
        
        # Initialize language detection components
        self.language_detector = LanguageDetector()
        self.response_formatter = BilingualResponseFormatter(self.language_detector)
        
        self._init_rag()

    def _init_rag(self):
        print("[INFO] Initializing FAISS and LLM...")
        logger.info("Initializing FAISS + LLM...")

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        print(f"[INFO] Loading FAISS index from: {FAISS_INDEX_PATH}")
        self.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("[INFO] ✅ FAISS index loaded successfully.")

        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.5,
            max_tokens=1200,
            model_kwargs={"top_p": 0.9}
        )
        print(f"[INFO] ✅ Groq LLM initialized with model: {GROQ_MODEL}")

        # Initialize Cross-Encoder for advanced re-ranking
        print("[INFO] Loading Cross-Encoder for document re-ranking...")
        try:
            self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
            print("[INFO] ✅ Cross-Encoder loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load Cross-Encoder: {e}")
            print("[INFO] Falling back to simple re-ranking...")
            self.reranker = None

        # Initialize retriever without creating chain yet (will be done per query with appropriate prompt)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": MAX_DOCS_FOR_RETRIEVAL})
        
        # Initialize Advanced RAG Feedback Loop
        print("[INFO] Initializing Advanced RAG Feedback Loop...")
        try:
            feedback_config = config.get_feedback_loop_config()
            self.feedback_loop = AdvancedRAGFeedbackLoop(
                vectorstore=self.vectorstore,
                rag_utils=self.rag_utils,
                config=feedback_config
            )
            print("[INFO] ✅ Advanced RAG Feedback Loop initialized successfully.")
            
            # Print configuration summary if enabled
            if feedback_config.get("enable_feedback_loop", True):
                config.print_config_summary()
            else:
                print("[INFO] ⚠️  Advanced RAG Feedback Loop is DISABLED - using traditional RAG")
                
        except Exception as e:
            print(f"[WARNING] Failed to initialize Advanced RAG Feedback Loop: {e}")
            print("[INFO] Falling back to traditional RAG approach...")
            self.feedback_loop = None

    def _rank_and_filter(self, docs: List[Document], query: str) -> List[Document]:
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
        ]
        
        for pattern in form_patterns:
            if re.search(pattern, content_lower):
                return True
        
        # Check if content is too short and uninformative
        if len(content.strip()) < 30 and ':' in content:
            return True
            
        return False

    def _cross_encoder_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """Re-rank documents using Cross-Encoder model"""
        if not docs or not self.reranker:
            return docs
        
        try:
            # Prepare query-document pairs for scoring
            pairs = []
            valid_docs = []
            
            for doc in docs:
                content = doc.page_content.strip()
                
                # Skip form fields and templates
                if self._is_form_field_or_template(content):
                    continue
                
                # Skip very short or empty content
                if len(content) < 50:
                    continue
                
                pairs.append([query, content])
                valid_docs.append(doc)
            
            if not pairs:
                print("[INFO] ⚠️ No valid documents after filtering form fields")
                return docs[:MAX_DOCS_FOR_CONTEXT]  # Return original docs as fallback
            
            # Get relevance scores
            scores = self.reranker.predict(pairs)
            
            # Combine documents with scores and sort
            doc_scores = list(zip(valid_docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by relevance threshold and limit
            filtered_docs = []
            for doc, score in doc_scores:
                if score >= RELEVANCE_THRESHOLD and len(filtered_docs) < MAX_DOCS_FOR_CONTEXT:
                    filtered_docs.append(doc)
                    print(f"[INFO] ✅ Document relevance score: {score:.3f}")
            
            if not filtered_docs:
                print(f"[INFO] ⚠️ No documents above relevance threshold ({RELEVANCE_THRESHOLD})")
                # Return top documents even if below threshold, but limit to 2
                filtered_docs = [doc for doc, _ in doc_scores[:2]]
            
            return filtered_docs
            
        except Exception as e:
            print(f"[WARNING] Cross-encoder re-ranking failed: {e}")
            return docs[:MAX_DOCS_FOR_CONTEXT]

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
            if score >= min_score and len(filtered_docs) < MAX_DOCS_FOR_CONTEXT:
                filtered_docs.append(doc)
                print(f"[INFO] ✅ Document lexical score: {score:.3f}")
        
        if not filtered_docs and scored_docs:
            # If no docs meet threshold, return the best one
            filtered_docs = [scored_docs[0][0]]
            print(f"[INFO] ⚠️ Using best document despite low score: {scored_docs[0][1]:.3f}")
        
        return filtered_docs

    def _prepare_docs(self, docs: List[Document]) -> List[Document]:
        """Prepare documents by truncating content if needed"""
        processed = []
        for doc in docs:
            content = doc.page_content
            if len(content) > CONTEXT_CHUNK_SIZE:
                content = content[:CONTEXT_CHUNK_SIZE] + "..."
            
            processed_doc = Document(
                page_content=content,
                metadata=doc.metadata
            )
            processed.append(processed_doc)
        
        return processed

    def _create_language_specific_chain(self, detected_language: str):
        """Create retrieval chain with appropriate language-specific prompt"""
        # Determine response language
        response_language = self.language_detector.determine_response_language(detected_language)
        
        # Get language-specific prompt
        language_prompt = self.language_detector.get_language_specific_prompt(response_language)
        
        # Create document chain with language-specific prompt
        doc_chain = create_stuff_documents_chain(self.llm, language_prompt)
        
        # Create retrieval chain
        return create_retrieval_chain(self.retriever, doc_chain)
    
    def _create_english_chain(self):
        """Create retrieval chain with English prompt (for translation approach)"""
        # Always use English prompt for translation approach
        english_prompt = self.language_detector._get_english_prompt()
        
        # Create document chain with English prompt
        doc_chain = create_stuff_documents_chain(self.llm, english_prompt)
        
        # Create retrieval chain
        return create_retrieval_chain(self.retriever, doc_chain)

    def process_query(self, query: str) -> Dict:
        """
        Enhanced process_query with translation-based multilingual support
        """
        # Detect language of the query
        detected_language, confidence = self.language_detector.detect_language(query)
        print(f"[INFO] 🌐 Language detected: {detected_language} (confidence: {confidence:.2f})")
        
        # Store original query and language for response formatting
        original_query = query
        original_language = detected_language
        
        # Get configuration for translation approach
        use_translation = config.get_feedback_loop_config().get("use_translation_approach", True)
        enable_query_translation = config.get_feedback_loop_config().get("enable_query_translation", True)
        
        # If query is in Bangla and translation is enabled, translate to English for processing
        if detected_language == 'bengali' and use_translation and enable_query_translation:
            print(f"[INFO] 🔄 Translating Bangla query to English...")
            query = self.language_detector.translate_bangla_to_english(query)
            print(f"[INFO] ✅ Translated query: {query}")
        
        category = self.processor.process(query)
        logger.info(f"Processing query (category={category}, original_language={original_language}): {query}")
        print(f"[INFO] 🔍 Processing query: {query}")

        # Create cache key with original language and query
        cache_key = f"{original_language}:{original_query}"
        cached = self.cache.get(cache_key)
        if cached:
            print("[INFO] ✅ Cache hit - returning stored response.")
            # Add language context to cached response
            cached['detected_language'] = original_language
            cached['language_confidence'] = confidence
            return cached

        try:
            # Use Advanced RAG Feedback Loop if available, otherwise fallback to traditional approach
            if self.feedback_loop is not None:
                result = self._process_query_with_feedback_loop(query, category, original_language)
            else:
                result = self._process_query_traditional(query, category, original_language)
            
            # Add language detection information to result
            result['detected_language'] = original_language
            result['language_confidence'] = confidence
            result['was_translated'] = (original_language == 'bengali' and use_translation and enable_query_translation)
            result['translated_query'] = query if result['was_translated'] else None
            
            # Cache the result with language-aware key
            self.cache.set(cache_key, result)
            
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"[ERROR] {e}")
            
            # Return error message in original language
            error_message = self.language_detector.translate_system_messages(
                f"Error: {e}", original_language
            )
            
            return {
                "response": error_message, 
                "sources": [], 
                "contexts": [],
                "detected_language": original_language,
                "language_confidence": confidence
            }
    
    def _process_query_with_feedback_loop(self, query: str, category: str, original_language: str) -> Dict:
        """Process query using the Advanced RAG Feedback Loop with translation support"""
        print("[INFO] 🔄 Using Advanced RAG Feedback Loop...")
        
        # Get configuration for translation approach
        use_translation = config.get_feedback_loop_config().get("use_translation_approach", True)
        enable_response_translation = config.get_feedback_loop_config().get("enable_response_translation", True)
        
        # Step 1: Use feedback loop to get the best documents (query is already in English if translated)
        feedback_result = self.feedback_loop.retrieve_with_feedback_loop(query, category)
        
        if not feedback_result["documents"]:
            print(f"[INFO] ❌ Feedback loop found no relevant documents. Reason: {feedback_result.get('failure_reason', 'Unknown')}")
            failure_msg = "I could not find relevant information in my database for your query."
            # Translate failure message if needed
            if original_language == 'bengali' and use_translation and enable_response_translation:
                failure_msg = self.language_detector.translate_english_to_bangla(failure_msg)
            return {"response": failure_msg, "sources": [], "contexts": []}
        
        # Log feedback loop results
        print(f"[INFO] 🎯 Feedback loop completed:")
        print(f"[INFO]   - Total iterations: {feedback_result['total_iterations']}")
        print(f"[INFO]   - Final query used: '{feedback_result['query_used']}'")
        print(f"[INFO]   - Relevance score: {feedback_result['relevance_score']:.3f}")
        print(f"[INFO]   - Documents found: {len(feedback_result['documents'])}")
        
        # Step 2: Apply additional filtering and ranking
        filtered = self._rank_and_filter(feedback_result["documents"], feedback_result["query_used"])
        if not filtered:
            print("[INFO] ❌ No documents passed final filtering.")
            failure_msg = "I could not find sufficiently relevant information in my database."
            # Translate failure message if needed
            if original_language == 'bengali' and use_translation and enable_response_translation:
                failure_msg = self.language_detector.translate_english_to_bangla(failure_msg)
            return {"response": failure_msg, "sources": [], "contexts": []}

        docs = self._prepare_docs(filtered)

        # Step 3: Generate answer using appropriate chain
        if use_translation:
            # Translation approach: always use English chain
            print("[INFO] ✅ Running LLM to generate English answer...")
            qa_chain = self._create_english_chain()
            result = qa_chain.invoke({"input": feedback_result["query_used"]})
            answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
            print("[INFO] ✅ English answer generated successfully.")
            
            # Translate answer back to Bangla if needed
            if original_language == 'bengali' and enable_response_translation:
                print("[INFO] 🔄 Translating answer back to Bangla...")
                answer = self.language_detector.translate_english_to_bangla(answer)
                print("[INFO] ✅ Answer translated to Bangla")
        else:
            # Traditional approach: use language-specific chain
            print("[INFO] ✅ Running LLM with language-specific prompt...")
            qa_chain = self._create_language_specific_chain(original_language)
            result = qa_chain.invoke({"input": feedback_result["query_used"]})
            answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
            print(f"[INFO] ✅ {original_language.capitalize()} answer generated successfully.")

        # Step 4: Validate the generated answer (always use English version for validation)
        context_texts = [d.page_content for d in docs]
        english_answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
        validation = self.rag_utils.validate_answer(feedback_result["query_used"], english_answer, context_texts)
        print(f"[INFO] ✅ Answer validation - Valid: {validation['valid']}, Confidence: {validation['confidence']:.2f}")
        
        # Step 5: Apply validation logic and disclaimers if needed
        if not validation['valid'] or validation['confidence'] < 0.15:
            if validation['confidence'] > 0.05 and answer and len(answer.strip()) > 20:
                disclaimer = "Note: I have moderate confidence in this answer. Please verify the information with official sources or consult a financial advisor for specific advice."
                if use_translation and original_language == 'bengali' and enable_response_translation:
                    disclaimer = self.language_detector.translate_english_to_bangla(disclaimer)
                answer = f"{answer}\n\n⚠️ *{disclaimer}*"
            else:
                fallback_msg = "I'm not confident in my answer based on the available information. Please rephrase your question or ask about a different topic."
                if use_translation and original_language == 'bengali' and enable_response_translation:
                    fallback_msg = self.language_detector.translate_english_to_bangla(fallback_msg)
                answer = fallback_msg

        # Step 6: Prepare sources and contexts
        sources = []
        contexts = []
        for doc in docs:
            source_info = {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0)
            }
            sources.append(source_info)
            contexts.append(doc.page_content)

        return {
            "response": answer,
            "sources": sources,
            "contexts": contexts,
            "feedback_iterations": feedback_result['total_iterations'],
            "final_query": feedback_result['query_used'],
            "relevance_score": feedback_result['relevance_score'],
            "validation_confidence": validation['confidence']
        }

    def _process_query_traditional(self, query: str, category: str, original_language: str) -> Dict:
        """Traditional RAG processing without feedback loop with translation support"""
        print("[INFO] 🔄 Using traditional RAG approach...")
        
        # Get configuration for translation approach
        use_translation = config.get_feedback_loop_config().get("use_translation_approach", True)
        enable_response_translation = config.get_feedback_loop_config().get("enable_response_translation", True)
        
        # Step 1: Create appropriate chain and retrieve documents
        if use_translation:
            # Translation approach: use English chain
            qa_chain = self._create_english_chain()
            result = qa_chain.invoke({"input": query})
            answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
            
            # Translate answer back to Bangla if needed
            if original_language == 'bengali' and enable_response_translation:
                print("[INFO] 🔄 Translating answer back to Bangla...")
                answer = self.language_detector.translate_english_to_bangla(answer)
                print("[INFO] ✅ Answer translated to Bangla")
        else:
            # Traditional approach: use language-specific chain
            qa_chain = self._create_language_specific_chain(original_language)
            result = qa_chain.invoke({"input": query})
            answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
        
        # Step 3: Get source documents
        source_docs = result.get("context", [])
        if not source_docs:
            return {"response": answer, "sources": [], "contexts": []}
        
        # Step 4: Apply filtering and ranking
        filtered = self._rank_and_filter(source_docs, query)
        docs = self._prepare_docs(filtered)
        
        # Step 5: Prepare sources and contexts
        sources = []
        contexts = []
        for doc in docs:
            source_info = {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0)
            }
            sources.append(source_info)
            contexts.append(doc.page_content)

        return {
            "response": answer,
            "sources": sources,
            "contexts": contexts
        }

# --- Global bot instance ---
bot_instance = FinancialAdvisorTelegramBot()

# --- Telegram Handlers ---
async def send_in_chunks(update: Update, text: str, chunk_size: int = 4000):
    """Send long messages in chunks to avoid Telegram's message length limit"""
    if len(text) <= chunk_size:
        await update.message.reply_text(text)
        return

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    for i, chunk in enumerate(chunks):
        if i == 0:
            await update.message.reply_text(f"{chunk}... (continued)")
        elif i == len(chunks) - 1:
            await update.message.reply_text(f"...{chunk}")
        else:
            await update.message.reply_text(f"...{chunk}... (continued)")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced start command with language detection"""
    # Detect language preference from user's message if any
    user_language = 'english'  # Default
    
    # Check if user sent any text with the start command
    if context.args:
        sample_text = ' '.join(context.args)
        detected_lang, _ = bot_instance.language_detector.detect_language(sample_text)
        user_language = detected_lang
    
    # Send welcome message in appropriate language
    if user_language == 'bengali':
        welcome_message = "হ্যালো! আমাকে যেকোনো আর্থিক প্রশ্ন করুন। আমি বাংলা এবং ইংরেজি দুই ভাষাতেই উত্তর দিতে পারি।"
    else:
        welcome_message = "Hi! Ask me any financial question. I can respond in both English and Bangla based on your question language."
    
    await update.message.reply_text(welcome_message)

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced query handler with language detection and bilingual response"""
    user_query = update.message.text.strip()
    if not user_query:
        # Detect language for error message
        error_message = bot_instance.language_detector.translate_system_messages(
            "Please enter a valid question.", 'english'
        )
        await update.message.reply_text(error_message)
        return

    print(f"[INFO] 👤 User asked: {user_query}")
    
    # Detect language and show processing message in appropriate language
    detected_language, confidence = bot_instance.language_detector.detect_language(user_query)
    processing_message = bot_instance.language_detector.translate_system_messages(
        "Processing your question...", detected_language
    )
    await update.message.reply_text(processing_message)
    
    # Process the query
    response = bot_instance.process_query(user_query)

    # ✅ Send the final answer
    answer = response.get("response") if isinstance(response, dict) else str(response)
    
    # Enhance answer with language-specific formatting if needed
    if isinstance(response, dict):
        detected_lang = response.get('detected_language', detected_language)
        lang_confidence = response.get('language_confidence', confidence)
        
        # Add confidence disclaimer if needed and validation confidence is low
        validation_confidence = response.get('validation_confidence', 1.0)
        if validation_confidence < 0.3:
            confidence_msg = bot_instance.language_detector.format_confidence_message(detected_lang)
            if confidence_msg not in answer:
                answer += confidence_msg
    
    await send_in_chunks(update, answer)

    # ✅ Organize chunks by source file with language-appropriate headers
    if isinstance(response, dict) and response.get("sources") and response.get("contexts"):
        detected_lang = response.get('detected_language', detected_language)
        
        grouped = {}
        for i, src in enumerate(response["sources"]):
            filename = src["file"]
            grouped.setdefault(filename, []).append(response["contexts"][i])

        # ✅ Build organized output with language-appropriate formatting
        organized_output = bot_instance.response_formatter.format_sources_section(detected_lang)
        
        for doc_idx, (file, chunks) in enumerate(grouped.items(), 1):
            organized_output += bot_instance.response_formatter.format_document_header(
                doc_idx, file, detected_lang
            )
            
            for idx, chunk in enumerate(chunks, 1):
                organized_output += bot_instance.response_formatter.format_chunk_header(
                    idx, detected_lang
                )
                organized_output += f"{chunk}\n"

        await send_in_chunks(update, organized_output)

    print("[INFO] ✅ Response sent to user.")

# --- Run Bot ---
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_TOKEN environment variable is required. Please set it in your .env file.")
    
    print("[INFO] 🚀 Starting Telegram Financial Advisor Bot with Language Detection...")
    app = ApplicationBuilder().token(token).build()
    logger.info("Bot started successfully with bilingual support.")
    print("[INFO] ✅ Telegram Bot is now polling for messages...")
    print("[INFO] 🌐 Language detection enabled: English ⟷ Bangla")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    app.run_polling()
