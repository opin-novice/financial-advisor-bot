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

# Import RAG utilities
from rag_utils import RAGUtils
from advanced_rag_feedback import AdvancedRAGFeedbackLoop
from config import config

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

# --- Prompt (Bangladesh Context) ---
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
- IMPORTANT: Respond in the SAME LANGUAGE as the user's question (either English or Bangla)
- The context may contain both English and Bangla text - use whichever is relevant to answer the question
- Focus on providing actionable, practical advice

Context Information:
{context}

Question: {input}

Answer (provide a helpful response based on available information):"""
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
        self.rag_utils = RAGUtils()  # Initialize RAG utilities
        self.feedback_loop = None  # Will be initialized after RAG setup
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

        self.doc_chain = create_stuff_documents_chain(self.llm, QA_PROMPT)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": MAX_DOCS_FOR_RETRIEVAL})
        self.qa_chain = create_retrieval_chain(retriever, self.doc_chain)
        
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
            
        # Check for repetitive dots, underscores, or numbers
        if content.count('.') > len(content) * 0.3 or content.count('_') > 5:
            return True
            
        return False

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
        """Improved lexical re-ranking as fallback"""
        if not docs:
            return docs
        
        terms = set(query.lower().split())
        scored = []
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            # Count exact term matches
            exact_matches = sum(1 for t in terms if t in content_lower)
            
            # Bonus for partial matches (substrings)
            partial_matches = sum(1 for t in terms if any(t in word for word in content_lower.split()))
            
            # Calculate final score
            score = exact_matches * 2 + partial_matches * 0.5
            
            if score > 0:  # Only include documents with some relevance
                scored.append((doc, score))
        
        # Sort by score and return
        return [d for d, s in sorted(scored, key=lambda x: x[1], reverse=True)]

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
            # Use Advanced RAG Feedback Loop if available, otherwise fallback to traditional approach
            if self.feedback_loop is not None:
                return self._process_query_with_feedback_loop(query, category)
            else:
                return self._process_query_traditional(query, category)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"[ERROR] {e}")
            return {"response": f"Error: {e}", "sources": [], "contexts": []}
    
    def _process_query_with_feedback_loop(self, query: str, category: str) -> Dict:
        """Process query using the Advanced RAG Feedback Loop"""
        print("[INFO] 🔄 Using Advanced RAG Feedback Loop...")
        
        # Step 1: Use feedback loop to get the best documents
        feedback_result = self.feedback_loop.retrieve_with_feedback_loop(query, category)
        
        if not feedback_result["documents"]:
            print(f"[INFO] ❌ Feedback loop found no relevant documents. Reason: {feedback_result.get('failure_reason', 'Unknown')}")
            return {"response": "I could not find relevant information in my database for your query.", "sources": [], "contexts": []}
        
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
            return {"response": "I could not find sufficiently relevant information in my database.", "sources": [], "contexts": []}

        docs = self._prepare_docs(filtered)

        # Step 3: Generate answer using the best query from feedback loop
        print("[INFO] ✅ Running LLM to generate answer...")
        result = self.qa_chain.invoke({"input": feedback_result["query_used"]})
        answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
        print("[INFO] ✅ Answer generated successfully.")

        # Step 4: Validate the generated answer
        context_texts = [d.page_content for d in docs]
        validation = self.rag_utils.validate_answer(feedback_result["query_used"], answer, context_texts)
        print(f"[INFO] ✅ Answer validation - Valid: {validation['valid']}, Confidence: {validation['confidence']:.2f}")
        
        # If answer is not valid according to validation, provide a fallback response
        # Lowered confidence threshold from 0.3 to 0.15 to be less conservative
        if not validation['valid'] or validation['confidence'] < 0.15:
            # Even with low confidence, try to provide a helpful answer with a disclaimer
            if validation['confidence'] > 0.05 and answer and len(answer.strip()) > 20:
                answer = f"{answer}\n\n⚠️ *Note: I have moderate confidence in this answer. Please verify the information with official sources or consult a financial advisor for specific advice.*"
            else:
                answer = "I'm not confident in my answer based on the available information. Please rephrase your question or ask about a different topic."

        response = {
            "response": answer,
            "sources": [{"file": d.metadata.get("source", "Unknown")} for d in docs],
            "contexts": context_texts,
            "feedback_loop_metadata": {
                "iterations_used": feedback_result['total_iterations'],
                "final_query": feedback_result['query_used'],
                "original_query": feedback_result['original_query'],
                "relevance_score": feedback_result['relevance_score'],
                "refinement_history": feedback_result.get('refinement_history', [])
            }
        }
        self.cache.set(query, response)
        return response
    
    def _process_query_traditional(self, query: str, category: str) -> Dict:
        """Traditional query processing (fallback when feedback loop is not available)"""
        print("[INFO] 🔧 Using traditional RAG approach...")
        
        # Refine the query for better retrieval
        refined_query = self.rag_utils.refine_query(query)
        print(f"[INFO] 🔧 Refined query: {refined_query}")

        # Retrieve documents using refined query
        retrieved = self.vectorstore.similarity_search(refined_query, k=MAX_DOCS_FOR_RETRIEVAL)
        
        # Check relevance of retrieved documents
        is_relevant, confidence = self.rag_utils.check_query_relevance(refined_query, retrieved)
        print(f"[INFO] 🔍 Query relevance - Relevant: {is_relevant}, Confidence: {confidence:.2f}")
        
        if not is_relevant or confidence < 0.1:  # Low relevance threshold
            print("[INFO] ❌ Retrieved documents not relevant to query.")
            return {"response": "I could not find relevant information in my database for your query.", "sources": [], "contexts": []}

        filtered = self._rank_and_filter(retrieved, refined_query)
        if not filtered:
            print("[INFO] ❌ No relevant documents found.")
            return {"response": "I could not find relevant information in my database.", "sources": [], "contexts": []}

        docs = self._prepare_docs(filtered)

        print("[INFO] ✅ Running LLM to generate answer...")
        result = self.qa_chain.invoke({"input": refined_query})
        answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
        print("[INFO] ✅ Answer generated successfully.")

        # Validate the generated answer
        context_texts = [d.page_content for d in docs]
        validation = self.rag_utils.validate_answer(refined_query, answer, context_texts)
        print(f"[INFO] ✅ Answer validation - Valid: {validation['valid']}, Confidence: {validation['confidence']:.2f}")
        
        # If answer is not valid according to validation, provide a fallback response
        # Lowered confidence threshold from 0.3 to 0.15 to be less conservative
        if not validation['valid'] or validation['confidence'] < 0.15:
            # Even with low confidence, try to provide a helpful answer with a disclaimer
            if validation['confidence'] > 0.05 and answer and len(answer.strip()) > 20:
                answer = f"{answer}\n\n⚠️ *Note: I have moderate confidence in this answer. Please verify the information with official sources or consult a financial advisor for specific advice.*"
            else:
                answer = "I'm not confident in my answer based on the available information. Please rephrase your question or ask about a different topic."

        response = {
            "response": answer,
            "sources": [{"file": d.metadata.get("source", "Unknown")} for d in docs],
            "contexts": context_texts,
            "feedback_loop_metadata": {
                "iterations_used": 1,
                "final_query": refined_query,
                "original_query": query,
                "relevance_score": confidence,
                "refinement_history": []
            }
        }
        self.cache.set(query, response)
        return response

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
        for doc_idx, (file, chunks) in enumerate(grouped.items(), 1):
            organized_output += f"\n📂 **Document {doc_idx}: {file}**\n"
            for idx, chunk in enumerate(chunks, 1):
                organized_output += f"\n🔹 Chunk {idx}:\n{chunk}\n"

        await send_in_chunks(update, organized_output)

    print("[INFO] ✅ Response sent to user.")

# --- Run Bot ---
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_TOKEN environment variable is required. Please set it in your .env file.")
    
    print("[INFO] 🚀 Starting Telegram Financial Advisor Bot...")
    app = ApplicationBuilder().token(token).build()
    logger.info("Bot started successfully.")
    print("[INFO] ✅ Telegram Bot is now polling for messages...")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    app.run_polling()
