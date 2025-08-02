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
OLLAMA_MODEL = "llama3.2:3b"
CACHE_TTL = 86400  # 24 hours

# Retrieval Settings
MAX_DOCS_FOR_RETRIEVAL = 12
MAX_DOCS_FOR_CONTEXT = 5
CONTEXT_CHUNK_SIZE = 1500

# --- Prompt (Bangladesh Context) ---
PROMPT_TEMPLATE = """
You are a financial assistant specialized in Bangladesh's financial system.
Always answer in the context of Bangladesh, using Bangladeshi Taka (৳/Tk) as the currency.

Use the following context to answer clearly. 
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{input}

Provide a helpful, fact-based financial answer:
"""
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

        self.doc_chain = create_stuff_documents_chain(self.llm, QA_PROMPT)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": MAX_DOCS_FOR_RETRIEVAL})
        self.qa_chain = create_retrieval_chain(retriever, self.doc_chain)

    def _rank_and_filter(self, docs: List[Document], query: str) -> List[Document]:
        terms = set(query.lower().split())
        scored = [(doc, sum(1 for t in terms if t in doc.page_content.lower())) for doc in docs]
        return [d for d, s in sorted(scored, key=lambda x: x[1], reverse=True) if s > 0]

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
