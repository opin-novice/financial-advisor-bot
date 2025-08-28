import os
import time
from typing import Dict, List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# --- Config ---
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Ollama Configuration
MAIN_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG Settings
MAX_DOCS_FOR_RETRIEVAL = 10
MAX_DOCS_FOR_CONTEXT = 6
CONTEXT_CHUNK_SIZE = 1000

# =============================================================================
# 🏦 VANILLA RAG SYSTEM
# =============================================================================

class VanillaRAGSystem:
    """Simple vanilla RAG system with basic retrieval and generation"""
    
    def __init__(self):
        self._init_rag()

    def _init_rag(self):
        print("[INFO] 🚀 Initializing Vanilla RAG system...")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, 
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )
        print(f"[INFO] Loading FAISS index from: {FAISS_INDEX_PATH}")
        self.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
        print("[INFO] ✅ FAISS index loaded successfully.")

        # Initialize LLM
        self.llm = ChatOllama(
            model=MAIN_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        print(f"[INFO] ✅ LLM initialized: {MAIN_MODEL}")

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": MAX_DOCS_FOR_RETRIEVAL})
        
        # Create simple prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful financial advisor. Answer the question based on the provided context.

Context:
{context}

Question: {input}

Answer:""",
            input_variables=["context", "input"]
        )
        
        # Create RAG chain
        self.doc_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        self.qa_chain = create_retrieval_chain(self.retriever, self.doc_chain)
        
        print("[INFO] 🎉 Vanilla RAG system ready!")

    def process_query(self, query: str) -> Dict:
        """
        Simple vanilla RAG pipeline: Retrieve → Generate
        """
        print(f"[INFO] 🚀 Processing query: {query}")
        start_time = time.time()

        try:
            # Use the RAG chain directly
            result = self.qa_chain.invoke({"input": query})
            answer = result.get("answer") or result.get("result") or result.get("output_text") or str(result)
            
            # Get source documents
            source_docs = result.get("source_documents", [])
            context_texts = [d.page_content for d in source_docs[:MAX_DOCS_FOR_CONTEXT]]
            processing_time = time.time() - start_time

            response = {
                "response": answer,
                "sources": [{"file": d.metadata.get("source", "Unknown")} for d in source_docs[:MAX_DOCS_FOR_CONTEXT]],
                "contexts": context_texts,
                "original_query": query,
                "num_docs": len(source_docs),
                "processing_time": round(processing_time, 2)
            }
            
            print(f"[INFO] ✅ RAG completed in {processing_time:.2f}s")
            return response

        except Exception as e:
            print(f"[ERROR] {e}")
            return {
                "response": f"I apologize, but I encountered an error: {e}",
                "sources": [], 
                "contexts": [],
                "original_query": query,
                "num_docs": 0,
                "processing_time": 0
            }

# --- Telegram Bot ---
bot_instance = VanillaRAGSystem()

async def send_in_chunks(update: Update, text: str):
    MAX_LEN = 4000
    for i in range(0, len(text), MAX_LEN):
        await update.message.reply_text(text[i:i+MAX_LEN])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = """
🏦 **Financial Advisor Bot**

I'm a simple AI financial advisor that can help you with:

💰 **Banking & Loans**
- How to get loans
- Bank account information
- Financial requirements

📊 **Investments**  
- Investment options
- Mutual funds guidance

🏛️ **Tax Services**
- Tax filing information
- Tax calculation help

🛡️ **Insurance**
- Insurance information

Just ask me any financial question and I'll help you!
"""
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text.strip()
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    
    if not user_query:
        await update.message.reply_text("Please ask a financial question!")
        return

    print(f"[INFO] 👤 User {username} asked: {user_query}")
    
    # Show processing message
    processing_msg = await update.message.reply_text("🤔 Thinking...")
    
    try:
        # Process query
        result = bot_instance.process_query(user_query)
        
        # Prepare response
        response_text = result["response"]
        
        # Add sources if available
        if result["sources"]:
            sources_text = "\n\n📚 Sources:\n"
            for i, source in enumerate(result["sources"][:3], 1):
                source_name = os.path.basename(source["file"]) if source["file"] != "Unknown" else "Unknown"
                sources_text += f"{i}. {source_name}\n"
            response_text += sources_text
        
        # Send response
        await processing_msg.delete()
        await send_in_chunks(update, response_text)
        
    except Exception as e:
        print(f"[ERROR] Failed to process query: {e}")
        await processing_msg.edit_text("❌ Sorry, I encountered an error processing your question. Please try again.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
🤖 **Financial Advisor Bot Help**

**Commands:**
/start - Start the bot
/help - Show this help message

**How to use:**
Simply ask me any financial question and I'll provide an answer based on my knowledge base.

**Example questions:**
• "What documents do I need for a bank account?"
• "How to apply for a car loan?"
• "What are the tax filing requirements?"
• "Tell me about investment options"

I'm here to help with your financial questions! 💰
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

def main():
    """Initialize and run the Telegram bot"""
    print("🚀 Starting Vanilla RAG Financial Advisor Bot...")
    
    # Get bot token from environment
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        print("❌ Error: TELEGRAM_BOT_TOKEN environment variable not set!")
        return
    
    # Create application
    application = ApplicationBuilder().token(bot_token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    
    print("✅ Bot handlers registered")
    print("🤖 Starting bot...")
    
    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
