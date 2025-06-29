import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import re

# Configuration
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_MODEL = "llama3.2:3b"
TELEGRAM_TOKEN = "7283974888:AAHLS1jodnbWxA-fqIz9YpPmpmdKcef7skw"  # Replace with your token

# --- Markdown Escaping Function ---
def escape_markdown(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

# Load FAISS index and setup QA chain
print("[INFO] Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"}  # Or "cpu" if no GPU
)

vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
llm = OllamaLLM(model=OLLAMA_MODEL)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("How can I help you with your finances?")

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text.strip()
    if not user_query:
        await update.message.reply_text("❗ Please send a non-empty question.")
        return

    try:
        response = qa_chain.invoke(user_query)

        # Format answer
        result_text = escape_markdown(f"🔍 *Answer*\n{response['result']}")
        await update.message.reply_text(result_text, parse_mode="MarkdownV2")

        # Format sources
        if response['source_documents']:
            sources_text = escape_markdown("\n📚 *Sources*")
            for i, doc in enumerate(response['source_documents'], 1):
                preview = escape_markdown(doc.page_content[:300].replace('\n', ' '))
                page = escape_markdown(str(doc.metadata.get('page', 'Unknown')))
                source = escape_markdown(str(doc.metadata.get('source', 'Unknown')))
                sources_text += f"\n\n*Source {i}* — Page: {page}, File: {source}\n_{preview}_"

            await update.message.reply_text(sources_text, parse_mode="MarkdownV2")

        # Follow-up question
        await update.message.reply_text("Are there any other issues I can help you with?")

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}")

# Main bot setup
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    print("[INFO] Bot is running. Press Ctrl+C to stop.")
    app.run_polling()
