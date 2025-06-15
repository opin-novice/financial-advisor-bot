import os
from dotenv import load_dotenv
import chardet

# Langchain community imports (for vectorstores and embeddings)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# Core langchain imports
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document  # This is the updated import for Document
from langchain.text_splitter import CharacterTextSplitter

# Ollama LLM (separate package)
from langchain_ollama import OllamaLLM

# Telegram imports
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes



load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Load LLaMA via Olama
llm = OllamaLLM(model="llama3")

# Initialize or load FAISS
def load_faiss():
    if os.path.exists("vectorstore/index.faiss"):
        return FAISS.load_local("vectorstore", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
    
    else:
        return None

vectorstore = load_faiss()
qa_chain = None

if vectorstore:
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ingest document
def ingest_document(file_path):
    # Read raw bytes and detect encoding
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    detected_encoding = chardet.detect(raw_bytes)['encoding']
    
    # Decode with detected encoding or fallback to utf-8, ignoring errors
    raw_text = raw_bytes.decode(detected_encoding or "utf-8", errors="ignore")
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([raw_text])
    
    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("vectorstore")
    return db

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me a question or upload a .txt file.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global qa_chain, vectorstore
    question = update.message.text
    if not qa_chain:
        await update.message.reply_text("Please upload a document first.")
        return
    answer = qa_chain.run(question)
    await update.message.reply_text(answer)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global qa_chain, vectorstore
    file = await update.message.document.get_file()
    file_path = f"documents/{file.file_unique_id}.txt"
    await file.download_to_drive(file_path)
    vectorstore = ingest_document(file_path)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    await update.message.reply_text("Document uploaded and processed.")

# Telegram bot setup
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()



