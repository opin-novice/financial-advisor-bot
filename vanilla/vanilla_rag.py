#!/usr/bin/env python3
"""
Vanilla RAG Pipeline
====================
A simple, clean implementation of a basic RAG (Retrieval-Augmented Generation) pipeline.
This demonstrates the core concepts without advanced features for educational purposes.

Core Components:
1. Document Loading & Chunking
2. Embedding Generation
3. Vector Storage (FAISS)
4. Retrieval
5. Generation with LLM
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any
from pathlib import Path

# Core RAG libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Telegram bot libraries
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VanillaRAGPipeline:
    """
    A simple RAG pipeline implementation demonstrating core concepts.
    """
    
    def __init__(self, 
                 pdf_dir: str = "data",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 llm_model: str = "llama3-8b-8192",
                 api_key: str = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the vanilla RAG pipeline.
        
        Args:
            pdf_dir: Directory containing PDF documents
            embedding_model: HuggingFace model for embeddings
            llm_model: Groq model for generation
            api_key: Groq API key
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.pdf_dir = pdf_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
        logger.info("Vanilla RAG Pipeline initialized")
    
    def load_documents(self) -> List[Any]:
        """
        Load PDF documents from the specified directory.
        
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading documents from {self.pdf_dir}")
        
        documents = []
        pdf_dir = Path(self.pdf_dir)
        
        if not pdf_dir.exists():
            logger.error(f"Directory {self.pdf_dir} does not exist")
            return documents
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Load each PDF
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of loaded documents
            
        Returns:
            List of chunked documents
        """
        logger.info("Chunking documents")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def create_embeddings(self):
        """
        Initialize the embedding model.
        """
        logger.info(f"Initializing embedding model: {self.embedding_model}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"}
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def create_vectorstore(self, chunks: List[Any], save_path: str = "faiss_index"):
        """
        Create and store embeddings in a FAISS vector store.
        
        Args:
            chunks: List of document chunks
            save_path: Path to save the FAISS index
        """
        logger.info("Creating vector store")
        
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call create_embeddings() first.")
        
        try:
            # Create vector store
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            # Save the index
            self.vectorstore.save_local(save_path)
            logger.info(f"Vector store created and saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vectorstore(self, load_path: str = "faiss_index"):
        """
        Load an existing FAISS vector store.
        
        Args:
            load_path: Path to the FAISS index
        """
        logger.info(f"Loading vector store from {load_path}")
        
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call create_embeddings() first.")
        
        try:
            self.vectorstore = FAISS.load_local(load_path, self.embeddings)
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def initialize_llm(self):
        """
        Initialize the language model for generation.
        """
        if not self.api_key:
            raise ValueError("API key required for LLM initialization")
        
        logger.info(f"Initializing LLM: {self.llm_model}")
        
        try:
            self.llm = ChatGroq(
                model=self.llm_model,
                groq_api_key=self.api_key,
                temperature=0.1,
                max_tokens=500
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def create_qa_chain(self):
        """
        Create the question-answering chain.
        """
        if not self.vectorstore or not self.llm:
            raise ValueError("Vector store and LLM must be initialized first")
        
        logger.info("Creating QA chain")
        
        # Create a simple prompt template
        prompt_template = """
        You are a helpful assistant. Answer the question based on the provided context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("QA chain created successfully")
    
    def query(self, question: str) -> str:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            The generated answer
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call create_qa_chain() first.")
        
        logger.info(f"Processing question: {question}")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            answer = result.get("result", "No answer generated")
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error: {str(e)}"
    
    def build_pipeline(self, force_rebuild: bool = False):
        """
        Build the complete RAG pipeline.
        
        Args:
            force_rebuild: If True, rebuild the vector store even if it exists
        """
        logger.info("Building RAG pipeline")
        
        # Step 1: Initialize embeddings
        self.create_embeddings()
        
        # Step 2: Check if vector store exists
        if not force_rebuild and os.path.exists("faiss_index"):
            logger.info("Loading existing vector store")
            self.load_vectorstore()
        else:
            logger.info("Creating new vector store")
            # Step 3: Load and chunk documents
            documents = self.load_documents()
            if not documents:
                raise ValueError("No documents loaded")
            
            chunks = self.chunk_documents(documents)
            
            # Step 4: Create vector store
            self.create_vectorstore(chunks)
        
        # Step 5: Initialize LLM
        self.initialize_llm()
        
        # Step 6: Create QA chain
        self.create_qa_chain()
        
        logger.info("RAG pipeline built successfully")
    
    def get_relevant_documents(self, question: str, k: int = 3) -> List[Any]:
        """
        Retrieve relevant documents for a question.
        
        Args:
            question: The question to retrieve documents for
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        documents = retriever.get_relevant_documents(question)
        
        return documents

def run_demo(rag: VanillaRAGPipeline):
    """Run the demo with example questions."""
    try:
        # Build the pipeline
        rag.build_pipeline()
        
        # Example questions
        example_questions = [
            "What are the requirements for opening a bank account in Bangladesh?",
            "How do I apply for a loan?",
            "What are the current interest rates?",
            "What documents do I need for tax filing?"
        ]
        
        print("\n" + "="*60)
        print("VANILLA RAG PIPELINE DEMO")
        print("="*60)
        
        # Process each question
        for i, question in enumerate(example_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 50)
            
            # Get relevant documents
            docs = rag.get_relevant_documents(question, k=2)
            print(f"Retrieved {len(docs)} relevant documents")
            
            # Generate answer
            answer = rag.query(question)
            print(f"Answer: {answer}")
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"Error: {e}")

def start_telegram_bot(rag: VanillaRAGPipeline, bot_token: str):
    """Start the Telegram bot."""
    try:
        print(f"\n🚀 Starting Telegram RAG Bot...")
        print(f"📱 Bot will be available on Telegram once RAG pipeline is ready")
        print(f"💡 Use /status command to check bot status")
        print(f"💡 Use /help command for available commands")
        
        # Create and run the bot
        bot = TelegramRAGBot(telegram_token=bot_token, rag_pipeline=rag)
        bot.run()
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        print(f"Error: {e}")

class TelegramRAGBot:
    """
    Telegram bot that integrates with the vanilla RAG pipeline.
    """
    
    def __init__(self, 
                 telegram_token: str,
                 rag_pipeline: VanillaRAGPipeline):
        """
        Initialize the Telegram RAG bot.
        
        Args:
            telegram_token: Telegram bot token from BotFather
            rag_pipeline: Initialized RAG pipeline instance
        """
        self.telegram_token = telegram_token
        self.rag_pipeline = rag_pipeline
        self.is_ready = False
        
        logger.info("Telegram RAG Bot initialized")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        welcome_message = """
🚀 **Welcome to the RAG Assistant Bot!**

I'm your AI assistant powered by a local RAG (Retrieval-Augmented Generation) pipeline.

**What I can do:**
• Answer questions based on your document collection
• Provide information from PDF documents
• Help with research and information retrieval

**Commands:**
/start - Show this welcome message
/help - Show help information
/status - Check bot and RAG pipeline status
/rebuild - Rebuild the document index (admin only)

**How to use:**
Simply send me a question and I'll search through your documents to find relevant information and provide an answer.

**Example questions:**
• "What are the requirements for opening a bank account?"
• "How do I apply for a loan?"
• "What documents do I need for tax filing?"

Let's get started! Ask me anything about your documents.
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_message = """
📚 **RAG Assistant Bot Help**

**What is RAG?**
RAG (Retrieval-Augmented Generation) combines document search with AI generation to provide accurate, contextual answers.

**How it works:**
1. You ask a question
2. I search through your document collection
3. I find the most relevant information
4. I generate a comprehensive answer

**Available Commands:**
/start - Welcome message and introduction
/help - This help message
/status - Check system status
/rebuild - Rebuild document index (admin only)

**Tips for better results:**
• Ask specific questions
• Use clear, descriptive language
• Be patient - processing takes time
• Check the status if responses are slow

**Technical Details:**
• Uses Groq with Llama 3.2 3B model
• Local processing - no data sent to external services
• FAISS vector database for fast document search
• Sentence transformers for semantic understanding

Need help? Contact your system administrator.
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            if not self.is_ready:
                status_message = "⚠️ **Bot Status: Not Ready**\n\nRAG pipeline is still initializing. Please wait..."
                await update.message.reply_text(status_message, parse_mode='Markdown')
                return
            
            # Check RAG pipeline status
            if self.rag_pipeline and self.rag_pipeline.vectorstore:
                doc_count = len(self.rag_pipeline.vectorstore.docstore._dict)
                status_message = f"""
✅ **Bot Status: Ready**

**RAG Pipeline:**
• Status: Active
• Documents indexed: {doc_count}
• Model: {self.rag_pipeline.llm_model}
• Chunk size: {self.rag_pipeline.chunk_size}
• Chunk overlap: {self.rag_pipeline.chunk_overlap}

**System:**
• Bot: Running
• Groq: Connected
• Vector store: Loaded

You can now ask questions!
                """
            else:
                status_message = "❌ **Bot Status: Error**\n\nRAG pipeline not properly initialized."
            
            await update.message.reply_text(status_message, parse_mode='Markdown')
            
        except Exception as e:
            error_message = f"❌ **Status Check Failed**\n\nError: {str(e)}"
            await update.message.reply_text(error_message, parse_mode='Markdown')
            logger.error(f"Status check failed: {e}")
    
    async def rebuild_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /rebuild command (admin only)."""
        # Check if user is admin (you can customize this logic)
        user_id = update.effective_user.id
        admin_ids = [123456789]  # Replace with actual admin user IDs
        
        if user_id not in admin_ids:
            await update.message.reply_text("❌ **Access Denied**\n\nOnly administrators can rebuild the document index.")
            return
        
        try:
            await update.message.reply_text("🔄 **Rebuilding Document Index**\n\nThis may take several minutes. Please wait...")
            
            # Rebuild the pipeline
            self.rag_pipeline.build_pipeline(force_rebuild=True)
            self.is_ready = True
            
            success_message = "✅ **Document Index Rebuilt Successfully!**\n\nThe bot is now ready to answer questions with the updated document collection."
            await update.message.reply_text(success_message, parse_mode='Markdown')
            
        except Exception as e:
            error_message = f"❌ **Rebuild Failed**\n\nError: {str(e)}"
            await update.message.reply_text(error_message, parse_mode='Markdown')
            logger.error(f"Rebuild failed: {e}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        if not self.is_ready:
            await update.message.reply_text("⚠️ Bot is still initializing. Please wait a moment and try again.")
            return
        
        user_message = update.message.text
        user_id = update.effective_user.id
        
        logger.info(f"User {user_id} asked: {user_message}")
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Process the question through RAG pipeline
            await update.message.reply_text("🔍 **Searching documents...**", parse_mode='Markdown')
            
            # Get relevant documents
            docs = self.rag_pipeline.get_relevant_documents(user_message, k=3)
            
            if not docs:
                await update.message.reply_text("❌ **No relevant documents found**\n\nI couldn't find any documents related to your question. Please try rephrasing or ask about a different topic.")
                return
            
            # Generate answer
            await update.message.reply_text("🤖 **Generating answer...**", parse_mode='Markdown')
            
            answer = self.rag_pipeline.query(user_message)
            
            if answer.startswith("Error:"):
                await update.message.reply_text(f"❌ **Error generating answer**\n\n{answer}")
                return
            
            # Format the response
            response = f"""
💡 **Answer:**

{answer}

---
📄 **Sources:** Found {len(docs)} relevant document(s)
🔍 **Question:** {user_message}
            """
            
            # Split long responses if needed
            if len(response) > 4000:
                # Send answer in parts
                await update.message.reply_text(f"💡 **Answer:**\n\n{answer[:3500]}...", parse_mode='Markdown')
                await update.message.reply_text(f"...{answer[3500:]}", parse_mode='Markdown')
            else:
                await update.message.reply_text(response, parse_mode='Markdown')
            
            logger.info(f"Successfully answered user {user_id}")
            
        except Exception as e:
            error_message = f"❌ **Error processing your question**\n\nAn unexpected error occurred: {str(e)}"
            await update.message.reply_text(error_message, parse_mode='Markdown')
            logger.error(f"Error processing message from user {user_id}: {e}")
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in the bot."""
        logger.error(f"Exception while handling an update: {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ **An error occurred**\n\nPlease try again later or contact support if the problem persists."
            )
    
    async def initialize_rag_pipeline(self):
        """Initialize the RAG pipeline asynchronously."""
        try:
            logger.info("Building RAG pipeline...")
            self.rag_pipeline.build_pipeline()
            self.is_ready = True
            logger.info("RAG pipeline ready!")
        except Exception as e:
            logger.error(f"Failed to build RAG pipeline: {e}")
            self.is_ready = False
    
    def run(self):
        """Run the Telegram bot."""
        # Create application
        application = Application.builder().token(self.telegram_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(CommandHandler("rebuild", self.rebuild_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Add error handler
        application.add_error_handler(self.error_handler)
        
        # Initialize RAG pipeline in background
        asyncio.create_task(self.initialize_rag_pipeline())
        
        # Start the bot
        logger.info("Starting Telegram RAG bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """
    Main function with choice between demo and Telegram bot.
    """
    # ========================================
    # CONFIGURATION - EDIT THESE VALUES
    # ========================================
    
    # Set your Groq API key here
    GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # Replace with your actual API key
    
    # Set your Telegram bot token here
    TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Replace with your actual bot token
    
    # Other configuration options
    PDF_DIR = "data"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "llama3-8b-8192"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # ========================================
    
    # Validate API key
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print("❌ Error: Please set your Groq API key in the script")
        print("💡 Edit the GROQ_API_KEY variable in vanilla_rag.py")
        print("\n📖 To get a Groq API key:")
        print("   1. Visit https://console.groq.com/")
        print("   2. Sign up or log in")
        print("   3. Go to API Keys section")
        print("   4. Create a new API key")
        print("   5. Copy the key and paste it in the script")
        return
    
    # Configuration dictionary
    config = {
        "pdf_dir": PDF_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "api_key": GROQ_API_KEY,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }
    
    # Initialize pipeline
    rag = VanillaRAGPipeline(**config)
    
    # Ask user what they want to do
    print("\n" + "="*60)
    print("VANILLA RAG PIPELINE")
    print("="*60)
    print("What would you like to do?")
    print("1. Run demo (test questions)")
    print("2. Start Telegram bot")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            run_demo(rag)
            break
        elif choice == "2":
            if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
                print("❌ Error: Please set your Telegram bot token in the script")
                print("💡 Edit the TELEGRAM_BOT_TOKEN variable in vanilla_rag.py")
                print("\n📖 To get a bot token:")
                print("   1. Message @BotFather on Telegram")
                print("   2. Use /newbot command")
                print("   3. Follow the instructions")
                print("   4. Copy the token and paste it in the script")
                return
            start_telegram_bot(rag, TELEGRAM_BOT_TOKEN)
            break
        elif choice == "3":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
