import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Import the refactored bot core
from bot_core import FinancialAdvisorBotCore

# --- Logging ---
logging.basicConfig(
    filename='logs/telegram_financial_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialAdvisorTelegramBot:
    """Telegram bot interface for the financial advisor"""
    
    def __init__(self):
        self.bot_core = FinancialAdvisorBotCore()
    
    async def send_in_chunks(self, update: Update, text: str, chunk_size: int = 4000):
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

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command with language detection"""
        # Detect language preference from user's message if any
        user_language = 'english'  # Default
        
        # Check if user sent any text with the start command
        if context.args:
            sample_text = ' '.join(context.args)
            detected_lang, _ = self.bot_core.language_detector.detect_language(sample_text)
            user_language = detected_lang
        
        # Send welcome message in appropriate language
        if user_language == 'bengali':
            welcome_message = "হ্যালো! আমাকে যেকোনো আর্থিক প্রশ্ন করুন। আমি বাংলা এবং ইংরেজি দুই ভাষাতেই উত্তর দিতে পারি।"
        else:
            welcome_message = "Hi! Ask me any financial question. I can respond in both English and Bangla based on your question language."
        
        await update.message.reply_text(welcome_message)

    async def handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced query handler with language detection and bilingual response"""
        user_query = update.message.text.strip()
        if not user_query:
            # Detect language for error message
            error_message = self.bot_core.language_detector.translate_system_messages(
                "Please enter a valid question.", 'english'
            )
            await update.message.reply_text(error_message)
            return

        print(f"[INFO] 👤 User asked: {user_query}")
        
        # Detect language and show processing message in appropriate language
        detected_language, confidence = self.bot_core.language_detector.detect_language(user_query)
        processing_message = self.bot_core.language_detector.translate_system_messages(
            "Processing your question...", detected_language
        )
        await update.message.reply_text(processing_message)
        
        # Process the query
        response = self.bot_core.process_query(user_query)

        # ✅ Send the final answer
        answer = response.get("response") if isinstance(response, dict) else str(response)
        
        # Enhance answer with language-specific formatting if needed
        if isinstance(response, dict):
            detected_lang = response.get('detected_language', detected_language)
            lang_confidence = response.get('language_confidence', confidence)
            
            # Add confidence disclaimer if needed and validation confidence is low
            validation_confidence = response.get('validation_confidence', 1.0)
            if validation_confidence < 0.3:
                confidence_msg = self.bot_core.language_detector.format_confidence_message(detected_lang)
                if confidence_msg not in answer:
                    answer += confidence_msg
        
        await self.send_in_chunks(update, answer)

        # ✅ Organize chunks by source file with language-appropriate headers
        if isinstance(response, dict) and response.get("sources") and response.get("contexts"):
            detected_lang = response.get('detected_language', detected_language)
            
            grouped = {}
            for i, src in enumerate(response["sources"]):
                filename = src["file"]
                grouped.setdefault(filename, []).append(response["contexts"][i])

            # ✅ Build organized output with language-appropriate formatting
            organized_output = self.bot_core.response_formatter.format_sources_section(detected_lang)
            
            for doc_idx, (file, chunks) in enumerate(grouped.items(), 1):
                organized_output += self.bot_core.response_formatter.format_document_header(
                    doc_idx, file, detected_lang
                )
                
                for idx, chunk in enumerate(chunks, 1):
                    organized_output += self.bot_core.response_formatter.format_chunk_header(
                        idx, detected_lang
                    )
                    organized_output += f"{chunk}\n"

            await self.send_in_chunks(update, organized_output)

        print("[INFO] ✅ Response sent to user.")

    def run(self):
        """Start the Telegram bot"""
        token = os.getenv("TELEGRAM_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_TOKEN environment variable is required. Please set it in your .env file.")
        
        print("[INFO] 🚀 Starting Telegram Financial Advisor Bot with Language Detection...")
        app = ApplicationBuilder().token(token).build()
        logger.info("Bot started successfully with bilingual support.")
        print("[INFO] ✅ Telegram Bot is now polling for messages...")
        print("[INFO] 🌐 Language detection enabled: English ⟷ Bangla")
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_query))
        app.run_polling()