import logging
import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from main import FinancialAdvisorBot
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot API Key from environment variable
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_TOKEN')

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not found in environment variables. Please check your .env file.")

# Initialize the FinancialAdvisorBot
financial_bot = FinancialAdvisorBot()

# Dictionary to store user sessions (for context management)
user_sessions = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I am FinAuxiBOT, your financial advisor. How can I help you today?"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message when the command /help is issued."""
    await update.message.reply_text("You can ask me questions about banking, investments, loans, and taxation.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages and provide financial advice."""
    user_id = update.effective_user.id
    user_message = update.message.text

    logger.info(f"Received message from {user_id}: {user_message}")

    # Get or create user session
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    
    # Add current message to session history
    user_sessions[user_id].append({"role": "user", "content": user_message})

    try:
        # Send typing indicator
        await update.message.reply_chat_action("typing")
        
        # Process the query using the FinancialAdvisorBot
        response = financial_bot.process_query(user_message)

        if 'error' in response:
            await update.message.reply_text(f"❌ Error: {response['error']}")
        else:
            # Handle the new response format from main.py
            answer = response.get('response', 'No response generated')
            sources = response.get('sources', [])
            disclaimer = response.get('disclaimer', financial_bot.LEGAL_DISCLAIMER)
            category = response.get('category', 'general')

            # Format sources for Telegram
            sources_text = ""
            if sources:
                sources_text = "\n\n📚 Sources:\n"
                for i, source in enumerate(sources[:3], 1):  # Limit to 3 sources to avoid message length issues
                    source_name = source.get('name', 'Unknown')
                    page = source.get('page', 'N/A')
                    sources_text += f"{i}. {source_name} (Page: {page})\n"
            
            # Create the full response with length limit for Telegram (4096 chars max)
            full_response = f"{answer}{sources_text}\n\n⚠️ {disclaimer}"
            
            # Split message if too long
            if len(full_response) > 4000:
                # Send answer first
                await update.message.reply_text(answer)
                
                # Send sources if available
                if sources_text:
                    await update.message.reply_text(sources_text)
                
                # Send disclaimer
                await update.message.reply_text(f"⚠️ {disclaimer}")
            else:
                await update.message.reply_text(full_response)

            # Add bot's response to session history
            user_sessions[user_id].append({"role": "bot", "content": answer})
            
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await update.message.reply_text("❌ I encountered an error while processing your request. Please try again.")

def main() -> None:
    """Start the bot."""
    # Create the Application and pass your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    logger.info("Bot is running. Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
