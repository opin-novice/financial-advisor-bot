import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the refactored Telegram bot
from telegram_bot import FinancialAdvisorTelegramBot

# --- Run Bot ---
if __name__ == "__main__":
    bot = FinancialAdvisorTelegramBot()
    bot.run()