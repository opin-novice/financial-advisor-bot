#!/usr/bin/env python3
"""
Startup script for the Financial Advisor Telegram Bot
"""

import os
import sys
import signal
from dotenv import load_dotenv

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check .env file
    if not os.path.exists('.env'):
        print("❌ .env file not found! Please create one with your TELEGRAM_TOKEN.")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check Telegram token
    if not os.getenv('TELEGRAM_TOKEN'):
        print("❌ TELEGRAM_TOKEN not found in .env file!")
        return False
    
    # Check FAISS index
    if not os.path.exists('faiss_index'):
        print("❌ FAISS index not found! Please build the index first.")
        return False
    
    # Check if required modules can be imported
    try:
        from main import FinancialAdvisorBot
        from telegram_bot_main import main
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        return False
    
    print("✅ All prerequisites met!")
    return True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down bot gracefully...")
    sys.exit(0)

def main():
    print("🤖 Financial Advisor Telegram Bot")
    print("=" * 50)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    print("\n🚀 Starting Telegram bot...")
    print("📱 Your bot is now ready to receive messages!")
    print("💡 Use Ctrl+C to stop the bot")
    print("=" * 50)
    
    try:
        # Import and run the main bot function
        from telegram_bot_main import main as bot_main
        bot_main()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot crashed with error: {e}")
        print("📋 Check the logs in 'logs/financial_advisor_bot.log' for more details")
        sys.exit(1)

if __name__ == "__main__":
    main()
