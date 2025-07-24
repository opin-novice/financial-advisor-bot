#!/usr/bin/env python3
"""
Test script for Telegram Bot functionality
"""

import os
from dotenv import load_dotenv
from main import FinancialAdvisorBot

# Load environment variables
load_dotenv()

def test_bot_initialization():
    """Test if the bot initializes correctly"""
    print("🔄 Testing bot initialization...")
    try:
        bot = FinancialAdvisorBot()
        print("✅ Bot initialized successfully!")
        return bot
    except Exception as e:
        print(f"❌ Bot initialization failed: {e}")
        return None

def test_query_processing(bot):
    """Test query processing functionality"""
    print("🔄 Testing query processing...")
    
    test_queries = [
        "What are the requirements for opening a bank account?",
        "How much can I borrow for a car loan?",
        "What are the tax rates in Bangladesh?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")
        try:
            response = bot.process_query(query)
            
            if 'error' in response:
                print(f"❌ Error: {response['error']}")
            elif 'response' in response:
                print(f"✅ Success!")
                print(f"   Category: {response.get('category', 'Unknown')}")
                print(f"   Sources: {len(response.get('sources', []))}")
                print(f"   Response preview: {response['response'][:100]}...")
            else:
                print(f"⚠️  Unexpected response format: {response}")
                
        except Exception as e:
            print(f"❌ Query processing failed: {e}")

def test_telegram_token():
    """Test if Telegram token is properly configured"""
    print("🔄 Testing Telegram token configuration...")
    
    token = os.getenv('TELEGRAM_TOKEN')
    if token:
        print("✅ Telegram token found in environment!")
        print(f"   Token preview: {token[:10]}...{token[-10:]}")
    else:
        print("❌ Telegram token not found in environment variables!")
        return False
    return True

def main():
    print("🤖 Telegram Bot Test Suite")
    print("=" * 50)
    
    # Test 1: Telegram token
    if not test_telegram_token():
        print("\n❌ Critical: Telegram token missing. Bot cannot start.")
        return
    
    # Test 2: Bot initialization
    bot = test_bot_initialization()
    if not bot:
        print("\n❌ Critical: Bot initialization failed. Cannot proceed.")
        return
    
    # Test 3: Query processing
    test_query_processing(bot)
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\n📋 Next steps:")
    print("   1. Run 'python3 telegram_bot_main.py' to start the bot")
    print("   2. Message your bot on Telegram to test it live")
    print("   3. Use /start to begin and /help for assistance")

if __name__ == "__main__":
    main()
