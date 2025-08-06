#!/usr/bin/env python3
"""
Test Bot Functionality Without Starting Telegram Bot
"""

import os
import asyncio
import warnings
warnings.filterwarnings("ignore")

# Set a dummy token for testing
os.environ['TG_TOKEN'] = 'dummy_token_for_testing'

async def test_bot_functionality():
    print("🧪 Testing Bot Functionality")
    print("=" * 40)
    
    try:
        # Import the main module
        from main import process_user_query, lang_processor
        
        print("✅ Bot modules imported successfully")
        
        # Test language detection
        test_cases = [
            ("What is investment?", "english"),
            ("¿Qué es la inversión?", "spanish"), 
            ("বিনিয়োগ কি?", "bangla")
        ]
        
        print("\n🔍 Testing Language Detection:")
        for query, expected_lang in test_cases:
            detected = lang_processor.detect_language(query)
            status = "✅" if detected == expected_lang else "⚠️"
            print(f"{status} '{query[:30]}...' -> {detected}")
        
        # Test a simple query processing
        print("\n💬 Testing Query Processing:")
        test_query = "What is a savings account?"
        
        print(f"Processing: '{test_query}'")
        
        # This will test the full pipeline
        result = await process_user_query(test_query)
        
        if result and 'answer' in result:
            print("✅ Query processing successful")
            print(f"📝 Answer preview: {result['answer'][:100]}...")
            print(f"🌐 Detected language: {result.get('detected_language', 'unknown')}")
            print(f"📚 Sources used: {len(result.get('sources', []))}")
        else:
            print("❌ Query processing failed - no answer generated")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 40)
    print("🎉 Functionality test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_bot_functionality())
