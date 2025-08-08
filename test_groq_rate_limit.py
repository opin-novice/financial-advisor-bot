#!/usr/bin/env python3
"""
Quick test script to verify Groq rate limiting works
"""
import time
import pytest
from langchain_groq import ChatGroq

# Same settings as eval.py
GROQ_MODEL = "llama3-8b-8192"
MAX_TOKENS = 500
DELAY_SECONDS = 40

# Get API key from environment
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def test_rate_limiting():
    """Test basic rate limiting with Groq"""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("⚠️ GROQ_API_KEY not found in environment or using template value")
        pytest.skip("GROQ_API_KEY not found in environment or using template value")
        
    print("🧪 Testing Groq Rate Limiting")
    print(f"📊 Model: {GROQ_MODEL}")
    print(f"⚙️ Max tokens: {MAX_TOKENS}, Delay: {DELAY_SECONDS}s")
    print("=" * 50)
    
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            model_kwargs={"top_p": 0.9}
        )
        
        # Test with 2 simple requests
        test_questions = [
            "What is 2+2?",
            "What is the capital of Bangladesh?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔹 Test {i}/2: {question}")
            
            try:
                # Add delay before each request (except first)
                if i > 1:
                    print(f"⏳ Waiting {DELAY_SECONDS}s...")
                    time.sleep(DELAY_SECONDS)
                
                response = llm.invoke(question)
                print(f"✅ Response: {response.content[:100]}...")
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    print(f"🔄 Rate limit hit: {e}")
                    print("⏳ Waiting 60s before retry...")
                    time.sleep(60)
                    
                    # Retry once
                    try:
                        response = llm.invoke(question)
                        print(f"✅ Retry successful: {response.content[:100]}...")
                    except Exception as retry_e:
                        print(f"❌ Retry failed: {retry_e}")
                        pytest.fail(f"Retry failed: {retry_e}")
                else:
                    print(f"❌ Non-rate-limit error: {e}")
                    pytest.fail(f"Non-rate-limit error: {e}")
        
        print("\n🎉 Rate limiting test passed!")
        assert True  # Test passed
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        pytest.fail(f"Test setup failed: {e}")

if __name__ == "__main__":
    success = test_rate_limiting()
    if success:
        print("\n✅ Ready to run full RAGAS evaluation!")
        print("Run: python eval.py")
    else:
        print("\n⚠️  Fix rate limiting issues before running eval.py") 