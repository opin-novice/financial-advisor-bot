#!/usr/bin/env python3
"""
Quick test script to verify Groq rate limiting works
"""
import time
from langchain_groq import ChatGroq

# Same settings as eval.py
GROQ_MODEL = "llama3-8b-8192"
GROQ_API_KEY = "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"
MAX_TOKENS = 500
DELAY_SECONDS = 40

def test_rate_limiting():
    """Test basic rate limiting with Groq"""
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
                        return False
                else:
                    print(f"❌ Non-rate-limit error: {e}")
                    return False
        
        print("\n🎉 Rate limiting test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rate_limiting()
    if success:
        print("\n✅ Ready to run full RAGAS evaluation!")
        print("Run: python eval.py")
    else:
        print("\n⚠️  Fix rate limiting issues before running eval.py") 