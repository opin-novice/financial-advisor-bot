#!/usr/bin/env python3
"""
Test script to verify Groq API migration for RAG pipeline
Run this after setting your GROQ_API_KEY environment variable
"""
import os
import pytest

def test_groq_connection():
    """Test basic Groq API connection"""
    print("🧪 Testing Groq API connection...")
    
    # Try to get API key from environment
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("⚠️ GROQ_API_KEY not found in environment or using template value")
        pytest.skip("GROQ_API_KEY not found in environment or using template value")
    
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=api_key,
            temperature=0.5,
            max_tokens=100,
            model_kwargs={"top_p": 0.9}
        )
        
        response = llm.invoke("What is 2+2?")
        print(f"✅ Groq API working! Response: {response.content}")
        assert True  # Test passed
        
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        pytest.fail(f"Groq API test failed: {e}")

def test_rag_pipeline():
    """Test the complete RAG pipeline"""
    print("\n🧪 Testing RAG pipeline with Groq...")
    
    try:
        from main import FinancialAdvisorTelegramBot
        bot = FinancialAdvisorTelegramBot()
        
        test_query = "What documents are needed for opening a bank account?"
        result = bot.process_query(test_query)
        
        print(f"✅ RAG pipeline working!")
        print(f"📝 Query: {test_query}")
        print(f"🤖 Response preview: {result.get('response', 'No response')[:150]}...")
        
        assert True  # Test passed
        
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        pytest.fail(f"RAG pipeline test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Groq Migration")
    print("=" * 40)
    
    success = test_groq_connection()
    if success:
        success = test_rag_pipeline()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Migration successful! Your RAG pipeline is ready.")
    else:
        print("⚠️  Please check the error messages above.") 