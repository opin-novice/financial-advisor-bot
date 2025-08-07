#!/usr/bin/env python3
"""
Test script to verify Groq API migration for RAG pipeline
Run this after setting your GROQ_API_KEY environment variable
"""
import os

def test_groq_connection():
    """Test basic Groq API connection"""
    print("🧪 Testing Groq API connection...")
    
    # Using API key directly
    api_key = "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"
    
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
        return True
        
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False

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
        
        return True
        
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        return False

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