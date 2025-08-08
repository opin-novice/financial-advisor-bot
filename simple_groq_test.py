#!/usr/bin/env python3
"""
Simple Groq API connectivity test
"""
import sys
import pytest

def test_groq_basic():
    """Test basic Groq API connection without RAG pipeline"""
    print("🧪 Testing basic Groq API connection...")
    
    try:
        from langchain_groq import ChatGroq
        
        # Try to get API key from environment or config
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            print("⚠️ GROQ_API_KEY not found in environment or using template value")
            pytest.skip("GROQ_API_KEY not found in environment or using template value")
        
        llm = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=api_key,
            temperature=0.3,
            max_tokens=50
        )
        
        # Simple test query
        response = llm.invoke("What is 2+2? Answer briefly.")
        print(f"✅ Groq API working! Response: {response.content}")
        assert response.content is not None
        
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        pytest.fail(f"Groq API test failed: {e}")

def test_rag_utils():
    """Test RAG utilities separately"""
    print("\n🧪 Testing RAG utilities...")
    
    try:
        from rag_utils import RAGUtils
        rag = RAGUtils()
        print("✅ RAG utilities initialized successfully")
        
        # Test query refinement
        original_query = "bank account"
        refined = rag.refine_query(original_query)
        print(f"📝 Query refinement: '{original_query}' -> '{refined}'")
        
        assert refined is not None
        
    except Exception as e:
        print(f"❌ RAG utilities test failed: {e}")
        pytest.fail(f"RAG utilities test failed: {e}")

if __name__ == "__main__":
    print("🚀 Simple Groq API Test")
    print("=" * 40)
    
    # Test 1: Basic Groq API
    groq_success = test_groq_basic()
    
    # Test 2: RAG utilities (without full pipeline)
    rag_success = test_rag_utils()
    
    print("\n" + "=" * 40)
    if groq_success and rag_success:
        print("🎉 Basic tests passed! Groq API and RAG utilities are working.")
    else:
        print("⚠️  Some tests failed. Check error messages above.")
        sys.exit(1)