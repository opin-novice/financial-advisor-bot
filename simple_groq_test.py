#!/usr/bin/env python3
"""
Simple Groq API connectivity test
"""
import sys

def test_groq_basic():
    """Test basic Groq API connection without RAG pipeline"""
    print("🧪 Testing basic Groq API connection...")
    
    try:
        from langchain_groq import ChatGroq
        
        # Using the API key from your config
        api_key = "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"
        
        llm = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=api_key,
            temperature=0.3,
            max_tokens=50
        )
        
        # Simple test query
        response = llm.invoke("What is 2+2? Answer briefly.")
        print(f"✅ Groq API working! Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False

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
        
        return True
        
    except Exception as e:
        print(f"❌ RAG utilities test failed: {e}")
        return False

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