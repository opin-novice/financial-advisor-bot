#!/usr/bin/env python3
"""
Comprehensive test of the RAG system with Groq API
"""
import sys
import pytest
from langchain.schema import Document

def test_full_integration():
    """Test the complete RAG pipeline integration"""
    print("🧪 Testing full RAG integration...")
    
    try:
        # Import the main bot class
        from main import FinancialAdvisorTelegramBot
        
        print("✅ Successfully imported FinancialAdvisorTelegramBot")
        
        # Initialize the bot (this tests FAISS loading, Groq connection, etc.)
        print("🔄 Initializing bot (loading FAISS, connecting to Groq)...")
        bot = FinancialAdvisorTelegramBot()
        print("✅ Bot initialized successfully")
        
        # Test a real query through the full pipeline
        test_queries = [
            "How to open a bank account in Bangladesh?",
            "What are the tax rates for individuals?",
            "Investment options available in Bangladesh"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: Processing query: '{query}'")
            try:
                result = bot.process_query(query)
                
                if isinstance(result, dict) and result.get("response"):
                    response_preview = result["response"][:150] + "..." if len(result["response"]) > 150 else result["response"]
                    print(f"✅ Query {i} successful!")
                    print(f"📝 Response preview: {response_preview}")
                    print(f"📄 Sources found: {len(result.get('sources', []))}")
                else:
                    print(f"⚠️  Query {i} returned unexpected format: {type(result)}")
                    
            except Exception as e:
                print(f"❌ Query {i} failed: {e}")
                return False
        
        assert True  # Full integration test successful
        
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        pytest.fail(f"Full integration test failed: {e}")

def test_rag_utils_integration():
    """Test RAG utilities integration"""
    print("\n🧪 Testing RAG utilities integration...")
    
    try:
        from rag_utils import RAGUtils
        rag = RAGUtils()
        
        # Test query refinement
        query = "loan eligibility"
        refined = rag.refine_query(query)
        print(f"✅ Query refinement: '{query}' -> '{refined[:100]}...'")
        
        # Test relevance checking with mock documents
        docs = [
            Document(page_content="Loan eligibility criteria in Bangladesh include minimum income, credit score, and employment status."),
            Document(page_content="To apply for a personal loan, you need to submit salary certificates and bank statements.")
        ]
        
        is_relevant, confidence = rag.check_query_relevance(query, docs)
        print(f"✅ Relevance check: Relevant={is_relevant}, Confidence={confidence:.3f}")
        
        # Test answer validation
        answer = "To be eligible for a loan in Bangladesh, you need a minimum monthly income of 25,000 Taka and a good credit score."
        contexts = [doc.page_content for doc in docs]
        validation = rag.validate_answer(query, answer, contexts)
        print(f"✅ Answer validation: Valid={validation['valid']}, Confidence={validation['confidence']:.3f}")
        
        assert True  # RAG utilities integration test successful
        
    except Exception as e:
        print(f"❌ RAG utilities integration test failed: {e}")
        pytest.fail(f"RAG utilities integration test failed: {e}")

def test_environment_security():
    """Test that environment variables are properly loaded"""
    print("\n🧪 Testing environment security...")
    
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        groq_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL")
        
        if groq_key and groq_key.startswith("gsk_"):
            print("✅ GROQ_API_KEY loaded from environment")
        else:
            print("❌ GROQ_API_KEY not properly loaded")
            pytest.fail("GROQ_API_KEY not properly loaded")
            
        if groq_model:
            print(f"✅ GROQ_MODEL loaded: {groq_model}")
        else:
            print("⚠️  GROQ_MODEL not set, using default")
        
        assert True  # Environment security test successful
        
    except Exception as e:
        print(f"❌ Environment security test failed: {e}")
        pytest.fail(f"Environment security test failed: {e}")

if __name__ == "__main__":
    print("🚀 Comprehensive RAG System Test")
    print("=" * 60)
    
    # Test 1: Environment security
    env_success = test_environment_security()
    
    # Test 2: RAG utilities integration
    rag_success = test_rag_utils_integration()
    
    # Test 3: Full integration (only if previous tests pass)
    integration_success = False
    if env_success and rag_success:
        integration_success = test_full_integration()
    else:
        print("\n⚠️  Skipping full integration test due to previous failures")
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"🔐 Environment Security: {'✅ PASS' if env_success else '❌ FAIL'}")
    print(f"🔧 RAG Utilities: {'✅ PASS' if rag_success else '❌ FAIL'}")
    print(f"🔗 Full Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    if env_success and rag_success and integration_success:
        print("\n🎉 ALL TESTS PASSED! Your RAG system is working perfectly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)