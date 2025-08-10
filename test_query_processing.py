#!/usr/bin/env python3
"""
Focused test for query processing functionality
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_query_processing():
    """Test the main query processing functionality"""
    print("🧪 Testing query processing functionality...")
    
    try:
        # Import the main bot class
        from main import FinancialAdvisorTelegramBot
        
        print("✅ Successfully imported FinancialAdvisorTelegramBot")
        
        # Initialize the bot
        print("🔄 Initializing bot...")
        bot = FinancialAdvisorTelegramBot()
        print("✅ Bot initialized successfully")
        
        # Test a simple query
        test_query = "How to open a bank account in Bangladesh?"
        print(f"🔍 Testing query: '{test_query}'")
        
        # Process the query
        result = bot.process_query(test_query)
        
        if isinstance(result, dict):
            print("✅ Query processed successfully!")
            print(f"📝 Response type: {type(result.get('response', 'N/A'))}")
            print(f"📄 Sources found: {len(result.get('sources', []))}")
            
            # Show response preview
            if result.get('response'):
                response_preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
                print(f"📋 Response preview: {response_preview}")
            
            # Show sources preview
            if result.get('sources'):
                print("📚 Source previews:")
                for i, source in enumerate(result['sources'][:3], 1):
                    source_preview = source[:100] + "..." if len(source) > 100 else source
                    print(f"   {i}. {source_preview}")
            
            return True
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Query processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_retrieval():
    """Test embedding-based retrieval specifically"""
    print("\n🧪 Testing embedding-based retrieval...")
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import config
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL, 
            model_kwargs={"device": "cpu"}
        )
        
        # Load FAISS index
        vectorstore = FAISS.load_local(
            config.FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Test different types of queries
        test_queries = [
            "bank account opening process",
            "loan eligibility requirements", 
            "tax calculation methods",
            "investment opportunities"
        ]
        
        for query in test_queries:
            print(f"🔍 Query: '{query}'")
            
            # Get embeddings and retrieve documents
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
            
            print(f"   📄 Retrieved {len(docs_with_scores)} documents")
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                print(f"      {i}. Score: {score:.4f} | {content_preview}")
        
        print("✅ Embedding-based retrieval test passed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding-based retrieval test failed: {e}")
        return False

def main():
    """Run focused tests"""
    print("🚀 Focused Query Processing Test")
    print("=" * 60)
    
    # Test 1: Embedding-based retrieval
    retrieval_success = test_embedding_retrieval()
    
    # Test 2: Full query processing
    processing_success = test_query_processing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print("=" * 60)
    
    print(f"🔍 Embedding Retrieval: {'✅ PASS' if retrieval_success else '❌ FAIL'}")
    print(f"🤖 Query Processing: {'✅ PASS' if processing_success else '❌ FAIL'}")
    
    if retrieval_success and processing_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your query embedding and processing functionality is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
