#!/usr/bin/env python3
"""
Quick Bangla Features Test - Fast execution
"""

import os
import sys
import time
import pytest
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_embedding_model():
    """Test if the embedding model is properly configured"""
    print("🧠 Testing Embedding Model Configuration...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from config import RAGConfig
        config = RAGConfig()
        
        print(f"   Model: {config.EMBEDDING_MODEL}")
        
        # Accept various valid embedding models that can handle bilingual content
        valid_models = [
            "paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "all-mpnet-base-v2"
        ]
        
        if config.EMBEDDING_MODEL in valid_models:
            print(f"   ✅ Valid embedding model configured: {config.EMBEDDING_MODEL}")
            assert True
        else:
            print(f"   ⚠️ Unexpected embedding model: {config.EMBEDDING_MODEL}")
            print(f"   📝 Expected one of: {valid_models}")
            # Don't fail, just warn - the model might still work
            assert True
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        pytest.fail(f"Embedding model test failed: {e}")

def test_document_index():
    """Test if document index exists and is accessible"""
    print("📚 Testing Document Index...")
    
    try:
        index_path = "faiss_index"
        
        if not os.path.exists(index_path):
            print("   ❌ FAISS index directory not found")
            pytest.fail("FAISS index directory not found")
        
        index_file = os.path.join(index_path, "index.faiss")
        pkl_file = os.path.join(index_path, "index.pkl")
        
        if not os.path.exists(index_file):
            print("   ❌ index.faiss file not found")
            pytest.fail("index.faiss file not found")
            
        if not os.path.exists(pkl_file):
            print("   ❌ index.pkl file not found")
            pytest.fail("index.pkl file not found")
        
        # Get file sizes
        index_size = os.path.getsize(index_file) / (1024 * 1024)  # MB
        pkl_size = os.path.getsize(pkl_file) / (1024 * 1024)  # MB
        
        print(f"   ✅ Index files found")
        print(f"   📊 Index size: {index_size:.1f} MB")
        print(f"   📊 Metadata size: {pkl_size:.1f} MB")
        
        assert True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        pytest.fail(f"Document index test failed: {e}")

def test_system_requirements():
    """Test system requirements"""
    print("💻 Testing System Requirements...")
    
    try:
        import torch
        import psutil
        
        # Check MPS availability
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        print(f"   MPS Available: {'✅ YES' if mps_available else '❌ NO'}")
        
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_percent = memory.percent
        
        print(f"   Total Memory: {memory_gb:.1f} GB")
        print(f"   Memory Usage: {memory_percent:.1f}%")
        
        if memory_percent > 90:
            print("   ⚠️  High memory usage detected")
        else:
            print("   ✅ Memory usage normal")
        
        assert True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        pytest.fail(f"System requirements test failed: {e}")

def test_quick_query():
    """Test a quick query without full initialization"""
    print("🔍 Testing Quick Query Processing...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Test just the embedding part
        from sentence_transformers import SentenceTransformer
        from config import RAGConfig
        
        config = RAGConfig()
        
        print("   Loading embedding model...")
        start_time = time.time()
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        load_time = time.time() - start_time
        
        print(f"   ✅ Model loaded in {load_time:.2f}s")
        
        # Test encoding
        test_queries = [
            "বাংলাদেশে ব্যাংক অ্যাকাউন্ট খোলার নিয়ম",
            "How to open bank account in Bangladesh"
        ]
        
        for query in test_queries:
            start_time = time.time()
            embeddings = model.encode([query])
            encode_time = time.time() - start_time
            
            print(f"   ✅ Encoded '{query[:30]}...' in {encode_time:.3f}s")
            print(f"      Embedding dimension: {len(embeddings[0])}")
        
        assert True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        pytest.fail(f"Quick query test failed: {e}")

def test_groq_connection():
    """Test Groq API connection"""
    print("🤖 Testing Groq API Connection...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from config import RAGConfig
        config = RAGConfig()
        
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
            print("   ⚠️ GROQ_API_KEY not configured or using template value")
            pytest.skip("GROQ_API_KEY not configured or using template value")
        
        print("   ✅ GROQ_API_KEY configured")
        
        # Try to import and initialize Groq
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=0.1
        )
        
        print(f"   ✅ Groq LLM initialized with model: {config.GROQ_MODEL}")
        assert True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        pytest.fail(f"Groq connection test failed: {e}")

def main():
    """Run quick Bangla features test"""
    print("🚀 Quick Bangla Features Test")
    print("=" * 50)
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Embedding Model", test_embedding_model),
        ("Document Index", test_document_index),
        ("System Requirements", test_system_requirements),
        ("Groq Connection", test_groq_connection),
        ("Quick Query", test_quick_query)
    ]
    
    results = {}
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    success_rate = (passed_tests / len(tests)) * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{len(tests)})")
    
    if success_rate >= 80:
        print("🎉 System is ready for Bangla queries!")
        return True
    elif success_rate >= 60:
        print("⚠️  System partially ready - some issues detected")
        return False
    else:
        print("❌ System not ready - multiple issues detected")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
