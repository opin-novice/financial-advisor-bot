#!/usr/bin/env python3
"""
Test script to verify all demo features work correctly
"""

import sys
import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

def test_language_detection():
    """Test language detection functionality"""
    print("🧪 Testing Language Detection...")
    try:
        from language_utils import LanguageDetector
        detector = LanguageDetector()
        
        # Test English
        lang, conf = detector.detect_language("What is the interest rate?")
        assert lang == "english", f"Expected 'english', got '{lang}'"
        assert conf > 0.5, f"Expected confidence > 0.5, got {conf}"
        
        # Test Bangla
        lang, conf = detector.detect_language("সুদের হার কত?")
        assert lang == "bengali", f"Expected 'bengali', got '{lang}'"
        assert conf > 0.5, f"Expected confidence > 0.5, got {conf}"
        
        print("✅ Language detection tests passed!")
        return True
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        return False

def test_rag_utils():
    """Test RAG utilities"""
    print("🧪 Testing RAG Utils...")
    try:
        from rag_utils import RAGUtils
        rag = RAGUtils()
        
        # Just test initialization
        assert rag.llm is not None, "LLM not initialized"
        
        print("✅ RAG utils tests passed!")
        return True
    except Exception as e:
        print(f"❌ RAG utils test failed: {e}")
        return False

def test_demo_imports():
    """Test that demo can import all required modules"""
    print("🧪 Testing Demo Imports...")
    try:
        # Test imports from demo.py
        from language_utils import LanguageDetector
        from rag_utils import RAGUtils
        
        print("✅ Demo import tests passed!")
        return True
    except Exception as e:
        print(f"❌ Demo import test failed: {e}")
        return False

def test_environment_setup():
    """Test environment setup"""
    print("🧪 Testing Environment Setup...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check API key
        groq_key = os.getenv("GROQ_API_KEY")
        assert groq_key is not None, "GROQ_API_KEY not found"
        assert len(groq_key) > 10, "GROQ_API_KEY seems too short"
        
        # Check data directory
        assert os.path.exists("data"), "Data directory not found"
        
        # Check vector database
        assert os.path.exists("faiss_index"), "FAISS index not found"
        
        print("✅ Environment setup tests passed!")
        return True
    except Exception as e:
        print(f"❌ Environment setup test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Demo Tests...")
    print("=" * 50)
    
    tests = [
        test_environment_setup,
        test_demo_imports,
        test_language_detection,
        test_rag_utils,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Demo should work correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
