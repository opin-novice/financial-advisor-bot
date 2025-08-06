#!/usr/bin/env python3
"""
Quick test script for multilingual financial advisor bot
Tests language detection, document processing, and query handling
"""

import os
import sys
from typing import List, Dict

def test_language_detection():
    """Test language detection functionality"""
    print("🔍 Testing Language Detection...")
    print("-" * 40)
    
    try:
        # Import after adding to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from multilingual_main import LanguageProcessor
        
        lang_processor = LanguageProcessor()
        
        test_cases = [
            ("What is the interest rate for home loans?", "english"),
            ("ব্যাংক অ্যাকাউন্ট খোলার জন্য কী কী কাগজপত্র প্রয়োজন?", "bangla"),
            ("How much is the VAT rate in Bangladesh?", "english"),
            ("আয়কর রিটার্ন কীভাবে জমা দিতে হয়?", "bangla"),
            ("What documents needed for TIN registration?", "english"),
            ("সঞ্চয়পত্রে বিনিয়োগের সুবিধা কী?", "bangla"),
            ("Bank account kholar jonno ki ki lagbe?", "english"),  # Romanized Bangla
        ]
        
        correct_predictions = 0
        total_tests = len(test_cases)
        
        for text, expected_lang in test_cases:
            detected_lang = lang_processor.detect_language(text)
            is_correct = detected_lang == expected_lang
            correct_predictions += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Text: {text[:50]}...")
            print(f"   Expected: {expected_lang}, Detected: {detected_lang}")
            print()
        
        accuracy = (correct_predictions / total_tests) * 100
        print(f"📊 Language Detection Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_tests})")
        
        return accuracy > 70  # Consider 70%+ as passing
        
    except ImportError as e:
        print(f"❌ Could not import language processor: {e}")
        return False
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        return False

def test_document_processing():
    """Test document processing capabilities"""
    print("\n📄 Testing Document Processing...")
    print("-" * 40)
    
    try:
        from multilingual_semantic_chunking import MultilingualSemanticChunker
        
        chunker = MultilingualSemanticChunker()
        
        # Test English text
        english_text = """
        Banking services in Bangladesh include account opening, loans, and investment products. 
        To open a bank account, you need national ID, photographs, and initial deposit. 
        Interest rates vary by bank and product type.
        """
        
        # Test Bangla text
        bangla_text = """
        বাংলাদেশে ব্যাংকিং সেবার মধ্যে রয়েছে অ্যাকাউন্ট খোলা, ঋণ এবং বিনিয়োগ পণ্য।
        ব্যাংক অ্যাকাউন্ট খুলতে জাতীয় পরিচয়পত্র, ছবি এবং প্রাথমিক জমা প্রয়োজন।
        সুদের হার ব্যাংক এবং পণ্যের ধরন অনুযায়ী ভিন্ন হয়।
        """
        
        # Test chunking
        english_chunks = chunker.chunk_text(english_text, {"source": "test_english.pdf"})
        bangla_chunks = chunker.chunk_text(bangla_text, {"source": "test_bangla.pdf"})
        
        print(f"✅ English text chunked into {len(english_chunks)} pieces")
        print(f"✅ Bangla text chunked into {len(bangla_chunks)} pieces")
        
        # Check metadata
        if english_chunks:
            eng_meta = english_chunks[0].metadata
            print(f"   English chunk language: {eng_meta.get('language', 'unknown')}")
        
        if bangla_chunks:
            ban_meta = bangla_chunks[0].metadata
            print(f"   Bangla chunk language: {ban_meta.get('language', 'unknown')}")
        
        return len(english_chunks) > 0 and len(bangla_chunks) > 0
        
    except ImportError as e:
        print(f"❌ Could not import document processor: {e}")
        return False
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False

def test_vector_index():
    """Test if multilingual vector index exists and is accessible"""
    print("\n🗂️ Testing Vector Index...")
    print("-" * 40)
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        index_path = "faiss_index_multilingual"
        
        if not os.path.exists(index_path):
            print(f"❌ Multilingual index not found at {index_path}")
            print("   Please run: python multilingual_semantic_chunking.py")
            return False
        
        # Try to load the index
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )
        
        vectorstore = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Test search in both languages
        english_results = vectorstore.similarity_search("bank account opening", k=3)
        bangla_results = vectorstore.similarity_search("ব্যাংক অ্যাকাউন্ট", k=3)
        
        print(f"✅ Index loaded successfully")
        print(f"✅ English search returned {len(english_results)} results")
        print(f"✅ Bangla search returned {len(bangla_results)} results")
        
        # Show sample results
        if english_results:
            print(f"   Sample English result: {english_results[0].page_content[:100]}...")
        
        if bangla_results:
            print(f"   Sample Bangla result: {bangla_results[0].page_content[:100]}...")
        
        return len(english_results) > 0 or len(bangla_results) > 0
        
    except Exception as e:
        print(f"❌ Vector index test failed: {e}")
        return False

def test_bot_queries():
    """Test actual bot queries"""
    print("\n🤖 Testing Bot Queries...")
    print("-" * 40)
    
    try:
        # Check if multilingual index exists first
        if not os.path.exists("faiss_index_multilingual"):
            print("❌ Multilingual index not found. Skipping bot query tests.")
            print("   Please run: python multilingual_semantic_chunking.py")
            return False
        
        from multilingual_main import MultilingualFinancialAdvisorBot
        
        bot = MultilingualFinancialAdvisorBot()
        
        test_queries = [
            ("What is TIN number?", "english"),
            ("ব্যাংক অ্যাকাউন্ট খোলার জন্য কী প্রয়োজন?", "bangla"),
            ("How to apply for home loan?", "english"),
            ("আয়কর রিটার্ন কী?", "bangla"),
        ]
        
        successful_queries = 0
        
        for query, expected_lang in test_queries:
            try:
                print(f"\n🔍 Testing: {query}")
                response = bot.process_query(query)
                
                if isinstance(response, dict) and response.get('response'):
                    detected_lang = response.get('language', 'unknown')
                    answer = response['response']
                    
                    print(f"   ✅ Response received (Language: {detected_lang})")
                    print(f"   📝 Answer: {answer[:100]}...")
                    
                    if detected_lang == expected_lang:
                        successful_queries += 1
                        print(f"   ✅ Language detection correct")
                    else:
                        print(f"   ⚠️ Language mismatch: expected {expected_lang}, got {detected_lang}")
                else:
                    print(f"   ❌ No valid response received")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        success_rate = (successful_queries / len(test_queries)) * 100
        print(f"\n📊 Bot Query Success Rate: {success_rate:.1f}% ({successful_queries}/{len(test_queries)})")
        
        return success_rate > 50  # Consider 50%+ as passing
        
    except ImportError as e:
        print(f"❌ Could not import multilingual bot: {e}")
        print("   Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Bot query test failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking Dependencies...")
    print("-" * 40)
    
    required_packages = [
        ("torch", "PyTorch"),
        ("sentence_transformers", "Sentence Transformers"),
        ("langchain", "LangChain"),
        ("langdetect", "Language Detection"),
        ("faiss", "FAISS"),
        ("transformers", "Transformers"),
        ("nltk", "NLTK"),
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_multilingual.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def main():
    """Run all tests"""
    print("🚀 Multilingual Financial Advisor Bot - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Language Detection", test_language_detection),
        ("Document Processing", test_document_processing),
        ("Vector Index", test_vector_index),
        ("Bot Queries", test_bot_queries),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if passed:
            passed_tests += 1
    
    overall_success = (passed_tests / total_tests) * 100
    print(f"\nOverall Success Rate: {overall_success:.1f}% ({passed_tests}/{total_tests})")
    
    if overall_success >= 80:
        print("🎉 Multilingual bot is ready to use!")
    elif overall_success >= 60:
        print("⚠️ Multilingual bot has some issues but may work")
    else:
        print("❌ Multilingual bot needs attention before use")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    if not results.get("Dependencies", False):
        print("- Install missing dependencies: pip install -r requirements_multilingual.txt")
    
    if not results.get("Vector Index", False):
        print("- Create multilingual index: python multilingual_semantic_chunking.py")
    
    if not results.get("Bot Queries", False):
        print("- Check Ollama is running: ollama serve")
        print("- Verify model is installed: ollama pull gemma3n:e2b")
    
    print("\n🔗 For detailed setup instructions, see: README_MULTILINGUAL.md")

if __name__ == "__main__":
    main()
