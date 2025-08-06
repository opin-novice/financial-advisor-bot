#!/usr/bin/env python3
"""
Test script for Spanish functionality in the multilingual financial advisor bot
"""

import os
import sys
from pathlib import Path

def test_spanish_translator():
    """Test the Spanish translator module"""
    print("🧪 Testing Spanish Translator Module...")
    print("=" * 50)
    
    try:
        from spanish_translator import SpanishTranslator
        
        translator = SpanishTranslator()
        
        # Test language detection
        print("\n📝 Language Detection Test:")
        test_cases = [
            ("¿Cómo abrir una cuenta bancaria?", "spanish"),
            ("How to open a bank account?", "english"),
            ("ব্যাংক অ্যাকাউন্ট কীভাবে খুলবো?", "bangla"),
            ("¿Cuáles son los requisitos para un préstamo?", "spanish"),
            ("What documents do I need for a loan?", "english"),
            ("¿Qué es el número TIN?", "spanish"),
        ]
        
        correct_detections = 0
        for text, expected in test_cases:
            detected = translator.detect_language(text)
            status = "✅" if detected == expected else "❌"
            if detected == expected:
                correct_detections += 1
            print(f"{status} '{text}' -> {detected} (expected: {expected})")
        
        accuracy = (correct_detections / len(test_cases)) * 100
        print(f"\n📊 Language Detection Accuracy: {accuracy:.1f}%")
        
        return accuracy > 80  # Consider 80%+ as passing
        
    except ImportError as e:
        print(f"❌ Could not import Spanish translator: {e}")
        return False
    except Exception as e:
        print(f"❌ Spanish translator test failed: {e}")
        return False

def test_spanish_translation():
    """Test Spanish translation functionality"""
    print("\n🔄 Testing Spanish Translation...")
    print("=" * 50)
    
    try:
        from spanish_translator import SpanishTranslator
        
        translator = SpanishTranslator()
        
        # Test Spanish to English translation
        spanish_queries = [
            "¿Cómo abrir una cuenta bancaria?",
            "¿Qué documentos necesito para un préstamo?",
            "¿Cuál es la tasa de interés?",
        ]
        
        print("\n📤 Spanish to English Translation:")
        for spanish_query in spanish_queries:
            print(f"Spanish: {spanish_query}")
            english_query, _ = translator.process_spanish_query(spanish_query)
            print(f"English: {english_query}")
            print()
        
        # Test English to Spanish translation
        english_responses = [
            "To open a bank account, you need national ID, photographs, and initial deposit.",
            "For a loan, you typically need income proof, collateral, and credit history.",
            "Interest rates vary by bank and loan type, typically ranging from 9-15% annually.",
        ]
        
        print("📥 English to Spanish Translation:")
        for english_response in english_responses:
            print(f"English: {english_response}")
            spanish_response = translator.translate_english_to_spanish(english_response)
            print(f"Spanish: {spanish_response}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        return False

def test_multilingual_bot_spanish():
    """Test the multilingual bot with Spanish queries"""
    print("\n🤖 Testing Multilingual Bot with Spanish...")
    print("=" * 50)
    
    try:
        # Check if multilingual index exists
        if not Path("faiss_index_multilingual").exists():
            print("❌ Multilingual index not found. Please run multilingual_semantic_chunking.py first")
            return False
        
        from multilingual_main import MultilingualFinancialAdvisorBot
        
        bot = MultilingualFinancialAdvisorBot()
        
        # Test Spanish queries
        spanish_test_queries = [
            "¿Qué es el número TIN?",
            "¿Cómo abrir una cuenta bancaria?",
            "¿Cuáles son los requisitos para un préstamo?",
        ]
        
        successful_queries = 0
        
        for query in spanish_test_queries:
            try:
                print(f"\n🔍 Testing Spanish query: {query}")
                response = bot.process_query(query)
                
                if isinstance(response, dict) and response.get('response'):
                    detected_lang = response.get('language', 'unknown')
                    answer = response['response']
                    translated_query = response.get('translated_query')
                    
                    print(f"   ✅ Response received")
                    print(f"   🌐 Detected language: {detected_lang}")
                    print(f"   📝 Answer: {answer[:100]}...")
                    if translated_query:
                        print(f"   🔄 Translated query: {translated_query}")
                    
                    if detected_lang == 'spanish':
                        successful_queries += 1
                        print(f"   ✅ Language handling correct")
                    else:
                        print(f"   ⚠️ Language mismatch: expected spanish, got {detected_lang}")
                else:
                    print(f"   ❌ No valid response received")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        success_rate = (successful_queries / len(spanish_test_queries)) * 100
        print(f"\n📊 Spanish Bot Query Success Rate: {success_rate:.1f}% ({successful_queries}/{len(spanish_test_queries)})")
        
        return success_rate > 60  # Consider 60%+ as passing for Spanish
        
    except ImportError as e:
        print(f"❌ Could not import multilingual bot: {e}")
        return False
    except Exception as e:
        print(f"❌ Bot test failed: {e}")
        return False

def test_mixed_language_queries():
    """Test mixed language functionality"""
    print("\n🌐 Testing Mixed Language Functionality...")
    print("=" * 50)
    
    try:
        from multilingual_main import MultilingualFinancialAdvisorBot
        
        bot = MultilingualFinancialAdvisorBot()
        
        # Test queries in different languages
        mixed_queries = [
            ("What is TIN number?", "english"),
            ("ব্যাংক অ্যাকাউন্ট খোলার জন্য কী প্রয়োজন?", "bangla"),
            ("¿Qué documentos necesito para abrir una cuenta?", "spanish"),
        ]
        
        successful_queries = 0
        
        for query, expected_lang in mixed_queries:
            try:
                print(f"\n🔍 Testing {expected_lang} query: {query}")
                response = bot.process_query(query)
                
                if isinstance(response, dict) and response.get('response'):
                    detected_lang = response.get('language', 'unknown')
                    answer = response['response']
                    
                    print(f"   ✅ Response received")
                    print(f"   🌐 Language: {detected_lang}")
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
        
        success_rate = (successful_queries / len(mixed_queries)) * 100
        print(f"\n📊 Mixed Language Success Rate: {success_rate:.1f}% ({successful_queries}/{len(mixed_queries)})")
        
        return success_rate > 60
        
    except Exception as e:
        print(f"❌ Mixed language test failed: {e}")
        return False

def main():
    """Run all Spanish functionality tests"""
    print("🚀 Spanish Language Support Test Suite")
    print("=" * 60)
    
    tests = [
        ("Spanish Translator Module", test_spanish_translator),
        ("Spanish Translation", test_spanish_translation),
        ("Multilingual Bot Spanish", test_multilingual_bot_spanish),
        ("Mixed Language Queries", test_mixed_language_queries),
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
    print("📊 SPANISH FUNCTIONALITY TEST SUMMARY")
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
    
    if overall_success >= 75:
        print("🎉 Spanish functionality is working well!")
    elif overall_success >= 50:
        print("⚠️ Spanish functionality has some issues but may work")
    else:
        print("❌ Spanish functionality needs attention")
    
    # Usage examples
    print("\n📋 USAGE EXAMPLES:")
    print("English: 'What documents do I need to open a bank account?'")
    print("Bangla: 'ব্যাংক অ্যাকাউন্ট খোলার জন্য কী কী কাগজপত্র প্রয়োজন?'")
    print("Spanish: '¿Qué documentos necesito para abrir una cuenta bancaria?'")
    
    print(f"\n🔗 To start the bot: python multilingual_main.py")

if __name__ == "__main__":
    main()
