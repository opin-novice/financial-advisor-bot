#!/usr/bin/env python3
"""
Test script for language detection functionality
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_utils import LanguageDetector, BilingualResponseFormatter

def test_language_detection():
    """Test the language detection with various inputs"""
    
    print("🧪 Testing Language Detection System")
    print("=" * 50)
    
    detector = LanguageDetector()
    formatter = BilingualResponseFormatter(detector)
    
    # Test cases
    test_cases = [
        # English queries
        "What is the interest rate for savings account?",
        "How can I open a bank account in Bangladesh?",
        "Tell me about loan requirements",
        "What documents do I need for investment?",
        
        # Bangla queries
        "সঞ্চয় হিসাবের সুদের হার কত?",
        "বাংলাদেশে কিভাবে ব্যাংক একাউন্ট খুলব?",
        "লোনের জন্য কি কি কাগজপত্র লাগে?",
        "বিনিয়োগের নিয়ম কি?",
        
        # Mixed/ambiguous cases
        "bank account খুলতে কি লাগে?",
        "What is ঋণ eligibility?",
        "123456",  # Numbers only
        "",  # Empty string
        
        # Edge cases
        "আমি একটি ব্যাংক একাউন্ট খুলতে চাই। What documents do I need?",
        "How much টাকা do I need to deposit?",
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Query: '{query}'")
        
        if not query.strip():
            print("    → Empty query, skipping...")
            continue
            
        language, confidence = detector.detect_language(query)
        print(f"    → Language: {language}")
        print(f"    → Confidence: {confidence:.2%}")
        
        # Test system message translation
        test_message = "Processing your question..."
        translated = detector.translate_system_messages(test_message, language)
        print(f"    → System message: '{translated}'")
        
        # Test prompt selection
        prompt = detector.get_language_specific_prompt(language)
        print(f"    → Prompt language: {'Bangla' if 'বাংলায়' in prompt.template else 'English'}")

def test_response_formatting():
    """Test bilingual response formatting"""
    
    print("\n\n🎨 Testing Response Formatting")
    print("=" * 50)
    
    detector = LanguageDetector()
    formatter = BilingualResponseFormatter(detector)
    
    # Test formatting for both languages
    for language in ['english', 'bangla']:
        print(f"\n--- {language.title()} Formatting ---")
        
        sources_header = formatter.format_sources_section(language)
        print(f"Sources header: {sources_header}")
        
        doc_header = formatter.format_document_header(1, "test_document.pdf", language)
        print(f"Document header: {doc_header.strip()}")
        
        chunk_header = formatter.format_chunk_header(1, language)
        print(f"Chunk header: {chunk_header.strip()}")
        
        confidence_msg = detector.format_confidence_message(language)
        print(f"Confidence message: {confidence_msg[:100]}...")

def test_prompt_templates():
    """Test language-specific prompt templates"""
    
    print("\n\n📝 Testing Prompt Templates")
    print("=" * 50)
    
    detector = LanguageDetector()
    
    # Test English prompt
    english_prompt = detector.get_language_specific_prompt('english')
    print("English Prompt Preview:")
    print(english_prompt.template[:200] + "...")
    
    print("\n" + "-" * 30)
    
    # Test Bangla prompt
    bangla_prompt = detector.get_language_specific_prompt('bangla')
    print("Bangla Prompt Preview:")
    print(bangla_prompt.template[:200] + "...")

def main():
    """Run all tests"""
    try:
        test_language_detection()
        test_response_formatting()
        test_prompt_templates()
        
        print("\n\n✅ All tests completed successfully!")
        print("\n💡 Tips for using the language detection:")
        print("   • The system detects language based on script and common words")
        print("   • Confidence scores help identify mixed-language queries")
        print("   • System messages are automatically translated")
        print("   • Responses will match the detected input language")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
