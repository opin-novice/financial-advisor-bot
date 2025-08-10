#!/usr/bin/env python3
"""
Demo script to showcase the language detection feature
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_utils import LanguageDetector, BilingualResponseFormatter

def demo_language_detection():
    """Demonstrate the language detection feature with interactive examples"""
    
    print("🎯 Language Detection Feature Demo")
    print("=" * 60)
    print("This demo shows how your Telegram bot will automatically")
    print("detect and respond in the user's language (English/Bangla)")
    print("=" * 60)
    
    detector = LanguageDetector()
    formatter = BilingualResponseFormatter(detector)
    
    # Demo scenarios
    scenarios = [
        {
            "title": "📝 English Financial Query",
            "query": "What documents do I need to open a savings account?",
            "expected": "english"
        },
        {
            "title": "📝 Bangla Financial Query", 
            "query": "সঞ্চয় হিসাব খুলতে কি কি কাগজপত্র লাগে?",
            "expected": "bangla"
        },
        {
            "title": "📝 Mixed Language Query (Bangla Dominant)",
            "query": "bank account খুলতে কি documents লাগে?",
            "expected": "bangla"
        },
        {
            "title": "📝 Mixed Language Query (English Dominant)",
            "query": "What is the ঋণ application process?",
            "expected": "english"
        },
        {
            "title": "📝 Short Bangla Query",
            "query": "লোন কিভাবে নিব?",
            "expected": "bangla"
        },
        {
            "title": "📝 Technical English Query",
            "query": "What is the current FDR interest rate?",
            "expected": "english"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['title']}")
        print("-" * 50)
        
        query = scenario['query']
        expected = scenario['expected']
        
        print(f"👤 User Query: '{query}'")
        
        # Detect language
        detected_language, confidence = detector.detect_language(query)
        
        print(f"🤖 Bot Detection: {detected_language} (confidence: {confidence:.1%})")
        
        # Show what the bot would do
        processing_msg = detector.translate_system_messages("Processing your question...", detected_language)
        print(f"🔄 Processing Message: '{processing_msg}'")
        
        # Show response formatting
        sources_header = formatter.format_sources_section(detected_language)
        print(f"📄 Sources Header: '{sources_header}'")
        
        doc_header = formatter.format_document_header(1, "loan_guidelines.pdf", detected_language)
        print(f"📂 Document Header: '{doc_header.strip()}'")
        
        # Accuracy check
        accuracy = "✅ Correct" if detected_language == expected else "❌ Incorrect"
        print(f"🎯 Detection Accuracy: {accuracy}")
        
        if detected_language != expected:
            print(f"   Expected: {expected}, Got: {detected_language}")

def demo_system_messages():
    """Demo system message translations"""
    
    print("\n\n🔄 System Message Translation Demo")
    print("=" * 60)
    
    detector = LanguageDetector()
    
    messages = [
        "Processing your question...",
        "Please enter a valid question.",
        "Hi! Ask me any financial question.",
        "I could not find relevant information in my database for your query."
    ]
    
    for msg in messages:
        print(f"\n📝 Original: '{msg}'")
        
        english_msg = detector.translate_system_messages(msg, 'english')
        bangla_msg = detector.translate_system_messages(msg, 'bangla')
        
        print(f"🇺🇸 English: '{english_msg}'")
        print(f"🇧🇩 Bangla: '{bangla_msg}'")

def demo_confidence_scenarios():
    """Demo different confidence scenarios"""
    
    print("\n\n📊 Confidence Level Scenarios")
    print("=" * 60)
    
    detector = LanguageDetector()
    
    test_cases = [
        ("Pure English", "What is the interest rate for savings accounts in Bangladesh?"),
        ("Pure Bangla", "বাংলাদেশে সঞ্চয় হিসাবের সুদের হার কত?"),
        ("Mixed (Bangla dominant)", "আমি একটি bank account খুলতে চাই"),
        ("Mixed (English dominant)", "I need to apply for a ঋণ"),
        ("Short query", "লোন"),
        ("Numbers only", "12345"),
        ("Technical terms", "FDR, DPS, SND"),
    ]
    
    for scenario, query in test_cases:
        language, confidence = detector.detect_language(query)
        
        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        
        print(f"\n📝 {scenario}")
        print(f"   Query: '{query}'")
        print(f"   Result: {language} ({confidence:.1%} - {confidence_level} confidence)")

def interactive_demo():
    """Interactive demo where user can test their own queries"""
    
    print("\n\n🎮 Interactive Demo")
    print("=" * 60)
    print("Type your own queries to test language detection!")
    print("(Type 'quit' to exit)")
    
    detector = LanguageDetector()
    formatter = BilingualResponseFormatter(detector)
    
    while True:
        try:
            query = input("\n👤 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Demo ended. Thanks for testing!")
                break
            
            if not query:
                print("❌ Please enter a valid query")
                continue
            
            # Detect and show results
            language, confidence = detector.detect_language(query)
            
            print(f"🤖 Detected Language: {language}")
            print(f"📊 Confidence: {confidence:.1%}")
            
            # Show what bot would respond with
            processing_msg = detector.translate_system_messages("Processing your question...", language)
            print(f"🔄 Bot would show: '{processing_msg}'")
            
            # Show response formatting
            sources_header = formatter.format_sources_section(language)
            print(f"📄 Sources would be titled: '{sources_header}'")
            
        except KeyboardInterrupt:
            print("\n👋 Demo ended. Thanks for testing!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run the complete demo"""
    
    try:
        demo_language_detection()
        demo_system_messages()
        demo_confidence_scenarios()
        
        print("\n\n🎉 Demo completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   ✅ Language detection works for both English and Bangla")
        print("   ✅ System messages are automatically translated")
        print("   ✅ Response formatting adapts to detected language")
        print("   ✅ Mixed language queries are handled intelligently")
        print("   ✅ Confidence scores help identify detection quality")
        
        # Ask if user wants interactive demo
        print("\n" + "=" * 60)
        response = input("🎮 Would you like to try the interactive demo? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            interactive_demo()
        
        print("\n🚀 Your bot is ready to use with language detection!")
        print("   Run: python main.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
