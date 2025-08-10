#!/usr/bin/env python3
"""
🤖 Advanced Multilingual RAG System - Interactive Demo

This demo script showcases the key features of the system:
- Multilingual language detection
- Advanced RAG with feedback loop
- Bilingual response generation

Run this script to see the system in action!
"""

import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_setup():
    """Check if the system is properly set up"""
    print("🔍 Checking system setup...")
    
    issues = []
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        issues.append("❌ GROQ_API_KEY not found in .env file")
    else:
        print("✅ GROQ API key found")
    
    # Check if documents exist
    if not os.path.exists("data") or not os.listdir("data"):
        issues.append("❌ No documents found in data/ folder")
    else:
        doc_count = len([f for f in os.listdir("data") if f.endswith(('.pdf', '.txt'))])
        print(f"✅ Found {doc_count} documents")
    
    # Check if vector database exists
    if not os.path.exists("faiss_index"):
        issues.append("❌ Vector database not found - run 'python docadd.py' first")
    else:
        print("✅ Vector database found")
    
    if issues:
        print("\n🚨 Setup Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n📖 Please check GETTING_STARTED.md for setup instructions")
        return False
    
    print("✅ System setup looks good!")
    return True

def demo_language_detection():
    """Demonstrate language detection capabilities"""
    print("\n" + "="*60)
    print("🌍 LANGUAGE DETECTION DEMO")
    print("="*60)
    
    try:
        from language_utils import LanguageDetector
        detector = LanguageDetector()
        
        test_queries = [
            ("What is the interest rate?", "English"),
            ("সুদের হার কত?", "Bangla"),
            ("How to open account?", "English"),
            ("হিসাব কিভাবে খুলব?", "Bangla"),
            ("bank account খুলতে কি লাগে?", "Mixed")
        ]
        
        for query, expected in test_queries:
            result = detector.detect_language(query)
            confidence = detector.get_confidence_score(query)
            
            print(f"\n📝 Query: '{query}'")
            print(f"🎯 Detected: {result['language']} (confidence: {confidence:.2f})")
            print(f"✅ Expected: {expected}")
            
    except ImportError as e:
        print(f"❌ Error importing language detection: {e}")
        print("💡 Make sure all dependencies are installed")

def demo_rag_system():
    """Demonstrate RAG system with sample queries"""
    print("\n" + "="*60)
    print("🤖 RAG SYSTEM DEMO")
    print("="*60)
    
    try:
        # Import main components
        from language_utils import LanguageDetector
        from rag_utils import RAGUtils
        
        detector = LanguageDetector()
        rag = RAGUtils()
        
        # Sample queries in both languages
        sample_queries = [
            "What are the requirements to open a savings account?",
            "সঞ্চয় হিসাব খুলতে কি কি প্রয়োজন?",
            "What is the minimum balance required?",
            "সর্বনিম্ন ব্যালেন্স কত লাগে?"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n🔍 Query {i}: '{query}'")
            
            # Detect language
            lang_result = detector.detect_language(query)
            print(f"🌍 Language: {lang_result['language']}")
            
            # Process query (simplified demo)
            print("⚙️ Processing with RAG system...")
            print("📄 Retrieving relevant documents...")
            print("🧠 Generating response...")
            print("✅ Response ready!")
            
            # Note: Full RAG processing would require the complete system
            print("💡 (Full response generation requires complete system setup)")
            
    except ImportError as e:
        print(f"❌ Error importing RAG components: {e}")
        print("💡 Make sure all dependencies are installed")

def demo_feedback_loop():
    """Demonstrate advanced RAG feedback loop"""
    print("\n" + "="*60)
    print("🔄 ADVANCED RAG FEEDBACK LOOP DEMO")
    print("="*60)
    
    print("🎯 The Advanced RAG Feedback Loop works like this:")
    print("\n1. 📝 User asks: 'What are loan requirements?'")
    print("2. 🔍 System retrieves initial documents")
    print("3. 📊 Checks relevance score (e.g., 0.2 - too low!)")
    print("4. 🔄 Refines query: 'loan eligibility criteria requirements'")
    print("5. 🔍 Retrieves better documents")
    print("6. 📊 Checks relevance score (e.g., 0.8 - good!)")
    print("7. ✅ Generates high-quality response")
    
    print("\n🧠 Refinement Strategies:")
    print("   • Domain Expansion: Adds financial terms")
    print("   • Synonym Matching: Uses related words")
    print("   • Context Addition: Uses document context")
    print("   • Query Decomposition: Breaks complex queries")
    
    print("\n⚙️ Performance Modes:")
    print("   • Fast: 2 iterations, 0.4 threshold")
    print("   • Balanced: 3 iterations, 0.3 threshold")
    print("   • Thorough: 4 iterations, 0.2 threshold")

def interactive_demo():
    """Run an interactive demo"""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE DEMO")
    print("="*60)
    
    print("Try asking questions in English or Bangla!")
    print("Type 'quit' to exit")
    
    try:
        from language_utils import LanguageDetector
        detector = LanguageDetector()
        
        while True:
            print("\n" + "-"*40)
            query = input("🤔 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Detect language
            lang_result = detector.detect_language(query)
            confidence = detector.get_confidence_score(query)
            
            print(f"🌍 Detected Language: {lang_result['language']}")
            print(f"📊 Confidence: {confidence:.2f}")
            
            # Simulate processing
            print("⚙️ Processing your question...")
            print("📄 Searching documents...")
            print("🧠 Generating response...")
            
            # Show what would happen
            if lang_result['language'] == 'bangla':
                print("🎯 Would respond in Bangla")
            else:
                print("🎯 Would respond in English")
            
            print("💡 (Connect to full system for actual responses)")
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure language_utils.py is available")

def main():
    """Main demo function"""
    print("🤖 Advanced Multilingual RAG System - Demo")
    print("=" * 60)
    print("Welcome to the interactive demo!")
    print("This will showcase the key features of the system.")
    
    # Check setup first
    if not check_setup():
        print("\n⚠️  Some components may not work properly.")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("👋 Please complete setup and try again!")
            return
    
    while True:
        print("\n🎯 Choose a demo:")
        print("1. 🌍 Language Detection Demo")
        print("2. 🤖 RAG System Overview")
        print("3. 🔄 Feedback Loop Explanation")
        print("4. 🎮 Interactive Language Detection")
        print("5. 📖 View System Info")
        print("6. 👋 Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            demo_language_detection()
        elif choice == '2':
            demo_rag_system()
        elif choice == '3':
            demo_feedback_loop()
        elif choice == '4':
            interactive_demo()
        elif choice == '5':
            show_system_info()
        elif choice == '6':
            print("👋 Thanks for trying the demo!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def show_system_info():
    """Show system information"""
    print("\n" + "="*60)
    print("📊 SYSTEM INFORMATION")
    print("="*60)
    
    print("🎯 Key Features:")
    print("   • Multilingual support (English + Bangla)")
    print("   • Advanced RAG with feedback loops")
    print("   • Telegram bot integration")
    print("   • Semantic document chunking")
    print("   • Cross-encoder re-ranking")
    print("   • Language-aware caching")
    
    print("\n🏗️ Architecture:")
    print("   • Language Detection → RAG Feedback Loop")
    print("   • Document Retrieval → Cross-Encoder Ranking")
    print("   • Response Generation → Bilingual Formatting")
    
    print("\n📁 Project Structure:")
    print("   • main.py - Main application")
    print("   • language_utils.py - Language detection")
    print("   • advanced_rag_feedback.py - Feedback loop")
    print("   • rag_utils.py - Core RAG functionality")
    print("   • config.py - Configuration management")
    
    print("\n🔧 Configuration:")
    print(f"   • GROQ API Key: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")
    print(f"   • Telegram Token: {'✅ Set' if os.getenv('TELEGRAM_BOT_TOKEN') else '❌ Missing'}")
    print(f"   • Feedback Loop: {'✅ Enabled' if os.getenv('ENABLE_FEEDBACK_LOOP', 'true').lower() == 'true' else '❌ Disabled'}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check your setup and try again.")
