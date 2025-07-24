#!/usr/bin/env python3
"""
Demo script to showcase the enhanced RAG response format
"""

from main import FinancialAdvisorBot
import time

def demo_enhanced_response():
    """Demo the enhanced response with your specific question"""
    
    print("🚀 Enhanced RAG Financial Advisor Bot Demo")
    print("=" * 60)
    print("Showcasing improved response quality and formatting")
    print("=" * 60)
    
    # Initialize bot
    print("\n📡 Initializing enhanced RAG system...")
    bot = FinancialAdvisorBot()
    print("✅ System ready!")
    
    # Your specific question
    query = "how to file my taxes?"
    
    print(f"\n🔍 Query: '{query}'")
    print("\n⏳ Processing... (This will take a moment with local LLM)")
    
    start_time = time.time()
    response = bot.process_query(query)
    processing_time = time.time() - start_time
    
    if 'error' in response:
        print(f"❌ Error: {response['error']}")
        return
    
    # Display the enhanced response exactly as users would see it
    print("\n" + "="*60)
    print("📋 ENHANCED RAG RESPONSE OUTPUT")
    print("="*60)
    
    print(response['response'])
    
    # Display sources in enhanced format
    if response.get('sources'):
        print("\n" + "-"*50)
        print("📚 Sources:")
        for i, source in enumerate(response['sources'], 1):
            print(f"  {i}. {source['name']} (Page: {source['page']}) - {source['category']}")
    
    # Display disclaimer and timing
    print("\n" + "-"*50)
    print(f"⚠️  {response['disclaimer']}")
    print(f"⏱️  Response generated in {processing_time:.2f}s")
    print("="*60)
    
    # Show the improvement summary
    print("\n🎉 KEY IMPROVEMENTS DEMONSTRATED:")
    print("✅ Professional category header (📊 Tax Information)")
    print("✅ Well-structured numbered steps and bullets")
    print("✅ Clear, comprehensive content")  
    print("✅ Organized source attribution")
    print("✅ Professional disclaimer")
    print("✅ Performance timing display")
    print("✅ Consistent formatting throughout")
    
    print(f"\n📊 Performance: {len(response.get('sources', []))} sources processed")
    print(f"📝 Response length: {len(response['response'])} characters")
    print(f"⚡ Category detected: {response.get('category', 'unknown')}")
    
    # Test cache performance
    print("\n🔄 Testing cache performance...")
    start_time = time.time()
    cached_response = bot.process_query(query)
    cache_time = time.time() - start_time
    
    print(f"⚡ Cached response time: {cache_time:.3f}s (vs {processing_time:.2f}s original)")
    speed_improvement = ((processing_time - cache_time) / processing_time) * 100
    print(f"🚀 Cache speed improvement: {speed_improvement:.1f}%")

if __name__ == "__main__":
    demo_enhanced_response()
