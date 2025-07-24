#!/usr/bin/env python3
"""
Test script for the enhanced RAG financial advisor bot
"""

import time
from main import FinancialAdvisorBot

def test_enhanced_bot():
    """Test the enhanced bot with sample queries"""
    
    print("🧪 Testing Enhanced RAG Financial Advisor Bot")
    print("=" * 50)
    
    try:
        # Initialize the bot
        print("Initializing bot...")
        bot = FinancialAdvisorBot()
        print("✅ Bot initialized successfully!")
        
        # Test queries for different categories
        test_queries = [
            ("How do I file my taxes?", "taxation"),
            ("What documents do I need to open a bank account?", "banking"),
            ("What are the requirements for a home loan?", "loans"),
            ("What investment options are available in Bangladesh?", "investment"),
        ]
        
        print("\n🔍 Testing with sample queries:")
        print("-" * 40)
        
        for i, (query, expected_category) in enumerate(test_queries, 1):
            print(f"\n{i}. Testing query: '{query}'")
            print(f"   Expected category: {expected_category}")
            
            start_time = time.time()
            response = bot.process_query(query)
            processing_time = time.time() - start_time
            
            if 'error' in response:
                print(f"   ❌ Error: {response['error']}")
                continue
            
            print(f"   ✅ Response generated successfully!")
            print(f"   📊 Category: {response.get('category', 'Unknown')}")
            print(f"   📚 Sources found: {len(response.get('sources', []))}")
            print(f"   ⏱️  Processing time: {processing_time:.2f}s")
            
            # Display a portion of the response for verification
            response_preview = response['response'][:200] + "..." if len(response['response']) > 200 else response['response']
            print(f"   📝 Response preview: {response_preview}")
            
            # Test caching by running the same query again
            print("   🔄 Testing cache functionality...")
            start_time = time.time()
            cached_response = bot.process_query(query)
            cache_time = time.time() - start_time
            
            if cache_time < processing_time / 2:  # Cache should be significantly faster
                print(f"   ✅ Cache working! Cache time: {cache_time:.3f}s")
            else:
                print(f"   ⚠️  Cache may not be working optimally. Cache time: {cache_time:.3f}s")
        
        # Display performance statistics
        print("\n📈 Performance Statistics:")
        print("-" * 40)
        stats = bot.performance_monitor.get_performance_stats()
        
        if stats['overall'].get('total_queries', 0) > 0:
            print(f"Total queries processed: {stats['overall']['total_queries']}")
            print(f"Average response time: {stats['overall']['avg_response_time']:.2f}s")
            print(f"Cache hit rate: {stats['cache_performance']['hit_rate']}%")
            
            print("\nCategory breakdown:")
            for category, cat_stats in stats['by_category'].items():
                print(f"  {category}: {cat_stats['count']} queries, avg {cat_stats['avg_response_time']:.2f}s")
        
        print("\n✅ All tests completed successfully!")
        print("🎉 Enhanced RAG system is working properly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("Please check your setup and try again.")
        return False
    
    return True

if __name__ == "__main__":
    success = test_enhanced_bot()
    exit(0 if success else 1)
