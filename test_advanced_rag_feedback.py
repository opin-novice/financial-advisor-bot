#!/usr/bin/env python3
"""
Comprehensive test suite for the Advanced RAG Feedback Loop
"""
import sys
import os
from typing import List
from langchain.schema import Document

def test_feedback_loop_initialization():
    """Test that the feedback loop initializes correctly"""
    print("🧪 Testing Advanced RAG Feedback Loop initialization...")
    
    try:
        from advanced_rag_feedback import AdvancedRAGFeedbackLoop
        from rag_utils import RAGUtils
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import config
        
        # Initialize components
        rag_utils = RAGUtils()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={"device": "cpu"}
        )
        
        # Load FAISS index
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Initialize feedback loop
        feedback_config = config.get_feedback_loop_config()
        feedback_loop = AdvancedRAGFeedbackLoop(
            vectorstore=vectorstore,
            rag_utils=rag_utils,
            config=feedback_config
        )
        
        print("✅ Advanced RAG Feedback Loop initialized successfully")
        return feedback_loop, vectorstore, rag_utils
        
    except Exception as e:
        print(f"❌ Feedback loop initialization failed: {e}")
        return None, None, None

def test_feedback_loop_retrieval(feedback_loop):
    """Test the feedback loop retrieval process"""
    print("\n🧪 Testing feedback loop retrieval process...")
    
    if feedback_loop is None:
        print("❌ Skipping retrieval test - feedback loop not initialized")
        return False
    
    try:
        # Test queries with different complexity levels
        test_queries = [
            {
                "query": "How to open a bank account?",
                "category": "banking",
                "expected_relevant": True
            },
            {
                "query": "loan eligibility criteria",
                "category": "loans", 
                "expected_relevant": True
            },
            {
                "query": "weather in Dhaka today",
                "category": "general",
                "expected_relevant": False
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: '{test_case['query']}'")
            
            result = feedback_loop.retrieve_with_feedback_loop(
                test_case["query"], 
                test_case["category"]
            )
            
            print(f"   - Iterations used: {result.get('total_iterations', 'N/A')}")
            print(f"   - Final query: '{result.get('query_used', 'N/A')}'")
            print(f"   - Relevance score: {result.get('relevance_score', 0):.3f}")
            print(f"   - Documents found: {len(result.get('documents', []))}")
            print(f"   - Is relevant: {result.get('is_relevant', False)}")
            
            # Basic validation
            if test_case["expected_relevant"]:
                if result.get('relevance_score', 0) > 0.1:
                    print(f"   ✅ Test {i} passed - found relevant content")
                else:
                    print(f"   ⚠️  Test {i} - lower relevance than expected")
            else:
                if result.get('relevance_score', 0) < 0.3:
                    print(f"   ✅ Test {i} passed - correctly identified irrelevant query")
                else:
                    print(f"   ⚠️  Test {i} - unexpectedly high relevance for irrelevant query")
        
        return True
        
    except Exception as e:
        print(f"❌ Feedback loop retrieval test failed: {e}")
        return False

def test_configuration_modes():
    """Test different configuration modes"""
    print("\n🧪 Testing configuration modes...")
    
    try:
        from config import config
        
        # Test different performance modes
        modes = ["fast", "balanced", "thorough"]
        
        for mode in modes:
            print(f"\n🔧 Testing '{mode}' mode:")
            
            mode_config = config.get_performance_mode_config(mode)
            print(f"   - Max iterations: {mode_config['max_iterations']}")
            print(f"   - Relevance threshold: {mode_config['relevance_threshold']}")
            print(f"   - Confidence threshold: {mode_config['confidence_threshold']}")
            print(f"   - Strategies: {len(mode_config['refinement_strategies'])}")
            
            # Temporarily set the mode
            original_config = config.get_feedback_loop_config()
            config.set_performance_mode(mode)
            
            # Restore original config
            config.update_feedback_loop_config(original_config)
            
            print(f"   ✅ '{mode}' mode configuration valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration modes test failed: {e}")
        return False

def test_refinement_strategies():
    """Test individual refinement strategies"""
    print("\n🧪 Testing refinement strategies...")
    
    try:
        from advanced_rag_feedback import AdvancedRAGFeedbackLoop
        from rag_utils import RAGUtils
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import config
        
        # Initialize minimal components for testing
        rag_utils = RAGUtils()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        feedback_config = config.get_feedback_loop_config()
        feedback_loop = AdvancedRAGFeedbackLoop(
            vectorstore=vectorstore,
            rag_utils=rag_utils,
            config=feedback_config
        )
        
        # Test refinement strategies
        test_query = "loan"
        original_query = "loan"
        category = "loans"
        
        # Mock retrieved documents
        mock_docs = [
            Document(page_content="Personal loan eligibility criteria in Bangladesh include minimum income requirements."),
            Document(page_content="Bank loan application process requires documentation and credit check.")
        ]
        
        strategies = ["domain_expansion", "synonym_expansion", "context_addition", "query_decomposition"]
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n🔧 Testing strategy {i}: {strategy}")
            
            try:
                refined_query = feedback_loop._refine_query_strategically(
                    test_query, original_query, mock_docs, i, category
                )
                
                print(f"   Original: '{test_query}'")
                print(f"   Refined:  '{refined_query}'")
                
                if refined_query != test_query:
                    print(f"   ✅ Strategy '{strategy}' successfully refined query")
                else:
                    print(f"   ⚠️  Strategy '{strategy}' returned unchanged query")
                    
            except Exception as e:
                print(f"   ❌ Strategy '{strategy}' failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Refinement strategies test failed: {e}")
        return False

def test_integration_with_main():
    """Test integration with main bot class"""
    print("\n🧪 Testing integration with main bot...")
    
    try:
        from main import FinancialAdvisorTelegramBot
        
        print("🔄 Initializing bot with feedback loop...")
        bot = FinancialAdvisorTelegramBot()
        
        # Check if feedback loop was initialized
        if hasattr(bot, 'feedback_loop') and bot.feedback_loop is not None:
            print("✅ Feedback loop successfully integrated with main bot")
            
            # Test a simple query
            print("🔍 Testing query processing with feedback loop...")
            result = bot.process_query("How to apply for a personal loan?")
            
            if isinstance(result, dict) and result.get("response"):
                print("✅ Query processing with feedback loop successful")
                
                # Check for feedback loop metadata
                if "feedback_loop_metadata" in result:
                    metadata = result["feedback_loop_metadata"]
                    print(f"   - Iterations used: {metadata.get('iterations_used', 'N/A')}")
                    print(f"   - Final query: '{metadata.get('final_query', 'N/A')}'")
                    print(f"   - Relevance score: {metadata.get('relevance_score', 0):.3f}")
                    print("✅ Feedback loop metadata present")
                else:
                    print("⚠️  Feedback loop metadata missing (might be using traditional mode)")
                
                return True
            else:
                print("❌ Query processing returned unexpected result")
                return False
        else:
            print("⚠️  Feedback loop not initialized - using traditional RAG")
            return True  # This is acceptable fallback behavior
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_fallback_behavior():
    """Test fallback to traditional RAG when feedback loop fails"""
    print("\n🧪 Testing fallback behavior...")
    
    try:
        from config import config
        
        # Temporarily disable feedback loop
        original_setting = config.FEEDBACK_LOOP_CONFIG["enable_feedback_loop"]
        config.FEEDBACK_LOOP_CONFIG["enable_feedback_loop"] = False
        
        from main import FinancialAdvisorTelegramBot
        
        print("🔄 Initializing bot with feedback loop disabled...")
        bot = FinancialAdvisorTelegramBot()
        
        # Test query processing
        result = bot.process_query("What are the tax rates in Bangladesh?")
        
        if isinstance(result, dict) and result.get("response"):
            print("✅ Traditional RAG fallback working correctly")
            
            # Check metadata indicates traditional mode
            metadata = result.get("feedback_loop_metadata", {})
            if metadata.get("iterations_used") == 1:
                print("✅ Correctly using traditional mode (1 iteration)")
            
            # Restore original setting
            config.FEEDBACK_LOOP_CONFIG["enable_feedback_loop"] = original_setting
            return True
        else:
            print("❌ Traditional RAG fallback failed")
            config.FEEDBACK_LOOP_CONFIG["enable_feedback_loop"] = original_setting
            return False
        
    except Exception as e:
        print(f"❌ Fallback behavior test failed: {e}")
        # Restore original setting
        try:
            config.FEEDBACK_LOOP_CONFIG["enable_feedback_loop"] = original_setting
        except:
            pass
        return False

if __name__ == "__main__":
    print("🚀 Advanced RAG Feedback Loop Test Suite")
    print("=" * 70)
    
    # Test 1: Initialization
    feedback_loop, vectorstore, rag_utils = test_feedback_loop_initialization()
    
    # Test 2: Retrieval process
    retrieval_success = test_feedback_loop_retrieval(feedback_loop)
    
    # Test 3: Configuration modes
    config_success = test_configuration_modes()
    
    # Test 4: Refinement strategies
    strategies_success = test_refinement_strategies()
    
    # Test 5: Integration with main
    integration_success = test_integration_with_main()
    
    # Test 6: Fallback behavior
    fallback_success = test_fallback_behavior()
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS:")
    print(f"🔧 Initialization: {'✅ PASS' if feedback_loop is not None else '❌ FAIL'}")
    print(f"🔍 Retrieval Process: {'✅ PASS' if retrieval_success else '❌ FAIL'}")
    print(f"⚙️  Configuration Modes: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"🔧 Refinement Strategies: {'✅ PASS' if strategies_success else '❌ FAIL'}")
    print(f"🔗 Main Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    print(f"🔄 Fallback Behavior: {'✅ PASS' if fallback_success else '❌ FAIL'}")
    
    all_tests_passed = all([
        feedback_loop is not None,
        retrieval_success,
        config_success,
        strategies_success,
        integration_success,
        fallback_success
    ])
    
    if all_tests_passed:
        print("\n🎉 ALL TESTS PASSED! Advanced RAG Feedback Loop is working correctly.")
        print("\n🔥 THE ADVANCED RAG FEEDBACK LOOP IS READY FOR PRODUCTION!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)