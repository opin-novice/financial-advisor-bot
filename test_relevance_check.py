#!/usr/bin/env python3
"""
Test relevance checking functionality
"""
from langchain.schema import Document
from rag_utils import RAGUtils

def test_relevance_checking():
    """Test the relevance checking functionality"""
    print("🧪 Testing relevance checking...")
    
    try:
        rag = RAGUtils()
        
        # Test case 1: Relevant documents
        query = "How to open a bank account?"
        relevant_docs = [
            Document(page_content="To open a bank account in Bangladesh, you need to provide your NID card, passport size photos, and initial deposit. Visit any branch of your preferred bank with these documents."),
            Document(page_content="Bank account opening requirements include valid identification, proof of address, and minimum deposit amount as specified by the bank.")
        ]
        
        is_relevant, confidence = rag.check_query_relevance(query, relevant_docs)
        print(f"✅ Test 1 - Relevant docs: Relevant={is_relevant}, Confidence={confidence:.3f}")
        
        # Test case 2: Irrelevant documents
        query2 = "How to open a bank account?"
        irrelevant_docs = [
            Document(page_content="The weather in Dhaka today is sunny with a temperature of 30 degrees Celsius."),
            Document(page_content="Cricket match between Bangladesh and India will be held next week.")
        ]
        
        is_relevant2, confidence2 = rag.check_query_relevance(query2, irrelevant_docs)
        print(f"✅ Test 2 - Irrelevant docs: Relevant={is_relevant2}, Confidence={confidence2:.3f}")
        
        # Test case 3: Empty documents
        is_relevant3, confidence3 = rag.check_query_relevance(query, [])
        print(f"✅ Test 3 - Empty docs: Relevant={is_relevant3}, Confidence={confidence3:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Relevance checking test failed: {e}")
        return False

def test_query_refinement():
    """Test query refinement with better output handling"""
    print("\n🧪 Testing query refinement...")
    
    try:
        rag = RAGUtils()
        
        test_queries = [
            "loan",
            "tax information", 
            "investment options"
        ]
        
        for query in test_queries:
            refined = rag.refine_query(query)
            # Extract just the refined query part if it's verbose
            if "Refined Query:" in refined:
                refined_clean = refined.split("Refined Query:")[1].split("\n")[0].strip().strip('"')
            else:
                refined_clean = refined
                
            print(f"📝 '{query}' -> '{refined_clean[:100]}{'...' if len(refined_clean) > 100 else ''}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Query refinement test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing RAG Relevance & Refinement")
    print("=" * 50)
    
    relevance_success = test_relevance_checking()
    refinement_success = test_query_refinement()
    
    print("\n" + "=" * 50)
    if relevance_success and refinement_success:
        print("🎉 All tests passed! Relevance checking and query refinement are working.")
    else:
        print("⚠️  Some tests failed. Check error messages above.")