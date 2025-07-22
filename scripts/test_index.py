#!/usr/bin/env python3
"""
Script for testing FAISS index functionality
This script verifies that the FAISS index is working correctly
"""

import os
import sys
import logging
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/test_index.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class IndexTester:
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.logger = setup_logging()
        
        # Test queries for different categories
        self.test_queries = {
            "banking": [
                "How to open a bank account?",
                "What are the requirements for personal banking?",
                "Bank account opening process"
            ],
            "investment": [
                "Investment opportunities in Bangladesh",
                "How to invest in savings certificate?",
                "Foreign investment guidelines"
            ],
            "loans": [
                "Car loan eligibility criteria",
                "Home loan application process",
                "Loan documentation requirements"
            ],
            "taxation": [
                "Income tax calculation",
                "Tax return filing process",
                "Personal taxation guide"
            ],
            "regulations": [
                "Banking regulations in Bangladesh",
                "Financial institution acts",
                "Compliance requirements"
            ],
            "sme": [
                "SME loan eligibility",
                "Small business startup guide",
                "Business account opening"
            ]
        }
    
    def load_index(self):
        """Load the FAISS index"""
        try:
            if not os.path.exists(self.index_path):
                self.logger.error(f"FAISS index not found at {self.index_path}")
                return None
            
            # Initialize embeddings (same as used in build_faiss_index.py)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"}
            )
            
            # Load the vectorstore
            vectorstore = FAISS.load_local(
                self.index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            self.logger.info(f"Successfully loaded FAISS index from {self.index_path}")
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {str(e)}")
            return None
    
    def test_similarity_search(self, vectorstore, query: str, k: int = 3) -> List[Dict]:
        """Test similarity search functionality"""
        try:
            results = vectorstore.similarity_search(query, k=k)
            
            search_results = []
            for i, doc in enumerate(results):
                result = {
                    "rank": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content)
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search for query '{query}': {str(e)}")
            return []
    
    def test_similarity_search_with_score(self, vectorstore, query: str, k: int = 3) -> List[Dict]:
        """Test similarity search with score functionality"""
        try:
            results = vectorstore.similarity_search_with_score(query, k=k)
            
            search_results = []
            for i, (doc, score) in enumerate(results):
                result = {
                    "rank": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "content_length": len(doc.page_content)
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search with score for query '{query}': {str(e)}")
            return []
    
    def run_comprehensive_test(self):
        """Run comprehensive tests on the FAISS index"""
        print("="*70)
        print("                    FAISS INDEX TESTING")
        print("="*70)
        
        # Load the index
        print("\n📊 Loading FAISS index...")
        vectorstore = self.load_index()
        
        if not vectorstore:
            print("❌ Failed to load FAISS index. Cannot proceed with testing.")
            return False
        
        print("✅ FAISS index loaded successfully")
        
        # Test basic functionality
        print("\n🔍 Testing basic similarity search...")
        test_passed = 0
        total_tests = 0
        
        for category, queries in self.test_queries.items():
            print(f"\n📂 Testing {category.upper()} queries:")
            
            for query in queries:
                total_tests += 1
                print(f"\n  Query: {query}")
                
                # Test similarity search
                results = self.test_similarity_search(vectorstore, query, k=3)
                
                if results:
                    test_passed += 1
                    print(f"    ✅ Found {len(results)} results")
                    
                    for result in results[:2]:  # Show top 2 results
                        category_found = result['metadata'].get('category', 'unknown')
                        file_name = result['metadata'].get('file_name', 'unknown')
                        print(f"      - [{category_found}] {file_name}")
                        print(f"        Content: {result['content'][:100]}...")
                else:
                    print("    ❌ No results found")
        
        # Test search with scores
        print("\n🎯 Testing similarity search with scores...")
        sample_query = "How to open a bank account in Bangladesh?"
        scored_results = self.test_similarity_search_with_score(vectorstore, sample_query, k=5)
        
        if scored_results:
            print(f"\n  Query: {sample_query}")
            print("  Results with scores:")
            for result in scored_results:
                print(f"    {result['rank']}. Score: {result['score']:.4f}")
                print(f"       Category: {result['metadata'].get('category', 'unknown')}")
                print(f"       File: {result['metadata'].get('file_name', 'unknown')}")
                print(f"       Content: {result['content'][:80]}...")
                print()
        
        # Print summary
        print("="*70)
        print("                        TEST SUMMARY")
        print("="*70)
        print(f"Total queries tested: {total_tests}")
        print(f"Successful queries: {test_passed}")
        print(f"Success rate: {(test_passed/total_tests)*100:.1f}%")
        
        if test_passed == total_tests:
            print("\n🎉 All tests passed! FAISS index is working correctly.")
            return True
        else:
            print(f"\n⚠️  {total_tests - test_passed} tests failed. Check the logs for details.")
            return False

def main():
    """Main function"""
    tester = IndexTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n✅ FAISS index testing completed successfully!")
    else:
        print("\n❌ FAISS index testing completed with issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
