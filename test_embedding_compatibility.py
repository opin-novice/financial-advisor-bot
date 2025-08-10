#!/usr/bin/env python3
"""
Test to check embedding model compatibility and cosine similarity scores
"""
import os
import sys
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_embedding_models():
    """Check what embedding models are being used for documents vs queries"""
    print("🔍 Checking Embedding Model Configuration...")
    print("=" * 80)
    
    # Check document embedding model (from docadd.py)
    try:
        with open('/Users/sayed/Downloads/final_rag/docadd.py', 'r') as f:
            docadd_content = f.read()
            
        # Extract embedding model from docadd.py
        import re
        doc_model_match = re.search(r'EMBEDDING_MODEL\s*=\s*["\']([^"\']+)["\']', docadd_content)
        doc_embedding_model = doc_model_match.group(1) if doc_model_match else "Not found"
        
        print(f"📄 Document Embedding Model (docadd.py): {doc_embedding_model}")
        
    except Exception as e:
        print(f"❌ Error reading docadd.py: {e}")
        doc_embedding_model = "Error reading file"
    
    # Check query embedding model (from config.py)
    try:
        from config import config
        query_embedding_model = config.EMBEDDING_MODEL
        print(f"🔍 Query Embedding Model (config.py): {query_embedding_model}")
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        query_embedding_model = "Error reading config"
    
    # Check if they match
    print("\n" + "=" * 80)
    if doc_embedding_model == query_embedding_model:
        print("✅ MODELS MATCH: Document and query embedding models are the same")
        compatibility_status = "COMPATIBLE"
    else:
        print("❌ MODELS MISMATCH: Document and query embedding models are different!")
        print(f"   📄 Documents indexed with: {doc_embedding_model}")
        print(f"   🔍 Queries processed with: {query_embedding_model}")
        compatibility_status = "INCOMPATIBLE"
    
    return {
        'doc_model': doc_embedding_model,
        'query_model': query_embedding_model,
        'compatible': doc_embedding_model == query_embedding_model,
        'status': compatibility_status
    }

def test_cosine_similarity_scores():
    """Test and display cosine similarity scores from retrieval"""
    print("\n🧪 Testing Cosine Similarity Scores in Retrieval...")
    print("=" * 80)
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import config
        
        # Initialize embeddings with current query model
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL, 
            model_kwargs={"device": "cpu"}
        )
        
        # Load FAISS index
        vectorstore = FAISS.load_local(
            config.FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Test queries
        test_queries = [
            "How to open a bank account in Bangladesh?",
            "What are the loan eligibility criteria?",
            "Tax calculation methods for individuals",
            "Investment opportunities in Bangladesh"
        ]
        
        print("📊 Cosine Similarity Scores from FAISS Retrieval:")
        print("-" * 80)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: '{query}'")
            
            # Get documents with similarity scores
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
            
            print(f"📄 Retrieved {len(docs_with_scores)} documents with scores:")
            
            for j, (doc, score) in enumerate(docs_with_scores, 1):
                # FAISS returns distance, convert to similarity
                # For cosine distance: similarity = 1 - distance
                cosine_similarity = 1 - score
                
                content_preview = doc.page_content[:100].replace('\n', ' ').strip()
                if len(content_preview) > 80:
                    content_preview = content_preview[:80] + "..."
                
                print(f"   {j}. Distance: {score:.4f} | Similarity: {cosine_similarity:.4f}")
                print(f"      Content: {content_preview}")
                
                # Analyze score quality
                if cosine_similarity > 0.8:
                    quality = "🟢 Excellent"
                elif cosine_similarity > 0.6:
                    quality = "🟡 Good"
                elif cosine_similarity > 0.4:
                    quality = "🟠 Fair"
                else:
                    quality = "🔴 Poor"
                
                print(f"      Quality: {quality}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing cosine similarity scores: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_encoder_scores():
    """Test cross-encoder relevance scores"""
    print("\n🧪 Testing Cross-Encoder Relevance Scores...")
    print("=" * 80)
    
    try:
        from sentence_transformers import CrossEncoder
        from config import config
        
        # Initialize cross-encoder
        cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
        
        # Test query-document pairs
        test_query = "How to open a bank account in Bangladesh?"
        
        # Sample documents (you can replace with actual retrieved documents)
        test_documents = [
            "To open a bank account in Bangladesh, you need to provide your NID card, passport size photos, and initial deposit amount.",
            "Investment opportunities in Bangladesh include fixed deposits, mutual funds, stock market, and real estate investments.",
            "Tax calculation for individuals in Bangladesh depends on income level, age, and residential status of the taxpayer.",
            "Loan eligibility criteria include minimum income requirements, credit history, employment status, and collateral security."
        ]
        
        print(f"🔍 Query: '{test_query}'")
        print("📊 Cross-Encoder Relevance Scores:")
        print("-" * 60)
        
        # Calculate relevance scores
        pairs = [(test_query, doc) for doc in test_documents]
        scores = cross_encoder.predict(pairs)
        
        for i, (doc, score) in enumerate(zip(test_documents, scores), 1):
            doc_preview = doc[:80] + "..." if len(doc) > 80 else doc
            
            # Analyze score quality
            if score > 5.0:
                quality = "🟢 Highly Relevant"
            elif score > 0.0:
                quality = "🟡 Relevant"
            elif score > -5.0:
                quality = "🟠 Somewhat Relevant"
            else:
                quality = "🔴 Not Relevant"
            
            print(f"{i}. Score: {score:8.4f} | {quality}")
            print(f"   Doc: {doc_preview}")
            
            # Check against threshold
            threshold_status = "✅ Above" if score >= config.RELEVANCE_THRESHOLD else "❌ Below"
            print(f"   Threshold ({config.RELEVANCE_THRESHOLD}): {threshold_status}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing cross-encoder scores: {e}")
        return False

def test_actual_retrieval_pipeline():
    """Test the actual retrieval pipeline to see similarity scores in action"""
    print("\n🧪 Testing Actual Retrieval Pipeline...")
    print("=" * 80)
    
    try:
        from main import FinancialAdvisorTelegramBot
        
        # Initialize bot
        bot = FinancialAdvisorTelegramBot()
        
        # Test query
        test_query = "How to open a bank account in Bangladesh?"
        print(f"🔍 Processing query: '{test_query}'")
        
        # Process query and capture output
        import io
        import contextlib
        
        # Capture print statements
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            result = bot.process_query(test_query)
        
        output = captured_output.getvalue()
        
        # Extract relevance scores from output
        import re
        score_matches = re.findall(r'Document relevance score: ([\d.-]+)', output)
        
        if score_matches:
            print("📊 Cross-Encoder Relevance Scores from Actual Pipeline:")
            for i, score in enumerate(score_matches, 1):
                score_float = float(score)
                if score_float > 5.0:
                    quality = "🟢 Highly Relevant"
                elif score_float > 0.0:
                    quality = "🟡 Relevant"
                else:
                    quality = "🔴 Not Relevant"
                
                print(f"   Document {i}: {score_float:.3f} ({quality})")
        else:
            print("⚠️  No relevance scores found in pipeline output")
        
        # Show result summary
        if isinstance(result, dict):
            print(f"\n✅ Pipeline completed successfully")
            print(f"📄 Sources found: {len(result.get('sources', []))}")
            print(f"📝 Response generated: {'Yes' if result.get('response') else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing actual retrieval pipeline: {e}")
        return False

def analyze_embedding_compatibility_impact():
    """Analyze the impact of embedding model mismatch"""
    print("\n📊 Analyzing Embedding Model Compatibility Impact...")
    print("=" * 80)
    
    compatibility_info = check_embedding_models()
    
    if not compatibility_info['compatible']:
        print("⚠️  CRITICAL ISSUE DETECTED: Embedding Model Mismatch")
        print("\n🔍 Impact Analysis:")
        print("1. 📄 Documents were indexed using:", compatibility_info['doc_model'])
        print("2. 🔍 Queries are processed using:", compatibility_info['query_model'])
        print("\n❌ Problems this causes:")
        print("   • Cosine similarity scores will be inaccurate")
        print("   • Document retrieval quality will be poor")
        print("   • Relevant documents may not be found")
        print("   • Irrelevant documents may be ranked highly")
        
        print("\n🔧 Solutions:")
        print("1. Re-index documents with the same model used for queries")
        print("2. Or change query model to match document model")
        print("3. Recommended: Use the same model for both (sentence-transformers/all-mpnet-base-v2)")
        
        return False
    else:
        print("✅ No compatibility issues detected")
        print("📊 Both documents and queries use the same embedding model")
        print("🎯 Cosine similarity scores should be accurate and meaningful")
        return True

def main():
    """Run all embedding compatibility tests"""
    print("🚀 Embedding Model Compatibility and Similarity Score Analysis")
    print("=" * 80)
    
    # Test results tracking
    test_results = {}
    
    # Test 1: Check embedding model compatibility
    compatibility_ok = analyze_embedding_compatibility_impact()
    test_results['model_compatibility'] = compatibility_ok
    
    # Test 2: Test cosine similarity scores
    cosine_test_ok = test_cosine_similarity_scores()
    test_results['cosine_similarity_test'] = cosine_test_ok
    
    # Test 3: Test cross-encoder scores
    cross_encoder_ok = test_cross_encoder_scores()
    test_results['cross_encoder_test'] = cross_encoder_ok
    
    # Test 4: Test actual pipeline
    pipeline_ok = test_actual_retrieval_pipeline()
    test_results['pipeline_test'] = pipeline_ok
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 EMBEDDING COMPATIBILITY TEST RESULTS:")
    print("=" * 80)
    
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        test_display = test_name.replace('_', ' ').title()
        print(f"{test_display:.<50} {status}")
    
    # Overall assessment
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if not test_results.get('model_compatibility', True):
        print("\n🚨 CRITICAL: Embedding model mismatch detected!")
        print("This will significantly impact retrieval quality.")
        print("Please fix the embedding model compatibility issue.")
        return False
    elif passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your embedding models are compatible and similarity scores are working correctly.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
        print("Check the error messages above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
