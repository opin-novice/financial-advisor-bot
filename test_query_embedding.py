#!/usr/bin/env python3
"""
Comprehensive test for query embedding functionality in the RAG system
"""
import os
import sys
import numpy as np
import torch
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_embedding_model_loading():
    """Test if the embedding model loads correctly"""
    print("🧪 Testing embedding model loading...")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import config
        
        # Initialize embeddings with the same configuration as main.py
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL, 
            model_kwargs={"device": "cpu"}
        )
        
        print(f"✅ Embedding model loaded successfully: {config.EMBEDDING_MODEL}")
        print(f"📊 Model device: cpu")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        return None

def test_query_embedding_generation(embeddings):
    """Test query embedding generation"""
    print("\n🧪 Testing query embedding generation...")
    
    if not embeddings:
        print("❌ Skipping - embedding model not loaded")
        return False
    
    try:
        # Test queries in different languages and domains
        test_queries = [
            "How to open a bank account in Bangladesh?",
            "What are the tax rates for individuals?",
            "Investment options available in Bangladesh",
            "ব্যাংক অ্যাকাউন্ট খোলার নিয়ম কি?",  # Bengali query
            "loan eligibility criteria",
            "mobile banking services"
        ]
        
        embeddings_results = []
        
        for i, query in enumerate(test_queries, 1):
            try:
                # Generate embedding for the query
                query_embedding = embeddings.embed_query(query)
                
                # Validate embedding properties
                embedding_array = np.array(query_embedding)
                
                print(f"✅ Query {i}: '{query[:50]}...'")
                print(f"   📏 Embedding dimension: {len(query_embedding)}")
                print(f"   📊 Embedding shape: {embedding_array.shape}")
                print(f"   🔢 Data type: {embedding_array.dtype}")
                print(f"   📈 Min/Max values: {embedding_array.min():.4f} / {embedding_array.max():.4f}")
                print(f"   🎯 L2 norm: {np.linalg.norm(embedding_array):.4f}")
                
                # Store for similarity testing
                embeddings_results.append({
                    'query': query,
                    'embedding': query_embedding,
                    'norm': np.linalg.norm(embedding_array)
                })
                
            except Exception as e:
                print(f"❌ Failed to generate embedding for query {i}: {e}")
                return False
        
        print(f"\n✅ Successfully generated embeddings for {len(embeddings_results)} queries")
        return embeddings_results
        
    except Exception as e:
        print(f"❌ Query embedding generation test failed: {e}")
        return False

def test_embedding_similarity(embeddings_results):
    """Test embedding similarity calculations"""
    print("\n🧪 Testing embedding similarity calculations...")
    
    if not embeddings_results or len(embeddings_results) < 2:
        print("❌ Skipping - insufficient embedding results")
        return False
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("📊 Similarity Matrix:")
        print("-" * 80)
        
        # Calculate pairwise similarities
        for i, result1 in enumerate(embeddings_results):
            for j, result2 in enumerate(embeddings_results):
                if i <= j:  # Only calculate upper triangle
                    emb1 = np.array(result1['embedding']).reshape(1, -1)
                    emb2 = np.array(result2['embedding']).reshape(1, -1)
                    
                    similarity = cosine_similarity(emb1, emb2)[0][0]
                    
                    query1_short = result1['query'][:30] + "..." if len(result1['query']) > 30 else result1['query']
                    query2_short = result2['query'][:30] + "..." if len(result2['query']) > 30 else result2['query']
                    
                    print(f"🔗 Query {i+1} vs Query {j+1}: {similarity:.4f}")
                    print(f"   '{query1_short}' <-> '{query2_short}'")
                    
                    # Check for reasonable similarity ranges
                    if i == j and similarity < 0.99:
                        print(f"⚠️  Warning: Self-similarity should be ~1.0, got {similarity:.4f}")
                    elif i != j and similarity > 0.95:
                        print(f"⚠️  Warning: Very high similarity between different queries: {similarity:.4f}")
        
        print("✅ Embedding similarity calculations completed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding similarity test failed: {e}")
        return False

def test_faiss_index_compatibility(embeddings):
    """Test compatibility with FAISS index"""
    print("\n🧪 Testing FAISS index compatibility...")
    
    if not embeddings:
        print("❌ Skipping - embedding model not loaded")
        return False
    
    try:
        from langchain_community.vectorstores import FAISS
        from config import config
        
        # Check if FAISS index exists
        if not os.path.exists(config.FAISS_INDEX_PATH):
            print(f"❌ FAISS index not found at: {config.FAISS_INDEX_PATH}")
            return False
        
        # Load FAISS index
        print(f"📂 Loading FAISS index from: {config.FAISS_INDEX_PATH}")
        vectorstore = FAISS.load_local(
            config.FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        print("✅ FAISS index loaded successfully")
        
        # Test query retrieval
        test_query = "How to open a bank account in Bangladesh?"
        print(f"🔍 Testing retrieval with query: '{test_query}'")
        
        # Retrieve similar documents
        docs = vectorstore.similarity_search(test_query, k=5)
        
        print(f"📄 Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"   Doc {i}: {content_preview}")
        
        # Test with score
        docs_with_scores = vectorstore.similarity_search_with_score(test_query, k=3)
        print(f"\n📊 Top 3 documents with similarity scores:")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
            print(f"   {i}. Score: {score:.4f} | Content: {content_preview}")
        
        print("✅ FAISS index compatibility test passed")
        return True
        
    except Exception as e:
        print(f"❌ FAISS index compatibility test failed: {e}")
        return False

def test_cross_encoder_compatibility(embeddings_results):
    """Test compatibility with cross-encoder re-ranking"""
    print("\n🧪 Testing cross-encoder compatibility...")
    
    if not embeddings_results:
        print("❌ Skipping - no embedding results available")
        return False
    
    try:
        from sentence_transformers import CrossEncoder
        from config import config
        
        # Initialize cross-encoder
        cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
        print(f"✅ Cross-encoder loaded: {config.CROSS_ENCODER_MODEL}")
        
        # Test query-document pairs
        test_query = "How to open a bank account?"
        test_documents = [
            "To open a bank account in Bangladesh, you need to provide your NID card, passport size photos, and initial deposit.",
            "Investment options in Bangladesh include fixed deposits, mutual funds, and stock market investments.",
            "Tax rates for individuals in Bangladesh vary based on income levels and tax zones."
        ]
        
        print(f"🔍 Testing cross-encoder with query: '{test_query}'")
        
        # Calculate relevance scores
        pairs = [(test_query, doc) for doc in test_documents]
        scores = cross_encoder.predict(pairs)
        
        print("📊 Cross-encoder relevance scores:")
        for i, (doc, score) in enumerate(zip(test_documents, scores), 1):
            doc_preview = doc[:60] + "..." if len(doc) > 60 else doc
            print(f"   {i}. Score: {score:.4f} | Doc: {doc_preview}")
        
        # Check if scores are reasonable
        max_score = max(scores)
        min_score = min(scores)
        print(f"📈 Score range: {min_score:.4f} to {max_score:.4f}")
        
        if max_score > config.RELEVANCE_THRESHOLD:
            print(f"✅ At least one document exceeds relevance threshold ({config.RELEVANCE_THRESHOLD})")
        else:
            print(f"⚠️  No documents exceed relevance threshold ({config.RELEVANCE_THRESHOLD})")
        
        print("✅ Cross-encoder compatibility test passed")
        return True
        
    except Exception as e:
        print(f"❌ Cross-encoder compatibility test failed: {e}")
        return False

def test_multilingual_embedding():
    """Test multilingual embedding capabilities"""
    print("\n🧪 Testing multilingual embedding capabilities...")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import config
        
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL, 
            model_kwargs={"device": "cpu"}
        )
        
        # Test queries in different languages
        multilingual_queries = [
            ("English", "How to open a bank account?"),
            ("Bengali", "ব্যাংক অ্যাকাউন্ট কিভাবে খুলবো?"),
            ("English Financial", "What are the loan eligibility criteria?"),
            ("Bengali Financial", "ঋণের যোগ্যতার মাপদণ্ড কি?")
        ]
        
        embeddings_by_lang = {}
        
        for lang, query in multilingual_queries:
            try:
                embedding = embeddings.embed_query(query)
                embeddings_by_lang[lang] = {
                    'query': query,
                    'embedding': np.array(embedding),
                    'norm': np.linalg.norm(np.array(embedding))
                }
                print(f"✅ {lang}: Generated embedding (dim: {len(embedding)})")
                
            except Exception as e:
                print(f"❌ Failed to generate embedding for {lang}: {e}")
        
        # Test cross-lingual similarity
        if len(embeddings_by_lang) >= 2:
            from sklearn.metrics.pairwise import cosine_similarity
            
            print("\n🌐 Cross-lingual similarity analysis:")
            langs = list(embeddings_by_lang.keys())
            
            for i, lang1 in enumerate(langs):
                for j, lang2 in enumerate(langs):
                    if i < j:
                        emb1 = embeddings_by_lang[lang1]['embedding'].reshape(1, -1)
                        emb2 = embeddings_by_lang[lang2]['embedding'].reshape(1, -1)
                        
                        similarity = cosine_similarity(emb1, emb2)[0][0]
                        print(f"🔗 {lang1} <-> {lang2}: {similarity:.4f}")
        
        print("✅ Multilingual embedding test completed")
        return True
        
    except Exception as e:
        print(f"❌ Multilingual embedding test failed: {e}")
        return False

def main():
    """Run all embedding tests"""
    print("🚀 Query Embedding Functionality Test")
    print("=" * 80)
    
    # Test results tracking
    test_results = {}
    
    # Test 1: Embedding model loading
    embeddings = test_embedding_model_loading()
    test_results['model_loading'] = embeddings is not None
    
    # Test 2: Query embedding generation
    embeddings_results = test_query_embedding_generation(embeddings)
    test_results['embedding_generation'] = embeddings_results is not False
    
    # Test 3: Embedding similarity
    similarity_success = test_embedding_similarity(embeddings_results)
    test_results['similarity_calculation'] = similarity_success
    
    # Test 4: FAISS index compatibility
    faiss_success = test_faiss_index_compatibility(embeddings)
    test_results['faiss_compatibility'] = faiss_success
    
    # Test 5: Cross-encoder compatibility
    cross_encoder_success = test_cross_encoder_compatibility(embeddings_results)
    test_results['cross_encoder_compatibility'] = cross_encoder_success
    
    # Test 6: Multilingual embedding
    multilingual_success = test_multilingual_embedding()
    test_results['multilingual_support'] = multilingual_success
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 QUERY EMBEDDING TEST RESULTS:")
    print("=" * 80)
    
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        test_display = test_name.replace('_', ' ').title()
        print(f"{test_display:.<50} {status}")
    
    # Overall result
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL EMBEDDING TESTS PASSED!")
        print("Your query embedding functionality is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
        print("Please check the error messages above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
