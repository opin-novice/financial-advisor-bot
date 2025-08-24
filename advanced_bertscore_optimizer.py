#!/usr/bin/env python3
"""
Cross-Encoder Optimization for RAG Pipeline
Improves document ranking and retrieval quality
"""
import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity

class CrossEncoderOptimizer:
    """Optimizes document ranking using cross-encoder models"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        """
        Initialize cross-encoder optimizer
        
        Args:
            model_name: Cross-encoder model name
        """
        print(f"[INFO] Initializing Cross-Encoder Optimizer with model: {model_name}")
        
        self.model_name = model_name
        self.cross_encoder = None
        self.batch_size = 16
        
        # Initialize components
        self._init_cross_encoder()
        print(f"[INFO] [OK] Cross-Encoder Optimizer initialized successfully")
    
    def _init_cross_encoder(self):
        """Initialize the cross-encoder model"""
        try:
            print(f"[INFO] Loading cross-encoder model: {self.model_name}")
            self.cross_encoder = CrossEncoder(
                self.model_name,
                max_length=512,
                device='cpu'
            )
            print(f"[INFO] [OK] Cross-encoder model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load cross-encoder model: {e}")
            self.cross_encoder = None
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: User query
            documents: List of Document objects
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of Document objects
        """
        if not documents or not self.cross_encoder:
            return documents[:top_k] if documents else []
        
        try:
            print(f"[INFO] [RERANK] Reranking {len(documents)} documents with cross-encoder...")
            
            # Prepare query-document pairs
            pairs = []
            valid_docs = []
            
            for doc in documents:
                content = doc.page_content.strip()
                if content:
                    pairs.append([query, content])
                    valid_docs.append(doc)
            
            if not pairs:
                print("[WARNING] No valid document content found for reranking")
                return documents[:top_k]
            
            # Get relevance scores
            print(f"[INFO] [PREDICT] Computing relevance scores for {len(pairs)} pairs...")
            scores = self.cross_encoder.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True
            )
            
            # Combine documents with scores and sort
            doc_scores = list(zip(valid_docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k documents
            reranked_docs = [doc for doc, score in doc_scores[:top_k]]
            
            print(f"[INFO] [OK] Cross-encoder reranking completed - returning {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            print(f"[ERROR] Cross-encoder reranking failed: {e}")
            return documents[:top_k]
    
    def optimize_document_ranking(self, query: str, documents: List[Document], 
                               threshold: float = 0.2) -> List[Document]:
        """
        Optimize document ranking with quality filtering
        
        Args:
            query: User query
            documents: List of Document objects
            threshold: Minimum relevance score threshold
            
        Returns:
            Optimized list of Document objects
        """
        if not documents:
            return []
        
        # First, rerank with cross-encoder
        reranked_docs = self.rerank_documents(query, documents, top_k=len(documents))
        
        if not reranked_docs or not self.cross_encoder:
            return reranked_docs[:5]  # Return top 5 without filtering
        
        try:
            # Score documents for quality filtering
            pairs = []
            valid_docs = []
            
            for doc in reranked_docs:
                content = doc.page_content.strip()
                if content:
                    pairs.append([query, content])
                    valid_docs.append(doc)
            
            if not pairs:
                return reranked_docs[:5]
            
            # Get scores
            scores = self.cross_encoder.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True
            )
            
            # Filter by threshold and return top documents
            filtered_docs = []
            for doc, score in zip(valid_docs, scores):
                if score >= threshold:
                    # Add score to metadata
                    doc.metadata['cross_encoder_score'] = float(score)
                    doc.metadata['ranking_quality'] = 'high' if score >= 0.5 else 'medium' if score >= 0.3 else 'low'
                    filtered_docs.append(doc)
            
            print(f"[INFO] [OPTIMIZE] Cross-encoder optimization completed - "
                  f"{len(filtered_docs)}/{len(reranked_docs)} documents passed threshold ({threshold})")
            
            return filtered_docs[:5]  # Return top 5 filtered documents
            
        except Exception as e:
            print(f"[ERROR] Cross-encoder optimization failed: {e}")
            return reranked_docs[:5]
    
    def evaluate_cross_encoder_performance(self, queries: List[str], 
                                        documents_sets: List[List[Document]], 
                                        reference_sets: List[List[Document]]) -> Dict[str, float]:
        """
        Evaluate cross-encoder performance
        
        Args:
            queries: List of queries
            documents_sets: List of document sets for each query
            reference_sets: List of reference document sets for each query
            
        Returns:
            Performance metrics
        """
        if not queries or not documents_sets or not reference_sets:
            return {"error": "Missing required evaluation data"}
        
        total_ndcg = 0.0
        total_map = 0.0
        total_precision = 0.0
        valid_evaluations = 0
        
        for query, docs, refs in zip(queries, documents_sets, reference_sets):
            try:
                # Rerank documents
                reranked_docs = self.rerank_documents(query, docs, top_k=len(docs))
                
                # Calculate NDCG@5
                ndcg = self._calculate_ndcg(reranked_docs, refs, k=5)
                
                # Calculate MAP
                map_score = self._calculate_map(reranked_docs, refs)
                
                # Calculate Precision@5
                precision = self._calculate_precision_at_k(reranked_docs, refs, k=5)
                
                total_ndcg += ndcg
                total_map += map_score
                total_precision += precision
                valid_evaluations += 1
                
            except Exception as e:
                print(f"[WARNING] Failed to evaluate query '{query[:50]}...': {e}")
                continue
        
        if valid_evaluations == 0:
            return {"error": "No valid evaluations completed"}
        
        avg_ndcg = total_ndcg / valid_evaluations
        avg_map = total_map / valid_evaluations
        avg_precision = total_precision / valid_evaluations
        
        return {
            "total_evaluations": len(queries),
            "valid_evaluations": valid_evaluations,
            "avg_ndcg@5": avg_ndcg,
            "avg_map": avg_map,
            "avg_precision@5": avg_precision,
            "performance_score": (avg_ndcg + avg_map + avg_precision) / 3
        }
    
    def _calculate_ndcg(self, ranked_docs: List[Document], 
                       relevant_docs: List[Document], k: int = 5) -> float:
        """Calculate NDCG@k"""
        if not ranked_docs or not relevant_docs or k <= 0:
            return 0.0
        
        # Create relevance scores (binary for simplicity)
        relevance_scores = []
        relevant_sources = {doc.metadata.get('source', '') for doc in relevant_docs}
        
        for doc in ranked_docs[:k]:
            source = doc.metadata.get('source', '')
            relevance_scores.append(1.0 if source in relevant_sources else 0.0)
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg += rel / np.log2(i + 2)  # log2(1+position)
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_scores):
            idcg += rel / np.log2(i + 2)
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_map(self, ranked_docs: List[Document], 
                      relevant_docs: List[Document]) -> float:
        """Calculate Mean Average Precision"""
        if not ranked_docs or not relevant_docs:
            return 0.0
        
        relevant_sources = {doc.metadata.get('source', '') for doc in relevant_docs}
        num_relevant = len(relevant_sources)
        
        if num_relevant == 0:
            return 0.0
        
        ap = 0.0
        retrieved_relevant = 0
        
        for i, doc in enumerate(ranked_docs):
            source = doc.metadata.get('source', '')
            if source in relevant_sources:
                retrieved_relevant += 1
                precision_at_i = retrieved_relevant / (i + 1)
                ap += precision_at_i
        
        return ap / num_relevant
    
    def _calculate_precision_at_k(self, ranked_docs: List[Document], 
                                relevant_docs: List[Document], k: int = 5) -> float:
        """Calculate Precision@k"""
        if not ranked_docs or not relevant_docs or k <= 0:
            return 0.0
        
        relevant_sources = {doc.metadata.get('source', '') for doc in relevant_docs}
        retrieved_relevant = 0
        
        for doc in ranked_docs[:k]:
            source = doc.metadata.get('source', '')
            if source in relevant_sources:
                retrieved_relevant += 1
        
        return retrieved_relevant / k if k > 0 else 0.0

def main():
    """Main function to demonstrate cross-encoder optimization"""
    print("=" * 80)
    print("CROSS-ENCODER OPTIMIZATION DEMO")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = CrossEncoderOptimizer()
    
    # Example documents
    from langchain.schema import Document
    
    documents = [
        Document(
            page_content="Bangladesh Bank is the central bank that regulates monetary policy and oversees the banking system in Bangladesh.",
            metadata={"source": "central_bank_info.pdf", "page": 1}
        ),
        Document(
            page_content="To open a savings account in Bangladesh, you need to provide your National ID and make an initial deposit at any bank branch.",
            metadata={"source": "account_opening_guide.pdf", "page": 2}
        ),
        Document(
            page_content="Home loan interest rates in Bangladesh typically range from 8% to 12% depending on the bank and loan amount.",
            metadata={"source": "loan_rates_2024.pdf", "page": 3}
        ),
        Document(
            page_content="Tax filing deadlines for individuals in Bangladesh are typically June 30th following the end of the fiscal year.",
            metadata={"source": "tax_guidelines.pdf", "page": 4}
        ),
        Document(
            page_content="Investment options in Bangladesh include government bonds, mutual funds, and stock market participation through Dhaka Stock Exchange.",
            metadata={"source": "investment_guide.pdf", "page": 5}
        )
    ]
    
    query = "What are the requirements for opening a savings account in Bangladesh?"
    
    print(f"Query: {query}")
    print(f"Documents to rerank: {len(documents)}")
    print("-" * 80)
    
    # Optimize document ranking
    optimized_docs = optimizer.optimize_document_ranking(query, documents, threshold=0.1)
    
    print("Optimized Document Ranking:")
    for i, doc in enumerate(optimized_docs):
        score = doc.metadata.get('cross_encoder_score', 'N/A')
        quality = doc.metadata.get('ranking_quality', 'unknown')
        source = doc.metadata.get('source', 'Unknown')
        content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"  {i+1}. [{score:.3f}] [{quality}] {source}")
        print(f"      {content_preview}")
        print()
    
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()