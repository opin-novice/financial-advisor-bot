"""
Bridge module between Document Retriever and Incremental Embedding Pipeline
This module safely integrates the existing delta indexing with the new incremental embedding pipeline
"""
import os
import json
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
from langchain.schema import Document

# Import existing components
from document_retriever import DocumentRetriever
from delta_indexing import DocumentVersionTracker

# Import new incremental embedding pipeline
from incremental_embedding import IncrementalEmbeddingPipeline, compute_content_hash

class DocumentRetrieverWithIncrementalEmbedding:
    """Enhanced document retriever with incremental embedding capabilities"""
    
    def __init__(self, faiss_index_path: str = "faiss_index"):
        """
        Initialize enhanced document retriever
        
        Args:
            faiss_index_path: Path to existing FAISS index
        """
        # Initialize existing document retriever
        self.document_retriever = DocumentRetriever()
        
        # Initialize version tracker
        self.version_tracker = DocumentVersionTracker(faiss_index_path)
        
        # Initialize incremental embedding pipeline
        self.embedding_pipeline = IncrementalEmbeddingPipeline(
            index_path=f"{faiss_index_path}_incremental",
            metadata_path=f"{faiss_index_path}_incremental_metadata.json"
        )
        
        print("[INFO] [OK] Enhanced document retriever with incremental embedding initialized")
    
    def process_documents_incrementally(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process documents through incremental embedding pipeline
        
        Args:
            documents: List of Document objects to process
            
        Returns:
            Processing results with statistics
        """
        print(f"[INFO] Processing {len(documents)} documents through incremental embedding pipeline...")
        
        # Convert documents to content and metadata for the pipeline
        contents = []
        doc_metadata = []
        
        for doc in documents:
            contents.append(doc.page_content)
            metadata = {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "original_metadata": doc.metadata
            }
            doc_metadata.append(metadata)
        
        # Process through incremental embedding pipeline
        results = self.embedding_pipeline.process_documents(contents, doc_metadata)
        
        # Update version tracker with processed documents
        self.version_tracker.update_versions(documents)
        
        # Statistics
        processed_count = sum(1 for r in results if r.get("status") == "processed")
        skipped_count = len(results) - processed_count
        
        print(f"[INFO] [OK] Incremental processing completed:")
        print(f"       - Processed: {processed_count} documents")
        print(f"       - Skipped: {skipped_count} documents (duplicates)")
        
        return {
            "results": results,
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "total_count": len(results)
        }
    
    def get_embedding_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics from the incremental embedding pipeline"""
        return self.embedding_pipeline.get_pipeline_stats()
    
    def search_similar_documents(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents using the embedding pipeline
        
        Args:
            query: Query string
            k: Number of similar documents to return
            
        Returns:
            Search results
        """
        return self.embedding_pipeline.search_similar(query, k)
    
    def update_faiss_index_delta(self, documents: List[Document]) -> bool:
        """
        Update the existing FAISS index with delta changes (existing functionality)
        
        Args:
            documents: List of Document objects to check for changes
            
        Returns:
            True if changes were applied, False if no changes detected
        """
        print("[INFO] Updating existing FAISS index with delta changes...")
        return self.document_retriever.update_index_delta(documents)
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """
        Get combined statistics from both systems
        
        Returns:
            Combined statistics
        """
        # Get existing FAISS index stats
        existing_stats = self.document_retriever.get_index_statistics()
        
        # Get incremental embedding pipeline stats
        incremental_stats = self.embedding_pipeline.get_pipeline_stats()
        
        return {
            "existing_faiss": existing_stats,
            "incremental_embedding": incremental_stats,
            "version_tracked_documents": self.version_tracker.get_document_count()
        }

# Convenience functions for backward compatibility
def create_enhanced_retriever(faiss_index_path: str = "faiss_index") -> DocumentRetrieverWithIncrementalEmbedding:
    """
    Create an enhanced document retriever with incremental embedding capabilities
    
    Args:
        faiss_index_path: Path to existing FAISS index
        
    Returns:
        Enhanced document retriever instance
    """
    return DocumentRetrieverWithIncrementalEmbedding(faiss_index_path)

def process_documents_with_deduplication(documents: List[Document], 
                                       faiss_index_path: str = "faiss_index") -> Dict[str, Any]:
    """
    Process documents with both delta indexing and incremental embedding
    
    Args:
        documents: List of Document objects to process
        faiss_index_path: Path to existing FAISS index
        
    Returns:
        Processing results
    """
    enhanced_retriever = create_enhanced_retriever(faiss_index_path)
    
    # Process through incremental embedding
    embedding_results = enhanced_retriever.process_documents_incrementally(documents)
    
    # Update existing FAISS index with delta changes
    delta_updated = enhanced_retriever.update_faiss_index_delta(documents)
    
    return {
        "embedding_results": embedding_results,
        "delta_indexing_updated": delta_updated,
        "combined_stats": enhanced_retriever.get_combined_stats()
    }