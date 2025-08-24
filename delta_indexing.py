"""
Delta Indexing System for Tracking Document Changes
"""
import hashlib
import json
import os
from typing import Dict, Set, Tuple
from langchain.schema import Document

class DocumentVersionTracker:
    """Tracks document versions to enable delta indexing"""
    
    def __init__(self, index_path: str):
        """
        Initialize version tracker
        
        Args:
            index_path: Path to the FAISS index directory
        """
        self.index_path = index_path
        self.version_file = os.path.join(index_path, "document_versions.json")
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, str]:
        """Load document versions from file"""
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[WARNING] Could not load version file: {e}")
                return {}
        return {}
    
    def _save_versions(self):
        """Save document versions to file"""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except IOError as e:
            print(f"[ERROR] Could not save version file: {e}")
    
    def get_document_hash(self, content: str) -> str:
        """
        Generate MD5 hash for document content
        
        Args:
            content: Document content string
            
        Returns:
            MD5 hash of the content
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_changed_documents(self, documents: list) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Identify changed, new, and deleted documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of (new_docs, changed_docs, deleted_docs) sets
        """
        current_doc_ids = {self._get_doc_id(doc) for doc in documents}
        existing_doc_ids = set(self.versions.keys())
        
        # Identify new documents
        new_docs = current_doc_ids - existing_doc_ids
        
        # Identify deleted documents
        deleted_docs = existing_doc_ids - current_doc_ids
        
        # Identify changed documents
        changed_docs = set()
        for doc in documents:
            doc_id = self._get_doc_id(doc)
            if doc_id in existing_doc_ids:
                content_hash = self.get_document_hash(doc.page_content)
                if content_hash != self.versions.get(doc_id):
                    changed_docs.add(doc_id)
        
        return new_docs, changed_docs, deleted_docs
    
    def update_versions(self, documents: list):
        """
        Update version tracking for documents
        
        Args:
            documents: List of Document objects
        """
        for doc in documents:
            doc_id = self._get_doc_id(doc)
            self.versions[doc_id] = self.get_document_hash(doc.page_content)
        self._save_versions()
    
    def remove_documents(self, doc_ids: Set[str]):
        """
        Remove documents from version tracking
        
        Args:
            doc_ids: Set of document IDs to remove
        """
        for doc_id in doc_ids:
            if doc_id in self.versions:
                del self.versions[doc_id]
        self._save_versions()
    
    def _get_doc_id(self, doc: Document) -> str:
        """
        Generate unique document ID
        
        Args:
            doc: Document object
            
        Returns:
            Unique document identifier
        """
        # Create unique ID based on source metadata and content position
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 0)
        # Use first 100 chars of content as additional identifier
        content_preview = doc.page_content[:100] if doc.page_content else ""
        return f"{source}_{page}_{hash(content_preview)}"
    
    def get_document_count(self) -> int:
        """Get total number of tracked documents"""
        return len(self.versions)
    
    def clear_versions(self):
        """Clear all version tracking (use with caution)"""
        self.versions = {}
        if os.path.exists(self.version_file):
            os.remove(self.version_file)