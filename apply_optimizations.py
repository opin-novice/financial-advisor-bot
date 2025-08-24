#!/usr/bin/env python3
"""
Script to apply cosine similarity optimizations to configuration files
"""

import os
import sys
import json
from pathlib import Path

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def update_config_py():
    """Update config.py with optimized settings"""
    try:
        config_file = "config.py"
        if not os.path.exists(config_file):
            print("Error: config.py not found")
            return False
            
        # Read current config
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update embedding model
        content = content.replace(
            'EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"',
            'EMBEDDING_MODEL = "BAAI/bge-m3"'
        )
        
        # Update cross-encoder model
        content = content.replace(
            'CROSS_ENCODER_MODEL = \'cross-encoder/ms-marco-MiniLM-L-12-v2\'',
            'CROSS_ENCODER_MODEL = \'BAAI/bge-reranker-large\''
        )
        
        # Update retrieval settings
        content = content.replace(
            'MAX_DOCS_FOR_RETRIEVAL = 12',
            'MAX_DOCS_FOR_RETRIEVAL = 20'
        )
        
        content = content.replace(
            'RELEVANCE_THRESHOLD = 0.4 # increase from 0.2 to improve precision',
            'RELEVANCE_THRESHOLD = 0.2 # lowered for more inclusive results'
        )
        
        # Add hybrid search configuration
        if 'ENABLE_HYBRID_SEARCH' not in content:
            hybrid_config = '''
        # Hybrid Search Configuration
        self.ENABLE_HYBRID_SEARCH = True
        self.HYBRID_SEMANTIC_WEIGHT = 0.7
        self.HYBRID_KEYWORD_WEIGHT = 0.3
'''
            # Insert before the closing of the config class
            content = content.replace(
                '# Advanced RAG Feedback Loop Configuration',
                f'{hybrid_config}\n        # Advanced RAG Feedback Loop Configuration'
            )
        
        # Write updated config
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("[SUCCESS] config.py updated successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Error updating config.py: {e}")
        return False

def update_docadd_py():
    """Update docadd.py with optimized chunking settings"""
    try:
        docadd_file = "docadd.py"
        if not os.path.exists(docadd_file):
            print("Error: docadd.py not found")
            return False
            
        # Read current docadd.py
        with open(docadd_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update chunking settings
        content = content.replace(
            'MAX_CHUNK_SIZE = 1500          # Increased from 1200 for more comprehensive chunks',
            'MAX_CHUNK_SIZE = 1000          # Reduced for more focused chunks'
        )
        
        content = content.replace(
            'MIN_CHUNK_SIZE = 200           # Increased from 200 to avoid very small chunks',
            'MIN_CHUNK_SIZE = 400           # Increased to avoid tiny chunks'
        )
        
        content = content.replace(
            'SIMILARITY_THRESHOLD = 0.8     # Increased from 0.7 for more semantically coherent chunks',
            'SIMILARITY_THRESHOLD = 0.75     # Adjusted for better balance'
        )
        
        # Add overlap setting if not present
        if 'OVERLAP_SENTENCES' not in content:
            content = content.replace(
                'OVERLAP_SENTENCES = 1          # Number of sentences to overlap between chunks',
                'OVERLAP_SENTENCES = 2          # Increased overlap for better context continuity'
            )
        
        # Write updated docadd.py
        with open(docadd_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("[SUCCESS] docadd.py updated successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Error updating docadd.py: {e}")
        return False

def update_main_py():
    """Update main.py with hybrid search implementation"""
    try:
        main_file = "main.py"
        if not os.path.exists(main_file):
            print("Error: main.py not found")
            return False
            
        # Read current main.py
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add hybrid search initialization if not present
        if 'self.enable_hybrid_retrieval()' not in content:
            # Find where hybrid retrieval is initialized
            hybrid_init = '''
        # Initialize Hybrid Retrieval (enabled by default for better performance)
        self.use_hybrid_retrieval = True  # Enabled hybrid retrieval for better similarity scores
        print(f"[INFO] Hybrid retrieval status: {'ENABLED' if self.use_hybrid_retrieval else 'DISABLED'}")
        print("[INFO] To disable hybrid retrieval, set self.use_hybrid_retrieval = False")
'''
            content = content.replace(
                '# Initialize Hybrid Retrieval (disabled by default)',
                '# Initialize Hybrid Retrieval (enabled by default for better performance)\n        self.use_hybrid_retrieval = True  # Enabled hybrid retrieval for better similarity scores'
            )
        
        # Write updated main.py
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("[SUCCESS] main.py updated successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Error updating main.py: {e}")
        return False

def create_requirements_update():
    """Create a script to update requirements with new models"""
    try:
        requirements_update = '''# Updated requirements with better embedding models
# Install these with: pip install -r requirements_update.txt

# Better embedding model
BAAI/bge-m3

# Better cross-encoder
BAAI/bge-reranker-large

# For hybrid search
rank-bm25
'''
        
        with open('requirements_update.txt', 'w', encoding='utf-8') as f:
            f.write(requirements_update)
        
        print("[SUCCESS] requirements_update.txt created successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Error creating requirements_update.txt: {e}")
        return False

def main():
    """Main function to apply all optimizations"""
    print("Applying Cosine Similarity Optimizations")
    print("=" * 50)
    
    # Apply all optimizations
    optimizations = [
        ("Updating config.py", update_config_py),
        ("Updating docadd.py", update_docadd_py),
        ("Updating main.py", update_main_py),
        ("Creating requirements update", create_requirements_update)
    ]
    
    results = []
    for name, func in optimizations:
        print(f"\n{name}...")
        success = func()
        results.append((name, success))
    
    # Summary
    print("\nUpdate Summary:")
    print("-" * 30)
    for name, success in results:
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"{name}: {status}")
    
    print("\nNext Steps:")
    print("-" * 30)
    print("1. Install new requirements:")
    print("   pip install -r requirements_update.txt")
    print("\n2. Rebuild your FAISS index with the new settings:")
    print("   python docadd.py")
    print("\n3. Test the improvements:")
    print("   python analyze_actual_faiss.py")
    print("\n4. Run sample queries to verify better similarity scores")

if __name__ == "__main__":
    main()