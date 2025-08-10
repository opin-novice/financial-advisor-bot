#!/usr/bin/env python3
"""
Re-index documents with multilingual embedding model for improved cross-language similarity
"""

import os
import shutil
from config import config

def backup_old_index():
    """Backup the existing FAISS index"""
    if os.path.exists(config.FAISS_INDEX_PATH):
        backup_path = f"{config.FAISS_INDEX_PATH}_backup_old_model"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.move(config.FAISS_INDEX_PATH, backup_path)
        print(f"[INFO] ✅ Existing index backed up to: {backup_path}")
    else:
        print("[INFO] ⚠️ No existing index found to backup")

def reindex_documents():
    """Re-index documents using multilingual embedding model"""
    print("[INFO] 🔄 Starting re-indexing with multilingual embedding model...")
    print(f"[INFO] 📊 New embedding model: {config.EMBEDDING_MODEL}")
    
    # Import docadd to trigger re-indexing
    import docadd
    
    print("[INFO] ✅ Re-indexing completed successfully!")
    print("[INFO] 🌐 Documents now indexed with multilingual embeddings")

def main():
    """Main function to orchestrate the re-indexing process"""
    print("=" * 60)
    print("🔧 MULTILINGUAL EMBEDDING MODEL UPGRADE")
    print("=" * 60)
    
    print(f"[INFO] Current embedding model: {config.EMBEDDING_MODEL}")
    
    if "multilingual" not in config.EMBEDDING_MODEL.lower():
        print("[ERROR] ❌ Embedding model is not multilingual!")
        print("[ERROR] Please update config.py with multilingual model first.")
        return
    
    # Step 1: Backup existing index
    backup_old_index()
    
    # Step 2: Re-index with new model
    reindex_documents()
    
    print("\n" + "=" * 60)
    print("✅ MULTILINGUAL UPGRADE COMPLETED")
    print("=" * 60)
    print("[INFO] Your system now supports better cross-language similarity!")
    print("[INFO] Bengali queries will now better match English documents and vice versa.")
    print("\n[INFO] Next steps:")
    print("1. Run test_multilingual_accuracy.py to verify improvements")
    print("2. Test with actual Bengali and English queries")
    
if __name__ == "__main__":
    main()
