#!/usr/bin/env python3
"""
Test script for multilingual delta indexing system
"""

import os
import sys
import shutil
from pathlib import Path
import json
import time

def test_delta_indexing():
    """Test the delta indexing functionality"""
    print("🧪 Testing Multilingual Delta Indexing System")
    print("=" * 50)
    
    # Check if required directories exist
    data_dir = Path("data")
    unsant_data_dir = Path("unsant_data")
    
    if not data_dir.exists() or not unsant_data_dir.exists():
        print("❌ Required directories (data/ or unsant_data/) not found")
        print("Please ensure you have PDF files in these directories")
        return False
    
    # Count PDF files
    english_pdfs = list(data_dir.glob("*.pdf"))
    bangla_pdfs = list(unsant_data_dir.glob("*.pdf"))
    
    print(f"📄 Found {len(english_pdfs)} English PDFs in data/")
    print(f"📄 Found {len(bangla_pdfs)} Bangla PDFs in unsant_data/")
    
    if len(english_pdfs) == 0 and len(bangla_pdfs) == 0:
        print("❌ No PDF files found in either directory")
        return False
    
    # Test 1: Check if the delta index script can be imported
    print("\n🔍 Test 1: Importing delta index module...")
    try:
        from multilingual_delta_index import multilingual_delta_update, verify_index
        print("✅ Successfully imported delta index module")
    except ImportError as e:
        print(f"❌ Failed to import delta index module: {e}")
        return False
    
    # Test 2: Check dependencies
    print("\n🔍 Test 2: Checking dependencies...")
    try:
        import torch
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from sentence_transformers import SentenceTransformer
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    # Test 3: Test with a small subset (if we have many files)
    print("\n🔍 Test 3: Testing delta update with small subset...")
    
    # Clean up any existing test index
    test_index_dir = Path("faiss_index_multilingual")
    if test_index_dir.exists():
        print("🧹 Cleaning up existing index for fresh test...")
        shutil.rmtree(test_index_dir)
    
    # Clean up state file for fresh test
    state_file = Path(".multilingual_delta_state.json")
    if state_file.exists():
        state_file.unlink()
        print("🧹 Cleaned up existing state file")
    
    try:
        # Run delta update (should be full build since no existing index)
        print("🔄 Running initial delta update...")
        multilingual_delta_update(force_full=False)
        
        # Check if index was created
        if test_index_dir.exists():
            print("✅ Index directory created successfully")
        else:
            print("❌ Index directory was not created")
            return False
        
        # Test 4: Verify the index works
        print("\n🔍 Test 4: Verifying index functionality...")
        if verify_index():
            print("✅ Index verification passed")
        else:
            print("❌ Index verification failed")
            return False
        
        # Test 5: Test incremental update (should find no changes)
        print("\n🔍 Test 5: Testing incremental update...")
        multilingual_delta_update(force_full=False)
        print("✅ Incremental update completed")
        
        # Test 6: Check state file
        print("\n🔍 Test 6: Checking state file...")
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            print(f"✅ State file contains {len(state)} entries")
        else:
            print("❌ State file was not created")
            return False
        
        print("\n🎉 All tests passed! Delta indexing system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_command_line_interface():
    """Test the command line interface"""
    print("\n🔍 Testing Command Line Interface...")
    
    import subprocess
    
    # Test help command
    try:
        result = subprocess.run([
            sys.executable, "multilingual_delta_index.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Help command works")
        else:
            print("❌ Help command failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Help command timed out")
    except Exception as e:
        print(f"❌ Error testing CLI: {e}")
        return False
    
    return True

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    files_to_clean = [
        "faiss_index_multilingual",
        "faiss_index_multilingual_tmp", 
        "faiss_index_multilingual_backup",
        ".multilingual_delta_state.json"
    ]
    
    for item in files_to_clean:
        path = Path(item)
        try:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"🗑️ Removed directory: {item}")
            elif path.is_file():
                path.unlink()
                print(f"🗑️ Removed file: {item}")
        except Exception as e:
            print(f"⚠️ Could not remove {item}: {e}")

def main():
    """Main test function"""
    print("🚀 Multilingual Delta Index Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("multilingual_delta_index.py").exists():
        print("❌ Please run this test from the project root directory")
        print("   (where multilingual_delta_index.py is located)")
        return
    
    try:
        # Run tests
        success = test_delta_indexing()
        
        if success:
            # Test CLI if main test passed
            test_command_line_interface()
            
            print("\n" + "=" * 60)
            print("✅ DELTA INDEXING SYSTEM IS WORKING!")
            print("=" * 60)
            print("\nYou can now use the delta indexing system:")
            print("• python multilingual_delta_index.py           # Incremental update")
            print("• python multilingual_delta_index.py --full    # Full rebuild")
            print("• python multilingual_delta_index.py --verify  # Verify index")
            print("• python multilingual_delta_index.py --help    # Show all options")
        else:
            print("\n" + "=" * 60)
            print("❌ DELTA INDEXING SYSTEM HAS ISSUES")
            print("=" * 60)
            print("\nPlease check the error messages above and:")
            print("1. Ensure all dependencies are installed")
            print("2. Make sure you have PDF files in data/ and/or unsant_data/")
            print("3. Check that you have sufficient disk space")
            print("4. Verify that Ollama models are available if needed")
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    
    finally:
        # Ask if user wants to clean up
        try:
            response = input("\n🧹 Clean up test files? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                cleanup_test_files()
                print("✅ Cleanup completed")
        except KeyboardInterrupt:
            print("\n🛑 Cleanup skipped")

if __name__ == "__main__":
    main()
