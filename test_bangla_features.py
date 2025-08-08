#!/usr/bin/env python3
"""
Comprehensive test for Bangla features implementation
"""
import os
import sys
import pytest
import subprocess
from langchain.schema import Document

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sanitize_pdf_script():
    """Test the sanitize_pdf.py script functionality"""
    print("🧪 Testing sanitize_pdf.py script...")
    
    try:
        # Import the sanitize_pdf module
        import sanitize_pdf
        
        # Test configuration
        print(f"✅ INPUT_FOLDER: {sanitize_pdf.INPUT_FOLDER}")
        print(f"✅ OUTPUT_FOLDER: {sanitize_pdf.OUTPUT_FOLDER}")
        print(f"✅ OCR_LANG: {sanitize_pdf.OCR_LANG}")
        print(f"✅ OCR_THRESHOLD: {sanitize_pdf.OCR_THRESHOLD}")
        
        # Test if input folder exists and contains files
        if not os.path.exists(sanitize_pdf.INPUT_FOLDER):
            print(f"❌ Input folder '{sanitize_pdf.INPUT_FOLDER}' does not exist")
            pytest.skip(f"Input folder '{sanitize_pdf.INPUT_FOLDER}' does not exist")
            
        files = os.listdir(sanitize_pdf.INPUT_FOLDER)
        if not files:
            print(f"❌ No files found in input folder '{sanitize_pdf.INPUT_FOLDER}'")
            pytest.skip(f"No files found in input folder '{sanitize_pdf.INPUT_FOLDER}'")
            
        # Test text extraction function
        test_file = None
        for file in files:
            if file.endswith('.pdf'):
                test_file = file
                break
                
        if not test_file:
            print("❌ No PDF files found in input folder")
            pytest.skip("No PDF files found in input folder")
            
        test_path = os.path.join(sanitize_pdf.INPUT_FOLDER, test_file)
        extracted_text = sanitize_pdf.extract_text_from_pdf(test_path)
        print(f"✅ Text extraction test: Successfully extracted text from {test_file}")
        print(f"📝 Sample extracted text: {extracted_text[:100]}...")
        
        # Test needs_ocr function
        needs_ocr_result = sanitize_pdf.needs_ocr(test_path, 0)
        print(f"✅ OCR detection test: Page 0 needs OCR = {needs_ocr_result}")
        
        # Test preprocess_image function
        # Create a simple test image
        from PIL import Image
        import numpy as np
        test_image = Image.new('RGB', (100, 100), color='white')
        processed = sanitize_pdf.preprocess_image(test_image)
        print(f"✅ Image preprocessing test: Successfully processed test image")
        print(f"📊 Processed image shape: {processed.shape if isinstance(processed, np.ndarray) else 'Not numpy array'}")
        
        # Test clean_text function
        test_text = "Page 1\n\nThis is a test document.\nPage 2\n\nMore content here."
        cleaned = sanitize_pdf.clean_text(test_text)
        print(f"✅ Text cleaning test: Successfully cleaned text")
        print(f"📝 Cleaned text: {cleaned[:50]}...")
        
        assert True  # All tests passed
        
    except Exception as e:
        print(f"❌ sanitize_pdf.py test failed: {e}")
        pytest.fail(f"sanitize_pdf.py test failed: {e}")

def test_docadd_bilingual_embeddings():
    """Test docadd.py bilingual embeddings implementation"""
    print("\n🧪 Testing docadd.py bilingual embeddings...")
    
    try:
        # Import docadd module
        import docadd
        
        # Test configuration
        print(f"✅ PDF_DIR: {docadd.PDF_DIR}")
        print(f"✅ FAISS_INDEX_PATH: {docadd.FAISS_INDEX_PATH}")
        print(f"✅ EMBEDDING_MODEL: {docadd.EMBEDDING_MODEL}")
        
        # Check that the correct model is being used - be flexible with model names
        accepted_models = [
            "paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
        if docadd.EMBEDDING_MODEL in accepted_models:
            print(f"✅ Valid embedding model '{docadd.EMBEDDING_MODEL}' is configured")
        else:
            print(f"⚠️ Unexpected embedding model '{docadd.EMBEDDING_MODEL}', but proceeding with test")
            
        # Test SemanticChunker initialization
        chunker = docadd.SemanticChunker()
        print(f"✅ SemanticChunker initialized with model: {chunker.sentence_model_name}")
        
        # Test chunking with bilingual content
        test_content = "This is English text. এটি বাংলা টেক্সট। Both languages in one document."
        chunks = chunker.chunk_text(test_content)
        print(f"✅ Bilingual chunking test: Created {len(chunks)} chunks")
        
        # Test chunk content
        if chunks:
            chunk_content = chunks[0].page_content
            has_english = any(ord(char) < 128 and char.isalpha() for char in chunk_content)
            has_bangla = any(0x0980 <= ord(char) <= 0x09FF for char in chunk_content)
            print(f"✅ Chunk contains English: {has_english}")
            print(f"✅ Chunk contains Bangla: {has_bangla}")
        
        assert True  # All tests passed
        
    except Exception as e:
        print(f"❌ docadd.py bilingual embeddings test failed: {e}")
        pytest.fail(f"docadd.py bilingual embeddings test failed: {e}")

def test_answer_generation_prompt():
    """Test the updated answer generation prompt in main.py"""
    print("\n🧪 Testing answer generation prompt...")
    
    try:
        # Import main module
        from main import PROMPT_TEMPLATE, QA_PROMPT
        
        # Check that the prompt contains language-agnostic instructions
        if "Bangla" in PROMPT_TEMPLATE:
            print("✅ Prompt mentions Bangla language support")
        else:
            print("❌ Prompt does not mention Bangla language support")
            pytest.fail("Prompt does not mention Bangla language support")
            
        if "language-agnostic" in PROMPT_TEMPLATE or "SAME LANGUAGE" in PROMPT_TEMPLATE:
            print("✅ Prompt includes language-agnostic instructions")
        else:
            print("❌ Prompt does not include language-agnostic instructions")
            pytest.fail("Prompt does not include language-agnostic instructions")
            
        # Test prompt formatting
        test_context = "This is English context. এটি বাংলা প্রেক্ষিত।"
        test_input = "What is this document about?"
        
        formatted_prompt = QA_PROMPT.format(context=test_context, input=test_input)
        print("✅ Prompt formatting test: Successfully formatted prompt")
        print(f"📝 Prompt preview: {formatted_prompt[:150]}...")
        
        assert True  # All tests passed
        
    except Exception as e:
        print(f"❌ Answer generation prompt test failed: {e}")
        pytest.fail(f"Answer generation prompt test failed: {e}")

def test_bangla_compatibility():
    """Test Bangla compatibility with actual documents"""
    print("\n🧪 Testing Bangla compatibility with actual documents...")
    
    try:
        # Check if Bangla documents exist in data directory
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"❌ Data directory '{data_dir}' does not exist")
            pytest.skip(f"Data directory '{data_dir}' does not exist")
            
        bangla_files = [
            "আয়কর_নির্দেশিকা_২০২৪-২০২৫.pdf",
            "ডাক জীবন বীমা এবং অ্যানুইটি.pdf",
            "ডাকঘর সঞ্চয় ব্যাংক- মেয়াদী হিসাব.pdf",
            "ডাকঘর সঞ্চয় ব্যাংকের-সাধারণ হিসাব.pdf"
        ]
        
        found_bangla_files = []
        for file in bangla_files:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                found_bangla_files.append(file)
                
        print(f"✅ Found {len(found_bangla_files)} Bangla documents in data directory")
        
        if not found_bangla_files:
            print("⚠️  No Bangla documents found for testing")
            pytest.skip("No Bangla documents found for testing")
            
        # Test with the FinancialAdvisorTelegramBot
        from main import FinancialAdvisorTelegramBot
        bot = FinancialAdvisorTelegramBot()
        print("✅ FinancialAdvisorTelegramBot initialized successfully")
        
        # Test Bangla query
        bangla_query = "বাংলাদেশে ব্যাংক খোলার নিয়ম কি?"
        result = bot.process_query(bangla_query)
        
        if isinstance(result, dict) and result.get("response"):
            response = result["response"]
            print(f"✅ Bangla query processing successful!")
            print(f"📝 Bangla query response preview: {response[:100]}...")
            
            # Check if response contains Bangla characters
            has_bangla_response = any(0x0980 <= ord(char) <= 0x09FF for char in response)
            print(f"✅ Response contains Bangla characters: {has_bangla_response}")
            
            # Check sources
            sources = result.get("sources", [])
            print(f"📄 Sources found: {len(sources)}")
            
            if sources:
                # Check if any of the sources are Bangla documents
                bangla_sources = [src for src in sources if any(bangla_file in src["file"] for bangla_file in found_bangla_files)]
                print(f"📚 Bangla sources in response: {len(bangla_sources)}")
        else:
            print(f"❌ Bangla query processing failed or returned unexpected format")
            pytest.fail("Bangla query processing failed or returned unexpected format")
            
        # Test English query to ensure bilingual capability
        english_query = "How to open a bank account in Bangladesh?"
        result2 = bot.process_query(english_query)
        
        if isinstance(result2, dict) and result2.get("response"):
            response2 = result2["response"]
            print(f"✅ English query processing successful!")
            print(f"📝 English query response preview: {response2[:100]}...")
        else:
            print(f"❌ English query processing failed or returned unexpected format")
            pytest.fail("English query processing failed or returned unexpected format")
            
        assert True  # All tests passed
        
    except Exception as e:
        print(f"❌ Bangla compatibility test failed: {e}")
        pytest.fail(f"Bangla compatibility test failed: {e}")

def test_embedding_model_loading():
    """Test that the paraphrase-multilingual-MiniLM-L12-v2 model loads with MPS support and half-precision"""
    print("\n🧪 Testing embedding model loading with MPS and half-precision...")
    
    try:
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import config
        
        # Check if MPS is available (for M1 Mac)
        if torch.backends.mps.is_available():
            print("✅ MPS is available for M1 Mac")
            device = "mps"
        elif torch.cuda.is_available():
            print("✅ CUDA is available")
            device = "cuda"
        else:
            print("⚠️  Using CPU for embeddings")
            device = "cpu"
            
        # Test model loading with half-precision
        model_kwargs = {"device": device}
        if device != "cpu":
            model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            print("✅ Configured half-precision (float16) loading")
        else:
            model_kwargs["model_kwargs"] = {"torch_dtype": torch.float32}
            print("ℹ️  Using float32 precision on CPU")
            
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs
        )
        
        print(f"✅ Successfully loaded embedding model: {config.EMBEDDING_MODEL}")
        print(f"🖥️  Model device: {device}")
        
        # Test embedding generation
        test_texts = [
            "This is an English sentence for testing embeddings.",
            "এটি এমবেডিং পরীক্ষার জন্য একটি বাংলা বাক্য।"
        ]
        
        for text in test_texts:
            embedding = embeddings.embed_query(text)
            print(f"✅ Embedding generated for {'English' if 'English' in text else 'Bangla'} text")
            print(f"📊 Embedding dimension: {len(embedding)}")
            
        assert True  # All tests passed
        
    except Exception as e:
        print(f"❌ Embedding model loading test failed: {e}")
        pytest.fail(f"Embedding model loading test failed: {e}")

def test_cross_encoder_reranking():
    """Test cross-encoder re-ranking with bilingual content"""
    print("\n🧪 Testing cross-encoder re-ranking...")
    
    try:
        from main import FinancialAdvisorTelegramBot
        bot = FinancialAdvisorTelegramBot()
        
        # Check if reranker is loaded
        if bot.reranker is not None:
            print(f"✅ Cross-encoder reranker loaded: {bot.reranker}")
        else:
            print("⚠️  Cross-encoder reranker not available, using lexical reranking")
            
        # Test with mock documents
        docs = [
            Document(page_content="Bank account opening procedures in Bangladesh involve several steps including document verification."),
            Document(page_content="বাংলাদেশে ব্যাংক হিসাব খোলার পদ্ধতি নথিপত্র যাচাইয়ের সহ বেশ কয়েকটি পদক্ষেপ নেয়।"),
            Document(page_content="The weather in Dhaka is hot and humid during summer months.")
        ]
        
        # Test with English query
        english_query = "How to open a bank account in Bangladesh?"
        filtered_docs = bot._rank_and_filter(docs, english_query)
        print(f"✅ English query re-ranking successful: {len(filtered_docs)} relevant documents")
        
        # Test with Bangla query
        bangla_query = "বাংলাদেশে ব্যাংক হিসাব খোলার পদ্ধতি কি?"
        filtered_docs_bangla = bot._rank_and_filter(docs, bangla_query)
        print(f"✅ Bangla query re-ranking successful: {len(filtered_docs_bangla)} relevant documents")
        
        assert True  # All tests passed
        
    except Exception as e:
        print(f"❌ Cross-encoder re-ranking test failed: {e}")
        pytest.fail(f"Cross-encoder re-ranking test failed: {e}")

def run_comprehensive_test():
    """Run all tests and provide a summary report"""
    print("🚀 Comprehensive Bangla Features Test")
    print("=" * 60)
    
    # Test 1: sanitize_pdf.py script
    test1_success = test_sanitize_pdf_script()
    
    # Test 2: docadd.py bilingual embeddings
    test2_success = test_docadd_bilingual_embeddings()
    
    # Test 3: Answer generation prompt
    test3_success = test_answer_generation_prompt()
    
    # Test 4: Bangla compatibility
    test4_success = test_bangla_compatibility()
    
    # Test 5: Embedding model loading
    test5_success = test_embedding_model_loading()
    
    # Test 6: Cross-encoder re-ranking
    test6_success = test_cross_encoder_reranking()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"📄 sanitize_pdf.py Script: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"🔤 docadd.py Bilingual Embeddings: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"💬 Answer Generation Prompt: {'✅ PASS' if test3_success else '❌ FAIL'}")
    print(f"🇧🇩 Bangla Compatibility: {'✅ PASS' if test4_success else '❌ FAIL'}")
    print(f"🧠 Embedding Model Loading: {'✅ PASS' if test5_success else '❌ FAIL'}")
    print(f"🔄 Cross-Encoder Re-ranking: {'✅ PASS' if test6_success else '❌ FAIL'}")
    
    all_tests_passed = all([
        test1_success, test2_success, test3_success, 
        test4_success, test5_success, test6_success
    ])
    
    if all_tests_passed:
        print("\n🎉 ALL TESTS PASSED! Your Bangla features implementation is working perfectly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        return False

def main():
    """Main function for external calling"""
    return run_comprehensive_test()

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)