import nltk

def download_nltk_data():
    """Download required NLTK data for sentence tokenization"""
    print("[INFO] Downloading NLTK data...")
    
    try:
        # Download punkt tokenizer for sentence splitting
        nltk.download('punkt', quiet=True)
        print("[INFO] ✅ NLTK punkt tokenizer downloaded successfully")
        
        # Download averaged perceptron tagger (optional, for better sentence detection)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("[INFO] ✅ NLTK averaged perceptron tagger downloaded successfully")
        
        # Download maxent ne chunker (optional, for named entity recognition)
        nltk.download('maxent_ne_chunker', quiet=True)
        print("[INFO] ✅ NLTK maxent ne chunker downloaded successfully")
        
        # Download words corpus (optional, for better text processing)
        nltk.download('words', quiet=True)
        print("[INFO] ✅ NLTK words corpus downloaded successfully")
        
        print("[DONE] ✅ All required NLTK data downloaded successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to download NLTK data: {e}")
        print("[INFO] You may need to run this script with administrator privileges")

if __name__ == "__main__":
    download_nltk_data() 