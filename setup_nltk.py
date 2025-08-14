import nltk
import ssl

def download_nltk_data():
    """
    Download required NLTK data for semantic chunking
    """
    print("[INFO] Setting up NLTK data for semantic chunking...")
    
    # Handle SSL certificate issues (common on Windows)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # List of required NLTK packages
    required_packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'averaged_perceptron_tagger',
        'wordnet'
    ]
    
    for package in required_packages:
        try:
            print(f"[INFO] Checking for NLTK package: {package}")
            nltk.data.find(f'tokenizers/{package}')
            print(f"[INFO] ✅ {package} already exists")
        except LookupError:
            try:
                print(f"[INFO] Downloading NLTK package: {package}")
                nltk.download(package, quiet=False)
                print(f"[INFO] ✅ Successfully downloaded {package}")
            except Exception as e:
                print(f"[WARNING] Failed to download {package}: {e}")
                # Try alternative approach
                try:
                    nltk.download(package, download_dir=nltk.data.path[0])
                    print(f"[INFO] ✅ Successfully downloaded {package} to custom path")
                except Exception as e2:
                    print(f"[ERROR] Could not download {package}: {e2}")
    
    # Verify installations
    print("\n[INFO] Verifying NLTK installations...")
    verification_tests = {
        'punkt': lambda: nltk.sent_tokenize("This is a test. This is another sentence."),
        'punkt_tab': lambda: nltk.word_tokenize("This is a test"),
        'stopwords': lambda: nltk.corpus.stopwords.words('english')[:5]
    }
    
    for package, test_func in verification_tests.items():
        try:
            result = test_func()
            print(f"[INFO] ✅ {package} working correctly")
        except Exception as e:
            print(f"[WARNING] {package} verification failed: {e}")
    
    print("\n[INFO] ✅ NLTK setup completed!")
    print("[INFO] You can now run docadd.py for semantic chunking")

if __name__ == "__main__":
    download_nltk_data() 