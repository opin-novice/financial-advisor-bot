#!/usr/bin/env python3
"""
Setup script for multilingual financial advisor bot
Downloads required models and data for Bangla-English support
"""

import nltk
import os
import subprocess
import sys
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

def download_nltk_data():
    """Download required NLTK data"""
    print("[INFO] Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("[INFO] ✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"[WARNING] NLTK download failed: {e}")

def download_multilingual_models():
    """Download and cache multilingual models"""
    print("[INFO] Downloading multilingual models...")
    
    models_to_download = [
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/distiluse-base-multilingual-cased",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ]
    
    for model_name in models_to_download:
        try:
            print(f"[INFO] Downloading {model_name}...")
            if "cross-encoder" in model_name:
                # For cross-encoder models
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(model_name)
            else:
                # For sentence transformer models
                model = SentenceTransformer(model_name)
            print(f"[INFO] ✅ {model_name} downloaded and cached")
        except Exception as e:
            print(f"[WARNING] Failed to download {model_name}: {e}")

def install_ollama_models():
    """Install required Ollama models"""
    print("[INFO] Installing Ollama models...")
    
    models_to_install = [
        "gemma3n:e2b",
        # "aya:8b",  # Multilingual model (if available)
    ]
    
    for model in models_to_install:
        try:
            print(f"[INFO] Installing Ollama model: {model}")
            result = subprocess.run(
                ["ollama", "pull", model], 
                capture_output=True, 
                text=True, 
                timeout=1800  # 30 minutes timeout
            )
            if result.returncode == 0:
                print(f"[INFO] ✅ {model} installed successfully")
            else:
                print(f"[WARNING] Failed to install {model}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"[WARNING] Timeout installing {model}")
        except FileNotFoundError:
            print("[WARNING] Ollama not found. Please install Ollama first:")
            print("  Visit: https://ollama.ai/")
        except Exception as e:
            print(f"[WARNING] Error installing {model}: {e}")

def create_directories():
    """Create necessary directories"""
    print("[INFO] Creating directories...")
    
    directories = [
        "logs",
        "faiss_index_multilingual",
        "data",
        "unsant_data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] ✅ Directory created: {directory}")

def check_dependencies():
    """Check if all dependencies are installed"""
    print("[INFO] Checking dependencies...")
    
    required_packages = [
        "torch",
        "sentence_transformers", 
        "langchain",
        "langdetect",
        "faiss",
        "telegram",
        "transformers",
        "nltk"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"[INFO] ✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"[WARNING] ❌ {package} is missing")
    
    if missing_packages:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print("pip install -r requirements_multilingual.txt")
        return False
    
    return True

def test_language_detection():
    """Test language detection functionality"""
    print("[INFO] Testing language detection...")
    
    try:
        from langdetect import detect
        
        # Test English
        english_text = "This is a test sentence in English about banking and finance."
        english_result = detect(english_text)
        print(f"[INFO] English detection: {english_result}")
        
        # Test Bangla
        bangla_text = "এটি ব্যাংকিং এবং আর্থিক বিষয়ে একটি বাংলা পরীক্ষার বাক্য।"
        bangla_result = detect(bangla_text)
        print(f"[INFO] Bangla detection: {bangla_result}")
        
        print("[INFO] ✅ Language detection working")
        return True
        
    except Exception as e:
        print(f"[WARNING] Language detection test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Multilingual Financial Advisor Bot")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Download NLTK data
    download_nltk_data()
    
    # Test language detection
    test_language_detection()
    
    # Download multilingual models
    download_multilingual_models()
    
    # Install Ollama models
    install_ollama_models()
    
    print("\n" + "=" * 50)
    print("✅ Multilingual setup completed!")
    print("\nNext steps:")
    print("1. Place your Bangla PDFs in the 'unsant_data' folder")
    print("2. Place your English PDFs in the 'data' folder")
    print("3. Run: python multilingual_semantic_chunking.py")
    print("4. Run: python multilingual_main.py")
    print("\nFor Telegram bot:")
    print("Set your TELEGRAM_TOKEN environment variable")
    print("export TELEGRAM_TOKEN='your_bot_token_here'")

if __name__ == "__main__":
    main()
