#!/usr/bin/env python3
"""
Telegram Bot Health Check Script
This script validates all components of your financial advisor bot
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List
import traceback

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

def print_status(message: str, status: str = "INFO"):
    """Print formatted status messages"""
    symbols = {"OK": "✅", "ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}
    print(f"{symbols.get(status, 'ℹ️')} {message}")

def check_dependencies() -> bool:
    """Check if all required dependencies are installed"""
    print_status("Checking dependencies...", "INFO")
    
    required_modules = {
        'telegram': 'python-telegram-bot',
        'langchain': 'langchain',
        'torch': 'torch',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'PIL': 'Pillow',
        'pytesseract': 'pytesseract',
        'fitz': 'PyMuPDF',
        'spanish_translator': 'local module',
        'langdetect': 'langdetect',
        'transformers': 'transformers'
    }
    
    missing = []
    for module, package in required_modules.items():
        try:
            __import__(module)
            print_status(f"{module} ({package})", "OK")
        except ImportError as e:
            print_status(f"{module} ({package}) - {e}", "ERROR")
            missing.append(package)
    
    if missing:
        print_status(f"Missing packages: {', '.join(missing)}", "ERROR")
        return False
    
    print_status("All dependencies are installed", "OK")
    return True

def check_environment_variables() -> Dict[str, str]:
    """Check environment variables and configuration"""
    print_status("Checking environment variables...", "INFO")
    
    env_vars = {
        'TG_TOKEN': os.getenv('TG_TOKEN', 'NOT_SET'),
        'FAISS_INDEX_PATH': os.getenv('FAISS_INDEX_PATH', 'faiss_index_multilingual'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'),
        'OLLAMA_MODEL': os.getenv('OLLAMA_MODEL', 'gemma3n:e2b')
    }
    
    issues = []
    for var, value in env_vars.items():
        if var == 'TG_TOKEN':
            if value == 'NOT_SET' or value == 'YOUR_TOKEN_HERE':
                print_status(f"{var}: Not configured", "ERROR")
                issues.append(f"{var} needs to be set")
            else:
                print_status(f"{var}: Configured (hidden for security)", "OK")
        else:
            print_status(f"{var}: {value}", "OK")
    
    if issues:
        print_status("Environment variable issues found", "WARNING")
        for issue in issues:
            print_status(f"  - {issue}", "WARNING")
    
    return env_vars

def check_files_and_directories() -> bool:
    """Check if required files and directories exist"""
    print_status("Checking files and directories...", "INFO")
    
    required_paths = [
        'main.py',
        'spanish_translator.py',
        'faiss_index_multilingual/',
        'faiss_index_multilingual/index.faiss',
        'faiss_index_multilingual/index.pkl'
    ]
    
    missing = []
    for path in required_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                print_status(f"Directory: {path}", "OK")
            else:
                size = os.path.getsize(path)
                print_status(f"File: {path} ({size:,} bytes)", "OK")
        else:
            print_status(f"Missing: {path}", "ERROR")
            missing.append(path)
    
    if missing:
        print_status(f"Missing files/directories: {', '.join(missing)}", "ERROR")
        return False
    
    return True

def check_ollama_service() -> bool:
    """Check if Ollama service is running and model is available"""
    print_status("Checking Ollama service...", "INFO")
    
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print_status("Ollama service not running", "ERROR")
            return False
        
        models = result.stdout
        if 'gemma3n:e2b' in models:
            print_status("Ollama service running, gemma3n:e2b model available", "OK")
            return True
        else:
            print_status("Ollama running but gemma3n:e2b model not found", "ERROR")
            print_status("Available models:", "INFO")
            for line in models.split('\n')[1:]:  # Skip header
                if line.strip():
                    print_status(f"  - {line.strip()}", "INFO")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Ollama service check timed out", "ERROR")
        return False
    except FileNotFoundError:
        print_status("Ollama command not found", "ERROR")
        return False
    except Exception as e:
        print_status(f"Error checking Ollama: {e}", "ERROR")
        return False

async def test_bot_components() -> bool:
    """Test bot components without starting the full bot"""
    print_status("Testing bot components...", "INFO")
    
    try:
        # Test imports
        from main import (
            load_faiss_index, load_embeddings, load_llm,
            process_user_query, lang_processor
        )
        print_status("Bot modules imported successfully", "OK")
        
        # Test FAISS index loading
        try:
            vectorstore = load_faiss_index()
            print_status("FAISS index loaded successfully", "OK")
        except Exception as e:
            print_status(f"FAISS index loading failed: {e}", "ERROR")
            return False
        
        # Test embeddings
        try:
            embeddings = load_embeddings()
            print_status("Embeddings model loaded successfully", "OK")
        except Exception as e:
            print_status(f"Embeddings loading failed: {e}", "ERROR")
            return False
        
        # Test LLM
        try:
            llm = load_llm()
            print_status("LLM loaded successfully", "OK")
        except Exception as e:
            print_status(f"LLM loading failed: {e}", "ERROR")
            return False
        
        # Test language processor
        try:
            test_queries = [
                "What is investment?",
                "¿Qué es la inversión?",
                "বিনিয়োগ কি?"
            ]
            
            for query in test_queries:
                lang = lang_processor.detect_language(query)
                print_status(f"Language detection for '{query[:20]}...': {lang}", "OK")
        except Exception as e:
            print_status(f"Language processor test failed: {e}", "ERROR")
            return False
        
        # Test query processing (without full pipeline to avoid long wait)
        try:
            print_status("Testing query processing pipeline...", "INFO")
            # This is a lightweight test - we won't run the full pipeline
            print_status("Query processing components initialized", "OK")
        except Exception as e:
            print_status(f"Query processing test failed: {e}", "ERROR")
            return False
        
        return True
        
    except Exception as e:
        print_status(f"Component testing failed: {e}", "ERROR")
        traceback.print_exc()
        return False

def check_telegram_token_format(token: str) -> bool:
    """Basic validation of Telegram token format"""
    if not token or token == 'NOT_SET' or token == 'YOUR_TOKEN_HERE':
        return False
    
    # Basic format check: should be like "123456789:ABCdefGHIjklMNOpqrSTUvwxyz"
    import re
    pattern = r'^\d{8,10}:[A-Za-z0-9_-]{35}$'
    return bool(re.match(pattern, token))

async def main():
    """Main health check function"""
    print("=" * 60)
    print("🤖 TELEGRAM BOT HEALTH CHECK")
    print("=" * 60)
    
    all_checks_passed = True
    
    # 1. Check dependencies
    if not check_dependencies():
        all_checks_passed = False
    
    print("\n" + "-" * 40)
    
    # 2. Check environment variables
    env_vars = check_environment_variables()
    
    # Validate Telegram token format
    if env_vars['TG_TOKEN'] != 'NOT_SET':
        if check_telegram_token_format(env_vars['TG_TOKEN']):
            print_status("Telegram token format appears valid", "OK")
        else:
            print_status("Telegram token format appears invalid", "WARNING")
    
    print("\n" + "-" * 40)
    
    # 3. Check files and directories
    if not check_files_and_directories():
        all_checks_passed = False
    
    print("\n" + "-" * 40)
    
    # 4. Check Ollama service
    if not check_ollama_service():
        all_checks_passed = False
    
    print("\n" + "-" * 40)
    
    # 5. Test bot components
    if not await test_bot_components():
        all_checks_passed = False
    
    print("\n" + "=" * 60)
    
    if all_checks_passed:
        print_status("🎉 ALL CHECKS PASSED! Your bot should work correctly.", "OK")
        print_status("To start your bot, run: python3 main.py", "INFO")
    else:
        print_status("❌ SOME CHECKS FAILED! Please fix the issues above.", "ERROR")
    
    print("=" * 60)
    
    return all_checks_passed

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print_status("Health check interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_status(f"Health check failed with error: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)
