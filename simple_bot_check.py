#!/usr/bin/env python3
"""
Simple Telegram Bot Health Check
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

def print_status(message: str, status: str = "INFO"):
    """Print formatted status messages"""
    symbols = {"OK": "✅", "ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}
    print(f"{symbols.get(status, 'ℹ️')} {message}")

def main():
    print("=" * 50)
    print("🤖 TELEGRAM BOT QUICK CHECK")
    print("=" * 50)
    
    issues = []
    
    # 1. Check Telegram Token
    tg_token = os.getenv('TG_TOKEN')
    if not tg_token or tg_token == 'YOUR_TOKEN_HERE':
        print_status("TG_TOKEN not set", "ERROR")
        issues.append("Set TG_TOKEN environment variable")
    else:
        print_status("TG_TOKEN is configured", "OK")
    
    # 2. Check FAISS index
    if os.path.exists('faiss_index_multilingual/index.faiss'):
        size = os.path.getsize('faiss_index_multilingual/index.faiss')
        print_status(f"FAISS index exists ({size:,} bytes)", "OK")
    else:
        print_status("FAISS index missing", "ERROR")
        issues.append("Run setup to create FAISS index")
    
    # 3. Check Ollama
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if 'gemma3n:e2b' in result.stdout:
            print_status("Ollama model available", "OK")
        else:
            print_status("Ollama model not found", "ERROR")
            issues.append("Install gemma3n:e2b model with: ollama pull gemma3n:e2b")
    except:
        print_status("Ollama not available", "ERROR")
        issues.append("Install and start Ollama service")
    
    # 4. Test basic imports
    try:
        import telegram
        import langchain
        import torch
        import sentence_transformers
        print_status("Core dependencies available", "OK")
    except ImportError as e:
        print_status(f"Missing dependency: {e}", "ERROR")
        issues.append("Install missing dependencies")
    
    # 5. Try to import main components
    try:
        # Test if main.py can be imported without errors
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        
        # This will test if all imports work
        print_status("Testing main.py imports...", "INFO")
        spec.loader.exec_module(main_module)
        print_status("Bot components loaded successfully", "OK")
        
    except Exception as e:
        print_status(f"Bot loading failed: {str(e)[:100]}...", "ERROR")
        issues.append("Fix bot component loading issues")
    
    print("\n" + "=" * 50)
    
    if not issues:
        print_status("🎉 ALL CHECKS PASSED!", "OK")
        print_status("Your bot should work. Start it with: python3 main.py", "INFO")
        
        if not tg_token or tg_token == 'YOUR_TOKEN_HERE':
            print_status("Don't forget to set your Telegram token!", "WARNING")
            print_status("export TG_TOKEN='your_bot_token_here'", "INFO")
    else:
        print_status("Issues found:", "ERROR")
        for i, issue in enumerate(issues, 1):
            print_status(f"{i}. {issue}", "WARNING")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
