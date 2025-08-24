#!/usr/bin/env python3
"""
Script to pull the required Ollama model
"""

import subprocess
import sys
import os

def pull_ollama_model():
    """Pull the required Ollama model"""
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    print(f"Pulling Ollama model: {model}")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        # Run the ollama pull command
        result = subprocess.run(["ollama", "pull", model], 
                              capture_output=True, text=True, check=True)
        
        print("[SUCCESS] Model pulled successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to pull model: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[ERROR] Ollama not found. Please install Ollama first from https://ollama.com/")
        return False

if __name__ == "__main__":
    pull_ollama_model()