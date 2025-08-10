#!/bin/bash

# 🤖 Advanced Multilingual RAG System - Installation Script
# This script will set up your RAG system automatically

echo "🚀 Setting up Advanced Multilingual RAG System..."
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
echo "✓ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Detect system architecture
if [[ $(uname -m) == "arm64" ]] && [[ $(uname) == "Darwin" ]]; then
    echo "🍎 Detected M1/M2 Mac - using optimized requirements"
    pip install -r requirements_optimized.txt
    python setup_m1_optimized.py
else
    echo "💻 Using standard requirements"
    pip install -r requirements.txt
fi

# Set up environment file
if [ ! -f .env ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo ""
    echo "🔑 IMPORTANT: Please edit .env file with your API keys:"
    echo "   - GROQ_API_KEY=your_groq_api_key_here"
    echo "   - TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here"
    echo ""
fi

# Download NLTK data
echo "📚 Setting up language models..."
python setup_nltk.py

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p faiss_index

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Add your documents: python docadd.py"
echo "3. Start the bot: python main.py"
echo ""
echo "📖 For detailed instructions, see README.md"
