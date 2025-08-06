#!/bin/bash

# 🚀 Financial Advisor Bot - Quick Setup Script
# This script automates the setup process for your friend

set -e  # Exit on any error

echo "🤖 Financial Advisor Telegram Bot - Quick Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_info "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
        
        # Check if version is 3.8+
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python version is compatible"
        else
            print_error "Python 3.8+ required. Please upgrade Python."
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Check if Ollama is installed
check_ollama() {
    print_info "Checking Ollama installation..."
    if command -v ollama &> /dev/null; then
        print_status "Ollama found"
        
        # Check if Ollama service is running
        if pgrep -f "ollama" > /dev/null; then
            print_status "Ollama service is running"
        else
            print_warning "Ollama service not running. Starting it..."
            ollama serve &
            sleep 5
        fi
        
        # Check if model exists
        if ollama list | grep -q "gemma3n:e2b"; then
            print_status "Required model (gemma3n:e2b) found"
        else
            print_warning "Required model not found. This will be downloaded later."
        fi
    else
        print_error "Ollama not found. Please install Ollama first:"
        echo "  curl -fsSL https://ollama.ai/install.sh | sh"
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    print_info "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_status "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    print_status "Pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    print_warning "This may take several minutes..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "Dependencies installed successfully"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
}

# Setup environment file
setup_env() {
    print_info "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp .env.template .env
            print_status "Environment file (.env) created from template"
            print_warning "IMPORTANT: Edit .env file and add your Telegram bot token!"
            print_info "Replace 'YOUR_TELEGRAM_BOT_TOKEN_HERE' with your actual bot token"
        else
            cat > .env << EOF
# Telegram Bot Token (REQUIRED)
TG_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE

# Optional configurations (these have defaults)
FAISS_INDEX_PATH=faiss_index_multilingual
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
OLLAMA_MODEL=gemma3n:e2b
EOF
            print_status "Environment file (.env) created"
            print_warning "IMPORTANT: Edit .env file and add your Telegram bot token!"
        fi
    else
        print_status "Environment file already exists"
    fi
}

# Download Ollama model
download_model() {
    print_info "Downloading AI model (this may take 10-15 minutes)..."
    print_warning "Model size: ~5.6GB - ensure you have good internet connection"
    
    if ! ollama list | grep -q "gemma3n:e2b"; then
        ollama pull gemma3n:e2b
        print_status "Model downloaded successfully"
    else
        print_status "Model already available"
    fi
}

# Run health check
run_health_check() {
    print_info "Running system health check..."
    
    if [ -f "simple_bot_check.py" ]; then
        python3 simple_bot_check.py
    else
        print_warning "Health check script not found, skipping..."
    fi
}

# Main setup process
main() {
    echo
    print_info "Starting automated setup process..."
    echo
    
    # Step 1: Check Python
    check_python
    echo
    
    # Step 2: Check Ollama
    check_ollama
    echo
    
    # Step 3: Setup virtual environment
    setup_venv
    echo
    
    # Step 4: Install dependencies
    install_dependencies
    echo
    
    # Step 5: Setup environment
    setup_env
    echo
    
    # Step 6: Download model
    download_model
    echo
    
    # Step 7: Health check
    run_health_check
    echo
    
    # Final instructions
    echo "================================================"
    print_status "🎉 Setup completed successfully!"
    echo
    print_info "To start the bot:"
    echo "  1. Make sure Ollama is running: ollama serve"
    echo "  2. Activate virtual environment: source venv/bin/activate"
    echo "  3. Start the bot: python3 main.py"
    echo
    print_info "To test the bot, send /start to your Telegram bot"
    echo "================================================"
}

# Run main function
main
