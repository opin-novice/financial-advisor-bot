# 🚀 Financial Advisor Telegram Bot - Deployment Guide

This guide will help you set up and run the multilingual financial advisor Telegram bot on any system.

## 📋 System Requirements

### Minimum Hardware Requirements:
- **RAM**: 16GB+ recommended (12GB minimum)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor (4+ cores recommended)
- **Internet**: Stable connection for model downloads

### Supported Operating Systems:
- Linux (Ubuntu 20.04+, CentOS 8+)
- macOS (10.15+)
- Windows 10/11

## 🛠️ Prerequisites Installation

### 1. Python Installation
```bash
# Check Python version (3.8+ required)
python3 --version

# If Python not installed:
# Ubuntu/Debian:
sudo apt update && sudo apt install python3 python3-pip python3-venv

# macOS (with Homebrew):
brew install python3

# Windows: Download from python.org
```

### 2. Ollama Installation
```bash
# Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download

# Start Ollama service
ollama serve

# Pull the required model (this will take time - ~5.6GB)
ollama pull gemma3n:e2b
```

### 3. Tesseract OCR (for image processing)
```bash
# Ubuntu/Debian:
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa tesseract-ocr-ben

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## 📦 Project Setup

### 1. Clone/Download the Project
```bash
# If you have the project files, navigate to the directory
cd financial-advisor-bot

# Or download/extract the project files to a directory
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter issues, try upgrading pip first:
pip install --upgrade pip
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```bash
# Create .env file
touch .env
```

Add the following content to `.env`:
```env
# Telegram Bot Token (REQUIRED)
TG_TOKEN=7596897324:AAG3TsT18amwRF2nRBcr1JS6NdGs96Ie-D0

# Optional configurations (these have defaults)
FAISS_INDEX_PATH=faiss_index_multilingual
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
OLLAMA_MODEL=gemma3n:e2b
```

### 5. Download Required Models
The bot will automatically download models on first run, but you can pre-download them:
```bash
# Run the setup script to download models
python3 setup_multilingual.py
```

## 🔧 System-Specific Configurations

### For Linux Systems:
```bash
# Install additional dependencies if needed
sudo apt install build-essential

# For better performance, install BLAS libraries
sudo apt install libblas-dev liblapack-dev
```

### For macOS Systems:
```bash
# Install Xcode command line tools if needed
xcode-select --install

# Update Tesseract path in main.py if needed (usually not required on macOS)
```

### For Windows Systems:
1. Install Microsoft Visual C++ Redistributable
2. Update Tesseract path in `main.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

## 🚀 Running the Bot

### 1. Pre-flight Check
Run the health check script to ensure everything is configured correctly:
```bash
python3 simple_bot_check.py
```

### 2. Start the Bot
```bash
# Make sure Ollama is running in another terminal:
ollama serve

# Start the Telegram bot
python3 main.py
```

You should see:
```
✅ Environment variables loaded from .env file
INFO:__main__:Bot started. Press Ctrl+C to stop.
```

### 3. Test the Bot
1. Open Telegram
2. Search for your bot using the username
3. Send `/start` command
4. Try asking: "What is investment?" or "¿Qué es la inversión?"

## 🔍 Troubleshooting

### Common Issues and Solutions:

#### 1. "TG_TOKEN not set" Error
```bash
# Make sure .env file exists and contains the token
cat .env
# Should show: TG_TOKEN=7596897324:AAG3TsT18amwRF2nRBcr1JS6NdGs96Ie-D0
```

#### 2. "FAISS index missing" Error
```bash
# Run the setup script
python3 setup_multilingual.py
```

#### 3. "Ollama model not found" Error
```bash
# Check if Ollama is running
ollama list

# If model missing, pull it:
ollama pull gemma3n:e2b
```

#### 4. Memory Issues
```bash
# Monitor memory usage
htop  # Linux/macOS
# Task Manager on Windows

# If running out of memory, consider:
# - Closing other applications
# - Using a smaller model
# - Adding swap space (Linux)
```

#### 5. Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Or install specific missing packages:
pip install python-telegram-bot langchain torch
```

## 📊 Performance Optimization

### For Better Performance:
1. **Use GPU if available**:
   ```bash
   # Install GPU version of PyTorch and FAISS
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install faiss-gpu
   ```

2. **Increase system resources**:
   - Close unnecessary applications
   - Increase virtual memory/swap
   - Use SSD storage

3. **Monitor resource usage**:
   ```bash
   # Check system resources
   python3 -c "
   import psutil
   print(f'RAM: {psutil.virtual_memory().percent}%')
   print(f'CPU: {psutil.cpu_percent()}%')
   print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
   "
   ```

## 🔒 Security Notes

1. **Keep your bot token secure**:
   - Never commit `.env` file to version control
   - Use environment variables in production
   - Regenerate token if compromised

2. **System security**:
   - Keep system updated
   - Use firewall if needed
   - Monitor bot usage

## 📝 Maintenance

### Regular Tasks:
1. **Update dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Update Ollama models**:
   ```bash
   ollama pull gemma3n:e2b
   ```

3. **Monitor logs**:
   ```bash
   # Check bot logs
   tail -f logs/bot.log  # if logging to file
   ```

## 🆘 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Run the health check: `python3 simple_bot_check.py`
3. Check system resources and requirements
4. Verify all prerequisites are installed correctly

## 📋 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Ollama installed and running
- [ ] Tesseract OCR installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with bot token
- [ ] FAISS index exists (`faiss_index_multilingual/`)
- [ ] Ollama model downloaded (`ollama pull gemma3n:e2b`)
- [ ] Health check passed (`python3 simple_bot_check.py`)
- [ ] Bot started successfully (`python3 main.py`)

---

**Note**: The initial setup may take 30-60 minutes due to model downloads. Subsequent runs will be much faster.
