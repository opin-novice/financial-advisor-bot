# 🤖 Financial Advisor Telegram Bot - Quick Start

Hey! This is your multilingual financial advisor Telegram bot. It can answer questions in **English**, **Spanish**, and **Bengali**.

## 🚀 Super Quick Setup (5 minutes)

### Option 1: Automated Setup (Recommended)

**For Linux/macOS:**
```bash
./quick_setup.sh
```

**For Windows:**
```batch
quick_setup.bat
```

### Option 2: Manual Setup

1. **Install Prerequisites:**
   - Python 3.8+ 
   - Ollama: https://ollama.ai/download

2. **Setup Project:**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # OR
   venv\Scripts\activate     # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Download AI model (5.6GB - takes time!)
   ollama pull gemma3n:e2b
   ```

3. **Start the Bot:**
   ```bash
   # Terminal 1: Start Ollama
   ollama serve
   
   # Terminal 2: Start Bot
   python3 main.py
   ```

## 💬 Testing Your Bot

1. Open Telegram
2. Search for your bot (use the username from BotFather)
3. Send: `/start`
4. Try these questions:
   - "What is investment?"
   - "¿Qué es la inversión?"
   - "বিনিয়োগ কি?"

## 🔧 System Requirements

- **RAM**: 16GB+ (12GB minimum)
- **Storage**: 10GB free space
- **Internet**: For initial model download

## 🆘 If Something Goes Wrong

1. **Run health check:**
   ```bash
   python3 simple_bot_check.py
   ```

2. **Common fixes:**
   - Make sure Ollama is running: `ollama serve`
   - Check if model exists: `ollama list`
   - Restart the bot: `Ctrl+C` then `python3 main.py`

## 📱 Bot Features

- ✅ Multilingual support (English, Spanish, Bengali)
- ✅ PDF document analysis
- ✅ Image text extraction (OCR)
- ✅ Financial advice and information
- ✅ Context-aware responses

## 🔒 Security Note

Your bot token is already configured in the `.env` file. Keep this file secure and don't share it publicly!

---

**Need help?** Check the detailed `DEPLOYMENT_GUIDE.md` for troubleshooting and advanced setup options.
