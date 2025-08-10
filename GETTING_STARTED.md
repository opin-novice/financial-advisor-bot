# 🚀 Getting Started Guide

Welcome to the Advanced Multilingual RAG System! This guide will help you get up and running in just a few minutes.

## 📋 Prerequisites Checklist

Before you start, make sure you have:

- [ ] **Python 3.8 or higher** installed
- [ ] **Git** installed
- [ ] **GROQ API key** (free at [console.groq.com](https://console.groq.com))
- [ ] **Telegram Bot Token** (optional, for bot functionality)

## 🎯 Quick Setup (5 minutes)

### Option 1: Automatic Installation (Recommended)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd final_rag

# 2. Run the installation script
./install.sh

# 3. Edit your API keys
nano .env  # or use any text editor

# 4. Add your documents
python docadd.py

# 5. Start the system
python main.py
```

### Option 2: Manual Installation

```bash
# 1. Clone and navigate
git clone <your-repo-url>
cd final_rag

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 5. Download language models
python setup_nltk.py

# 6. Process documents
python docadd.py

# 7. Start the bot
python main.py
```

## 🔑 Getting Your API Keys

### GROQ API Key (Required)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

### Telegram Bot Token (Optional)

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow the instructions to create your bot
4. Copy the bot token to your `.env` file

## 📄 Adding Your Documents

The system can process various document types:

### Supported Formats
- PDF files
- Text files
- Word documents (with additional setup)

### Adding Documents

1. **Place your documents** in the `data/` folder
2. **Run the document processor**:
   ```bash
   python docadd.py
   ```
3. **Wait for processing** - this creates the vector database

### Example Document Structure
```
data/
├── banking_policies.pdf
├── loan_guidelines.pdf
├── customer_handbook.pdf
└── faq_document.txt
```

## 🤖 Using the System

### Telegram Bot Usage

1. **Start your bot**:
   ```bash
   python main.py
   ```

2. **Find your bot** on Telegram using the username you created

3. **Start chatting**:
   ```
   /start
   ```

4. **Ask questions** in English or Bangla:
   ```
   English: "What are the loan requirements?"
   Bangla: "ঋণের শর্তাবলী কি?"
   ```

### Python API Usage

```python
from main import FinancialRAGBot
from language_utils import LanguageDetector

# Initialize
bot = FinancialRAGBot()
detector = LanguageDetector()

# Ask a question
query = "What is the interest rate?"
language = detector.detect_language(query)
response = bot.process_query(query, language)
print(response)
```

## ⚙️ Configuration Options

### Basic Configuration (`.env` file)

```env
# Required
GROQ_API_KEY=your_groq_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Optional - Performance tuning
GROQ_MODEL=llama3-8b-8192
MAX_DOCS_FOR_RETRIEVAL=12
MAX_DOCS_FOR_CONTEXT=5

# Optional - Advanced RAG Feedback Loop
ENABLE_FEEDBACK_LOOP=true
FEEDBACK_MAX_ITERATIONS=3
FEEDBACK_RELEVANCE_THRESHOLD=0.3
```

### Advanced Configuration (`config.py`)

You can modify `config.py` for:
- Embedding model selection
- Retrieval parameters
- Language detection settings
- Performance optimization

## 🧪 Testing Your Setup

### Quick Test

```bash
# Test language detection
python demo_language_detection.py

# Test the RAG system
python test_language_detection.py

# Test multilingual accuracy
python test_multilingual_accuracy.py
```

### Example Test Queries

Try these queries to test your system:

**English:**
- "What are the account opening requirements?"
- "How do I apply for a loan?"
- "What is the minimum balance?"

**Bangla:**
- "হিসাব খোলার শর্ত কি?"
- "ঋণের জন্য কিভাবে আবেদন করব?"
- "সর্বনিম্ন ব্যালেন্স কত?"

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. "GROQ API Error"
- Check your API key in `.env`
- Verify you have credits in your GROQ account
- Check internet connection

#### 3. "No documents found"
```bash
# Make sure documents are in data/ folder
ls data/
# Re-run document processing
python docadd.py
```

#### 4. "Language detection not working"
```bash
# Download NLTK data
python setup_nltk.py
```

#### 5. "Memory issues on M1 Mac"
```bash
# Use optimized requirements
pip install -r requirements_optimized.txt
python setup_m1_optimized.py
```

### Getting Help

1. **Check logs**: Look in `logs/` folder for error details
2. **Enable debug mode**: Set `logging.basicConfig(level=logging.DEBUG)`
3. **Run tests**: Use the test files to identify issues
4. **Check documentation**: Review the detailed guides in the project

## 🎯 What's Next?

Once your system is running:

1. **Add more documents** to improve knowledge base
2. **Customize responses** by modifying prompts
3. **Monitor performance** using built-in monitoring tools
4. **Scale up** for production use
5. **Contribute** improvements back to the project

## 📚 Additional Resources

- **README.md** - Complete project overview
- **LANGUAGE_DETECTION_GUIDE.md** - Language detection details
- **ADVANCED_RAG_FEEDBACK_LOOP_GUIDE.md** - Feedback loop documentation
- **Test files** - Examples of system usage
- **Config files** - Customization options

## 🆘 Need Help?

If you're stuck:

1. Check the troubleshooting section above
2. Look at the test files for examples
3. Review the logs in `logs/` folder
4. Open an issue on GitHub with:
   - Your operating system
   - Python version
   - Error messages
   - Steps you've tried

---

**🎉 Congratulations! You now have a powerful multilingual RAG system running!**

The system will automatically detect whether users ask questions in English or Bangla and respond in the same language, making it perfect for multilingual customer support and document Q&A.
