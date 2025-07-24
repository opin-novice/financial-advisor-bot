# Financial Advisor Telegram Bot - Setup & Usage Guide

## 🎯 Overview

This guide will help you set up and run the Financial Advisor Telegram Bot that provides intelligent responses to financial questions using advanced RAG (Retrieval-Augmented Generation) technology.

## ✅ Prerequisites

Before starting, ensure you have:

1. **Python 3.8+** installed
2. **Ollama** running with the required models
3. **Telegram Bot Token** from @BotFather
4. **FAISS Index** built from your financial documents

## 🔧 Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create/update your `.env` file:

```bash
TELEGRAM_TOKEN=your_telegram_bot_token_here
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
OLLAMA_MODEL=mistral:7b-instruct-v0.2-q4_K_M
FAISS_INDEX_PATH=faiss_index
```

### 3. Verify Ollama Models

Ensure you have the required Ollama model:

```bash
ollama list
# Should show: mistral:7b-instruct-v0.2-q4_K_M
```

If not installed:
```bash
ollama pull mistral:7b-instruct-v0.2-q4_K_M
```

### 4. Test the Setup

Run the test script to verify everything is working:

```bash
python3 test_telegram_bot.py
```

## 🚀 Running the Bot

### Option 1: Using the Startup Script (Recommended)

```bash
python3 start_telegram_bot.py
```

This script will:
- Check all prerequisites
- Provide helpful error messages
- Start the bot with proper error handling

### Option 2: Direct Launch

```bash
python3 telegram_bot_main.py
```

## 📱 Using the Bot

### Commands

- `/start` - Welcome message and introduction
- `/help` - Display available commands and usage information

### Query Examples

The bot can handle various financial queries:

1. **Banking Questions:**
   - "How do I open a bank account?"
   - "What documents are needed for account opening?"
   - "What are the banking fees?"

2. **Loan Information:**
   - "What is the maximum car loan amount?"
   - "What documents are required for a personal loan?"
   - "What are the current interest rates?"

3. **Investment Guidance:**
   - "What are the benefits of Bangladesh Sanchayapatra?"
   - "What investment options are available?"
   - "How do I invest in government bonds?"

4. **Tax Information:**
   - "What are the current tax rates in Bangladesh?"
   - "How do I calculate income tax?"
   - "What are the tax exemption limits?"

## 🔧 Configuration

### Main Settings (main.py)

```python
# Performance optimization
MAX_DOCS_FOR_RETRIEVAL = 8  # Number of documents to retrieve
MAX_DOCS_FOR_CONTEXT = 3    # Number of documents to use for context
CONTEXT_CHUNK_SIZE = 1500   # Maximum content size per document

# Cache settings
CACHE_TTL = 86400  # 24 hours cache time
```

### Telegram Settings (telegram_bot_main.py)

- Message length automatically handled (4096 char Telegram limit)
- Typing indicators for better user experience
- Session management for conversation context
- Proper error handling and user feedback

## 🎯 Features Implemented

### ✅ Message Handling
- Asynchronous message processing
- Typing indicators during processing
- Error handling with user-friendly messages
- Automatic message splitting for long responses

### ✅ User Session Management
- Conversation history tracking per user
- Context preservation across messages
- Session-based response optimization

### ✅ Performance Optimizations
- Response caching for faster replies
- Document retrieval optimization
- Performance monitoring and statistics
- Query categorization for better responses

### ✅ Comprehensive Testing
- Unit tests for core functionality
- Integration tests for Telegram API
- Performance benchmarks and monitoring
- Automated test suite validation

## 📊 Monitoring & Logs

### Log Files

- `logs/financial_advisor_bot.log` - Main application logs
- Performance metrics and error tracking
- User interaction logging (privacy-compliant)

### Performance Statistics

Query the bot with 'stats' in the CLI version to see:
- Total queries processed
- Average response times
- Cache hit rates
- Category-wise performance breakdown

## 🛠️ Troubleshooting

### Common Issues

1. **Bot not responding:**
   ```bash
   # Check if Ollama is running
   curl http://127.0.0.1:11434/api/tags
   
   # Verify bot token
   python3 test_telegram_bot.py
   ```

2. **"Model not found" error:**
   ```bash
   # Install the required model
   ollama pull mistral:7b-instruct-v0.2-q4_K_M
   ```

3. **FAISS index not found:**
   ```bash
   # Check if index exists
   ls -la faiss_index/
   
   # Rebuild index if needed
   python3 scripts/build_faiss_index.py
   ```

4. **Import errors:**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

### Debug Mode

For detailed debugging, modify the logging level in `main.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 🔒 Security & Privacy

- Bot token stored securely in environment variables
- No user data stored permanently
- Session data cleared regularly
- All financial advice includes appropriate disclaimers

## 📈 Performance Metrics

- **Average Response Time:** ~5-15 seconds (depending on query complexity)
- **Cache Hit Rate:** ~30-40% for common queries
- **Document Retrieval:** Optimized to top 8 most relevant documents
- **Context Processing:** Limited to 3 documents for faster LLM processing

## 🎉 Success Verification

Your bot is working correctly if:

1. ✅ `python3 test_telegram_bot.py` passes all tests
2. ✅ Bot responds to `/start` command on Telegram
3. ✅ Bot provides accurate answers to financial questions
4. ✅ Responses include proper source citations and disclaimers
5. ✅ Performance is within acceptable limits (< 30 seconds per query)

## 💡 Tips for Best Performance

1. **Keep Ollama running** in the background
2. **Use specific questions** for better responses
3. **Allow 10-30 seconds** for complex queries
4. **Check logs** if issues occur
5. **Restart bot** if memory usage gets high

---

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs in `logs/financial_advisor_bot.log`
3. Run `python3 test_telegram_bot.py` for diagnostics
4. Ensure all prerequisites are met

Your Financial Advisor Bot is now ready to help users with their financial questions! 🎉
