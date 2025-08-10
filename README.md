# 🤖 Advanced Multilingual RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system with intelligent feedback loops, multilingual support, and Telegram bot integration. This system can understand and respond in both English and Bangla, making it perfect for financial document Q&A and customer support.

## 🌟 Key Features

### 🧠 Advanced RAG with Feedback Loop
- **Intelligent Query Refinement**: Automatically improves queries if initial results aren't relevant
- **Iterative Retrieval**: Keeps refining until finding the most relevant documents
- **Multiple Refinement Strategies**: Domain expansion, synonym matching, context addition, and query decomposition
- **Relevance Scoring**: Uses cross-encoder models to ensure high-quality results

### 🌍 Multilingual Support
- **Automatic Language Detection**: Detects English and Bangla automatically
- **Bilingual Responses**: Responds in the same language as the user's query
- **Mixed Language Handling**: Smart processing of queries containing both languages
- **Language-Aware Caching**: Separate caching for different languages

### 📱 Telegram Bot Integration
- **Real-time Chat Interface**: Interactive Telegram bot for easy access
- **Document Source Citations**: Shows which documents were used for answers
- **Processing Status Updates**: Real-time feedback during query processing
- **Error Handling**: Graceful error messages in appropriate language

### ⚡ Performance Optimizations
- **M1 Mac Optimized**: Special optimizations for Apple Silicon
- **GPU/CPU Support**: Flexible deployment options
- **Semantic Chunking**: Intelligent document splitting for better retrieval
- **Vector Database**: FAISS-powered fast similarity search
- **Caching System**: Reduces response time for repeated queries

## 🏗️ Architecture

```
User Query → Language Detection → RAG Feedback Loop → Document Retrieval → 
Cross-Encoder Ranking → Response Generation → Bilingual Formatting → User
```

### Core Components

1. **Language Detection Engine** (`language_utils.py`)
   - Detects query language using script analysis and word recognition
   - Formats responses in matching language

2. **Advanced RAG Feedback Loop** (`advanced_rag_feedback.py`)
   - Iteratively refines queries for better results
   - Multiple refinement strategies
   - Configurable performance modes

3. **Main RAG System** (`main.py`)
   - Orchestrates the entire pipeline
   - Telegram bot integration
   - Caching and performance monitoring

4. **Document Processing** (`docadd.py`)
   - PDF processing and text extraction
   - Semantic chunking for optimal retrieval
   - Vector embedding generation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GROQ API key (for LLM)
- Telegram Bot Token (for bot functionality)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd final_rag
   ```

2. **Install dependencies**
   ```bash
   # For regular systems
   pip install -r requirements.txt
   
   # For M1 Mac (optimized)
   pip install -r requirements_optimized.txt
   python setup_m1_optimized.py
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Download language models**
   ```bash
   python setup_nltk.py
   ```

5. **Add your documents**
   ```bash
   python docadd.py
   ```

6. **Start the system**
   ```bash
   # For Telegram bot
   python main.py
   
   # For language detection demo
   python demo_language_detection.py
   ```

### Environment Variables

Create a `.env` file with:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Optional - Advanced RAG Feedback Loop
ENABLE_FEEDBACK_LOOP=true
FEEDBACK_MAX_ITERATIONS=3
FEEDBACK_RELEVANCE_THRESHOLD=0.3
FEEDBACK_CONFIDENCE_THRESHOLD=0.2

# Optional - Performance
GROQ_MODEL=llama3-8b-8192
MAX_DOCS_FOR_RETRIEVAL=12
MAX_DOCS_FOR_CONTEXT=5
```

## 📖 Usage Examples

### Telegram Bot

1. Start a chat with your bot
2. Ask questions in English or Bangla:

**English:**
```
User: What are the requirements to open a savings account?
Bot: To open a savings account, you typically need...
```

**Bangla:**
```
User: সঞ্চয় হিসাব খুলতে কি কি লাগে?
Bot: সঞ্চয় হিসাব খুলতে সাধারণত প্রয়োজন...
```

### Python API

```python
from main import FinancialRAGBot
from language_utils import LanguageDetector

# Initialize the system
bot = FinancialRAGBot()
detector = LanguageDetector()

# Process a query
query = "What is the interest rate?"
language = detector.detect_language(query)
response = bot.process_query(query, language)
print(response)
```

## 🔧 Configuration

### Performance Modes

The system supports three performance modes:

- **Fast Mode**: 2 iterations, 0.4 relevance threshold
- **Balanced Mode**: 3 iterations, 0.3 relevance threshold (default)
- **Thorough Mode**: 4 iterations, 0.2 relevance threshold

### Customization

Edit `config.py` to customize:
- Model parameters
- Retrieval settings
- Language detection sensitivity
- Feedback loop behavior

## 📁 Project Structure

```
final_rag/
├── main.py                          # Main application entry point
├── main_with_language_detection.py  # Language-aware version
├── language_utils.py                # Language detection utilities
├── advanced_rag_feedback.py         # Feedback loop implementation
├── config.py                        # Configuration management
├── rag_utils.py                     # Core RAG utilities
├── docadd.py                        # Document processing
├── api_utils.py                     # API utilities
├── requirements.txt                 # Dependencies
├── requirements_optimized.txt       # M1 Mac optimized dependencies
├── setup_m1_optimized.py           # M1 Mac setup script
├── data/                           # Document storage
├── faiss_index/                    # Vector database
├── logs/                           # Application logs
└── tests/                          # Test files
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test language detection
python test_language_detection.py

# Test multilingual accuracy
python test_multilingual_accuracy.py

# Test RAG feedback loop
python test_advanced_rag_feedback.py

# Test embedding compatibility
python test_embedding_compatibility.py

# Run all tests
python -m pytest
```

## 📊 Performance Monitoring

The system includes built-in monitoring:

```bash
# Monitor performance metrics
python monitor_performance.py

# Check multilingual accuracy
python test_multilingual_accuracy.py

# Analyze embedding compatibility
python test_embedding_compatibility.py
```

## 🔍 Troubleshooting

### Common Issues

1. **GROQ API Rate Limits**
   - The system includes automatic rate limiting
   - Check `logs/` for detailed error messages

2. **Memory Issues on M1 Macs**
   - Use the optimized requirements: `requirements_optimized.txt`
   - Run the M1 setup script: `python setup_m1_optimized.py`

3. **Language Detection Issues**
   - Check if NLTK data is downloaded: `python setup_nltk.py`
   - Verify text encoding is UTF-8

4. **Document Processing Errors**
   - Ensure PDFs are not corrupted
   - Check if Tesseract OCR is installed for image-based PDFs

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain** for RAG framework
- **GROQ** for fast LLM inference
- **Sentence Transformers** for embeddings
- **FAISS** for vector search
- **Hugging Face** for model hosting

## 📞 Support

For support and questions:
- Check the documentation in the `docs/` folder
- Review test files for usage examples
- Open an issue on GitHub
- Check the logs in `logs/` folder for debugging

---

## 🎯 What Makes This Special?

This isn't just another RAG system. It's a **production-ready, multilingual, intelligent document Q&A system** that:

- **Thinks Before Answering**: Uses feedback loops to refine queries
- **Speaks Your Language**: Automatically detects and responds in English or Bangla
- **Gets Smarter**: Learns from relevance feedback to improve results
- **Scales Gracefully**: Optimized for both development and production
- **Fails Safely**: Comprehensive error handling and fallback mechanisms

Perfect for financial institutions, customer support, legal document analysis, and any scenario requiring intelligent, multilingual document search and Q&A.
