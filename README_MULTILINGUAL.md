# 🌐 Multilingual Financial Advisor Bot (Bangla + English)

A sophisticated Telegram bot that provides financial advice in both **Bangla** and **English**, specifically designed for Bangladesh's banking and financial services sector.

## 🚀 Features

### Core Capabilities
- **Bilingual Support**: Understands and responds in both Bangla (বাংলা) and English
- **Automatic Language Detection**: Detects user query language and responds accordingly
- **Multilingual Document Processing**: Processes PDFs in both languages
- **Advanced RAG System**: Retrieval-Augmented Generation with multilingual embeddings
- **Smart Re-ranking**: Hybrid semantic and lexical document ranking
- **Cross-lingual Retrieval**: Can find relevant information across languages

### Technical Features
- **Semantic Chunking**: Intelligent document segmentation for both languages
- **Form Field Filtering**: Removes irrelevant form templates from responses
- **Response Caching**: 24-hour cache for improved performance
- **Comprehensive Logging**: Detailed multilingual activity logs
- **Evaluation Framework**: Built-in testing for both languages

## 📁 Project Structure

```
financial-advisor-bot/
├── multilingual_main.py              # Main multilingual bot
├── multilingual_semantic_chunking.py # Document processing
├── multilingual_eval.py              # Evaluation system
├── setup_multilingual.py             # Setup script
├── requirements_multilingual.txt     # Dependencies
├── data/                             # English PDFs
├── unsant_data/                      # Bangla PDFs
├── faiss_index_multilingual/         # Vector database
├── logs/                             # Log files
└── dataqa/                           # Evaluation datasets
```

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements_multilingual.txt

# Install Ollama (for LLM)
# Visit: https://ollama.ai/
```

### 2. Run Setup Script

```bash
python setup_multilingual.py
```

This will:
- Download multilingual models
- Install Ollama models
- Create necessary directories
- Test language detection

### 3. Prepare Documents

```bash
# Place English PDFs in data/ folder
cp your_english_pdfs/* data/

# Place Bangla PDFs in unsant_data/ folder
cp your_bangla_pdfs/* unsant_data/
```

### 4. Process Documents

```bash
# Create multilingual vector index
python multilingual_semantic_chunking.py
```

### 5. Set Telegram Token

```bash
# Set your bot token
export TELEGRAM_TOKEN="your_telegram_bot_token"
```

### 6. Run the Bot

```bash
python multilingual_main.py
```

## 🔧 Configuration

### Key Settings in `multilingual_main.py`:

```python
# Embedding Model (supports 50+ languages)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Cross-encoder for multilingual re-ranking
CROSS_ENCODER_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

# LLM Model
OLLAMA_MODEL = "llama3.2:3b"

# Retrieval Settings
MAX_DOCS_FOR_RETRIEVAL = 15
MAX_DOCS_FOR_CONTEXT = 6
CONTEXT_CHUNK_SIZE = 1800
```

### Language Detection:

The bot uses multiple methods for language detection:
1. **langdetect** library for primary detection
2. **Unicode range analysis** for Bangla characters (`\u0980-\u09FF`)
3. **Keyword-based fallback** using financial terms in both languages

## 📊 Usage Examples

### English Queries:
```
User: "What documents are required to open a bank account?"
Bot: "To open a bank account in Bangladesh, you typically need..."
```

### Bangla Queries:
```
User: "ব্যাংক অ্যাকাউন্ট খোলার জন্য কী কী কাগজপত্র প্রয়োজন?"
Bot: "বাংলাদেশে ব্যাংক অ্যাকাউন্ট খোলার জন্য সাধারণত প্রয়োজন..."
```

### Mixed Language Support:
The bot can handle queries that mix both languages and will respond in the primary language detected.

## 🧪 Testing & Evaluation

### Run Evaluation:

```bash
python multilingual_eval.py
```

This will test:
- **English language performance**
- **Bangla language performance** 
- **Cross-lingual capabilities**
- **Context retrieval accuracy**

### Manual Testing:

```bash
# Test individual queries
python -c "
from multilingual_main import MultilingualFinancialAdvisorBot
bot = MultilingualFinancialAdvisorBot()
print(bot.process_query('What is TIN number?'))
print(bot.process_query('আয়কর কী?'))
"
```

## 🎯 Supported Topics

### Banking (ব্যাংকিং):
- Account opening (অ্যাকাউন্ট খোলা)
- Banking services (ব্যাংকিং সেবা)
- Mobile banking (মোবাইল ব্যাংকিং)
- Online banking (অনলাইন ব্যাংকিং)

### Loans (ঋণ):
- Home loans (গৃহঋণ)
- Car loans (গাড়ির ঋণ)
- Business loans (ব্যবসায়িক ঋণ)
- Student loans (শিক্ষা ঋণ)

### Taxation (কর):
- Income tax (আয়কর)
- VAT (ভ্যাট)
- Tax returns (কর রিটার্ন)
- TIN registration (টিআইএন নিবন্ধন)

### Investment (বিনিয়োগ):
- Savings certificates (সঞ্চয়পত্র)
- Government bonds (সরকারি বন্ড)
- Fixed deposits (স্থায়ী আমানত)
- Investment policies (বিনিয়োগ নীতি)

## 🔍 Advanced Features

### 1. Multilingual Semantic Chunking
- Language-aware sentence splitting
- Different token estimation for Bangla vs English
- Similarity-based breakpoint detection
- Overlap handling for context preservation

### 2. Hybrid Re-ranking System
- **Semantic scoring**: Cross-encoder for meaning-based relevance
- **Lexical scoring**: Keyword matching with language bonuses
- **Language preference**: Prioritizes documents in query language
- **Form field filtering**: Removes template content

### 3. Smart Response Generation
- Language-specific prompts
- Cultural context awareness
- Currency formatting (৳/Tk)
- Appropriate tone for each language

## 📈 Performance Optimization

### Caching Strategy:
- 24-hour response cache
- Language-aware cache keys
- Automatic cache invalidation

### Memory Management:
- Efficient document chunking
- Optimized embedding storage
- Lazy model loading

### Speed Optimizations:
- Parallel document processing
- Batch embedding generation
- Optimized similarity search

## 🐛 Troubleshooting

### Common Issues:

1. **Language Detection Errors**:
   ```bash
   # Install language detection dependencies
   pip install langdetect polyglot
   ```

2. **Bangla Text Not Displaying**:
   - Ensure UTF-8 encoding in terminal
   - Use fonts that support Bangla characters

3. **Model Download Failures**:
   ```bash
   # Manual model download
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"
   ```

4. **Ollama Connection Issues**:
   ```bash
   # Check Ollama status
   ollama list
   ollama serve
   ```

### Debug Mode:

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

### Planned Features:
- **Voice Support**: Speech-to-text in both languages
- **Image Processing**: OCR for Bangla documents
- **More Languages**: Support for other regional languages
- **Advanced Analytics**: Usage patterns and performance metrics
- **API Integration**: REST API for external applications

### Model Improvements:
- Fine-tuned models for Bangladesh financial domain
- Better cross-lingual understanding
- Improved Bangla text processing
- Domain-specific embeddings

## 📝 Contributing

### Adding New Languages:
1. Update language detection in `MultilingualLanguageDetector`
2. Add language-specific prompt templates
3. Include language in evaluation framework
4. Test with native speakers

### Adding New Documents:
1. Place PDFs in appropriate language folder
2. Run `multilingual_semantic_chunking.py`
3. Test retrieval with sample queries
4. Update evaluation dataset

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs in `logs/` directory

## 🙏 Acknowledgments

- **Sentence Transformers** for multilingual embeddings
- **LangChain** for RAG framework
- **Ollama** for local LLM serving
- **FAISS** for efficient similarity search
- **Bangladesh Bank** for financial documentation

---

**Made with ❤️ for Bangladesh's financial inclusion**
