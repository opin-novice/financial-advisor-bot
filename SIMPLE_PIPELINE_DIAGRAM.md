# 🚀 Simple RAG Pipeline Diagram for Reports

## 🔄 Main Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    🚀 COMPLETE RAG PIPELINE FLOW                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

📱 USER INPUT
    │
    ▼
┌─────────────────┐
│   main.py       │ ← System Entry Point
└─────────────────┘
    │
    ▼
┌─────────────────┐
│telegram_bot.py  │ ← Telegram Interface
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  bot_core.py    │ ← Central Orchestrator
└─────────────────┘
    │
    ▼
┌─────────────────┐
│language_utils.py│ ← Language Detection & Translation
└─────────────────┘
    │
    ▼
┌─────────────────┐
│document_retriever│ ← Document Search (FAISS + BM25)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│document_ranker  │ ← Quality Ranking & Filtering
└─────────────────┘
    │
    ▼
┌─────────────────┐
│response_generator│ ← AI Answer Generation
└─────────────────┘
    │
    ▼
┌─────────────────┐
│telegram_bot.py  │ ← Response Delivery
└─────────────────┘
    │
    ▼
📤 USER RECEIVES ANSWER
```

## 📚 Document Processing Pipeline

```
📄 PDF DOCUMENTS
    │
    ▼
┌─────────────────┐
│extract_pdf_text │ ← Text Extraction
└─────────────────┘
    │
    ▼
┌─────────────────┐
│pdf_preprocessor │ ← Document Cleaning
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   docadd.py     │ ← Semantic Chunking & Embedding
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  faiss_index/  │ ← Vector Database
└─────────────────┘
```

## 🏗️ Supporting Components

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ⚙️ SUPPORTING INFRASTRUCTURE                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   config.py     │  │  rag_utils.py   │  │delta_indexing.py│  │index_manager.py │
│                 │  │                 │  │                 │  │                 │
│ • Settings      │  │ • Utilities     │  │ • Version       │  │ • Index         │
│ • Parameters    │  │ • Validation    │  │   Tracking      │  │   Management    │
│ • Model Config  │  │ • Helper Tools  │  │ • Updates       │  │ • Operations    │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│setup_nltk.py    │  │setup_ollama.py  │  │requirements.txt │  │     .env        │
│                 │  │                 │  │                 │  │                 │
│ • Language      │  │ • AI Model      │  │ • Dependencies  │  │ • API Keys      │
│   Tools         │  │   Setup         │  │ • Versions      │  │ • Environment   │
│ • NLTK Data     │  │ • Configuration │  │ • Management    │  │ • Variables     │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 📊 Evaluation Framework

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    📈 EVALUATION & MONITORING                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│bertscore_evaluation │  │comprehensive_eval   │  │rag_file_analyzer    │
│                     │  │                     │  │                     │
│ • BERTScore        │  │ • Multi-Metric      │  │ • Pipeline          │
│ • Cosine Similarity│  │ • Cross-Encoder     │  │   Analysis          │
│ • ROUGE Scores     │  │ • Validation        │  │ • Performance       │
│ • Quality Metrics  │  │ • Assessment        │  │   Monitoring        │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐
│evaluation_test_cases│  │rag_accuracy_report  │
│                     │  │                     │
│ • Test Scenarios    │  │ • Performance       │
│ • Reference Answers │  │   Metrics           │
│ • Benchmark Data    │  │ • Historical Data   │
└─────────────────────┘  └─────────────────────┘
```

## 🔄 Data Flow Summary

### **📥 INPUT → PROCESSING → OUTPUT**

```
1. 📄 DOCUMENTS → 📚 PROCESSING → 🔍 SEARCHABLE INDEX
2. ❓ QUESTION → 🌍 LANGUAGE DETECTION → 🔍 DOCUMENT SEARCH
3. 📚 DOCUMENTS → 📊 RANKING → 💬 ANSWER GENERATION
4. 💬 ANSWER → 🌍 TRANSLATION → 📤 RESPONSE DELIVERY
```

## 🎯 Key Innovation Points

```
🔄 Iterative Feedback Loop    🌍 Multilingual Processing
🔍 Hybrid Retrieval          📊 Cross-Encoder Re-ranking
⚡ Delta Indexing            🎯 Semantic Chunking
```

## 🏆 System Characteristics

- **Modular Architecture** - Each component has a specific role
- **Scalable Design** - Handles thousands of documents efficiently  
- **Fault Tolerant** - Graceful degradation if components fail
- **Language Agnostic** - Supports multiple languages with translation
- **Performance Optimized** - Hybrid search with intelligent caching
- **Research Ready** - Comprehensive evaluation and monitoring

---

*This pipeline represents a production-ready, multilingual RAG system designed for financial document intelligence.*
