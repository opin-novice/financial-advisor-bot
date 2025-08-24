# 🚀 RAG System Pipeline Architecture Diagram

## 🔄 Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    📱 USER INTERFACE LAYER                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    🎯 ENTRY POINT                                                        │
│  ┌─────────────┐    ┌─────────────────────────────┐    ┌─────────────────────────────────────────────┐   │
│  │   main.py   │───▶│    telegram_bot.py         │───▶│         bot_core.py                        │   │
│  │             │    │                             │    │                                             │   │
│  │ Starts      │    │ • Receives user questions   │    │ • Orchestrates entire RAG pipeline         │   │
│  │ the system  │    │ • Handles Telegram API      │    │ • Manages language detection               │   │
│  │             │    │ • Sends responses            │    │ • Coordinates all components               │   │
│  └─────────────┘    └─────────────────────────────┘    └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    🌍 LANGUAGE PROCESSING LAYER                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              language_utils.py                                                      │   │
│  │                                                                                                     │   │
│  │  • Language Detection (English/Bangla)                                                              │   │
│  │  • Translation Services                                                                              │   │
│  │  • Bilingual Response Formatting                                                                    │   │
│  │  • Script-based Detection (Unicode ranges)                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    🔍 DOCUMENT RETRIEVAL LAYER                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                            document_retriever.py                                                    │   │
│  │                                                                                                     │   │
│  │  • Hybrid Search (Semantic + Keyword)                                                              │   │
│  │  • FAISS Vector Search                                                                              │   │
│  │  • BM25 Keyword Search                                                                              │   │
│  │  • Multi-Query Generation                                                                           │   │
│  │  • Delta Indexing Integration                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                                                │
│                                        ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                             document_ranker.py                                                      │   │
│  │                                                                                                     │   │
│  │  • Cross-Encoder Re-ranking                                                                        │   │
│  │  • Relevance Threshold Filtering                                                                   │   │
│  │  • Quality Assessment                                                                              │   │
│  │  • Document Filtering                                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    💬 RESPONSE GENERATION LAYER                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                           response_generator.py                                                     │   │
│  │                                                                                                     │   │
│  │  • LLM Integration (Ollama)                                                                        │   │
│  │  • Context-Aware Answer Generation                                                                 │   │
│  │  • Language-Specific Prompting                                                                     │   │
│  │  • Response Translation                                                                            │   │
│  │  • Source Citation                                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    📤 RESPONSE DELIVERY LAYER                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              telegram_bot.py                                                        │   │
│  │                                                                                                     │   │
│  │  • Message Chunking                                                                                │   │
│  │  • Source Organization                                                                             │   │
│  │  • Language-Appropriate Formatting                                                                 │   │
│  │  • User Response Delivery                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

## 🏗️ SUPPORTING INFRASTRUCTURE

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ⚙️ CONFIGURATION & UTILITIES                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │   config.py     │    │  rag_utils.py   │    │delta_indexing.py│    │index_manager.py │              │
│  │                 │    │                 │    │                 │    │                 │              │
│  │ • System        │    │ • Cosine        │    │ • Document      │    │ • Index         │              │
│  │   Settings      │    │   Similarity    │    │   Version       │    │   Management    │              │
│  │ • Model Config  │    │ • Validation    │    │   Tracking      │    │ • CRUD          │              │
│  │ • Parameters    │    │ • Helper Tools  │    │ • Delta Updates │    │   Operations    │              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 📚 DOCUMENT PROCESSING PIPELINE

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    📄 DOCUMENT INGESTION FLOW                                            │
│                                                                                                         │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   data/     │───▶│extract_pdf_text │───▶│pdf_preprocessor │───▶│   docadd.py     │───▶│faiss_index/│ │
│  │             │    │                 │    │                 │    │                 │    │             │ │
│  │ • PDF Files │    │ • Text          │    │ • Document      │    │ • Semantic      │    │ • Vector    │ │
│  │ • JSON Docs │    │   Extraction    │    │   Cleaning      │    │   Chunking      │    │   Database  │ │
│  │ • Raw Data  │    │ • OCR Support   │    │ • Format        │    │ • Embedding     │    │ • Searchable│ │
│  │             │    │ • Error Handling│    │   Standardization│    │   Generation    │    │   Index     │ │
│  └─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 SETUP & INITIALIZATION

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    🚀 SYSTEM INITIALIZATION                                             │
│                                                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │setup_nltk.py    │    │setup_ollama.py  │    │requirements.txt │    │     .env        │              │
│  │                 │    │                 │    │                 │    │                 │              │
│  │ • NLTK Data     │    │ • Ollama Model  │    │ • Python        │    │ • API Keys      │              │
│  │   Download      │    │   Installation  │    │   Dependencies  │    │ • Configuration │              │
│  │ • Language      │    │ • Model         │    │ • Version       │    │ • Environment   │              │
│  │   Tools Setup   │    │   Configuration │    │   Management    │    │   Variables     │              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 📊 EVALUATION & MONITORING

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    📈 EVALUATION FRAMEWORK                                              │
│                                                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐                        │
│  │bertscore_evaluation │    │comprehensive_eval   │    │rag_file_analyzer    │                        │
│  │                     │    │                     │    │                     │                        │
│  │ • BERTScore Metrics │    │ • Multi-Metric      │    │ • Pipeline Analysis │                        │
│  │ • Cosine Similarity │    │   Assessment        │    │ • Performance       │                        │
│  │ • ROUGE Scores      │    │ • Cross-Encoder     │    │   Monitoring        │                        │
│  │ • Quality Metrics   │    │   Validation        │    │ • Optimization      │                        │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘                        │
│                                                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐                                                    │
│  │evaluation_test_cases│    │rag_accuracy_report  │                                                    │
│  │                     │    │                     │                                                    │
│  • Test Scenarios      │    │ • Performance       │                                                    │
│  • Reference Answers   │    │   Metrics           │                                                    │
│  • Benchmark Data      │    │ • Historical Data   │                                                    │
│  └─────────────────────┘    └─────────────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 COMPLETE DATA FLOW SUMMARY

### **Phase 1: Document Processing**
```
PDF Documents → Text Extraction → Preprocessing → Semantic Chunking → Vector Embeddings → FAISS Index
```

### **Phase 2: Query Processing**
```
User Question → Language Detection → Query Translation (if needed) → Document Retrieval → Re-ranking
```

### **Phase 3: Answer Generation**
```
Ranked Documents → Context Preparation → LLM Generation → Response Translation → Formatting
```

### **Phase 4: Response Delivery**
```
Formatted Answer → Source Citation → Message Chunking → Telegram Delivery → User Receives Answer
```

## 🎯 KEY INNOVATION POINTS

1. **🔄 Iterative Feedback Loop** (Advanced RAG with query refinement)
2. **🌍 Multilingual Processing** (English ↔ Bangla with translation pipeline)
3. **🔍 Hybrid Retrieval** (Semantic + Keyword + Domain-specific variations)
4. **📊 Cross-Encoder Re-ranking** (Quality-based document filtering)
5. **⚡ Delta Indexing** (Efficient document updates)
6. **🎯 Semantic Chunking** (Intelligent document splitting)

## 🏆 SYSTEM CHARACTERISTICS

- **Modular Architecture**: Each component has a specific, well-defined role
- **Scalable Design**: Can handle thousands of documents efficiently
- **Fault Tolerant**: Graceful degradation if components fail
- **Language Agnostic**: Supports multiple languages with translation
- **Performance Optimized**: Hybrid search with intelligent caching
- **Research Ready**: Comprehensive evaluation and monitoring capabilities

---

*This pipeline represents a production-ready, multilingual RAG system designed for financial document intelligence with advanced retrieval capabilities and robust evaluation frameworks.*
