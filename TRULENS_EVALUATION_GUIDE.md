# 🔍 TruLens Evaluation Guide for Advanced Multilingual RAG System

This guide explains how to use the TruLens evaluation feature integrated into your Advanced Multilingual RAG System.

## 🎯 What is TruLens Evaluation?

TruLens evaluation provides comprehensive metrics for your RAG system including:

- **Local Fast Metrics**: Cross-encoder based relevance and groundedness scores
- **Language Consistency**: Measures if responses match query language
- **TruLens LLM Judge**: Advanced AI-powered evaluation metrics (optional)
- **Multilingual Support**: Evaluates both English and Bangla queries

## 🚀 Quick Start

### 1. **Basic Setup**

```bash
# Install TruLens dependencies (optional)
pip install -r requirements_trulens.txt

# Test the evaluation system
python test_trulens_eval.py
```

### 2. **Basic Evaluation**

```bash
# Evaluate 10 samples from your dataset
python TruLens_eval.py --qa dataqa/multilingual_eval_set.json --limit 10

# Evaluate all samples
python TruLens_eval.py --qa dataqa/multilingual_eval_set.json --limit 0
```

### 3. **Save Results to CSV**

```bash
python TruLens_eval.py --qa dataqa/multilingual_eval_set.json --limit 20 --out reports/evaluation_results.csv
```

### 4. **Enable TruLens Dashboard (Optional)**

```bash
# Set OpenAI API key for LLM judge metrics
export OPENAI_API_KEY="sk-your-openai-key-here"

# Run with TruLens dashboard
python TruLens_eval.py --qa dataqa/multilingual_eval_set.json --trulens

# View dashboard (run in separate terminal)
python -m trulens_eval.dashboard --port 8501
# Open: http://localhost:8501/
```

## 📊 Evaluation Metrics

### **Local Fast Metrics** (Always Available)

1. **Context Relevance (ctx_relevance_ce)**
   - Measures how relevant retrieved contexts are to the question
   - Range: 0.0 to 1.0 (higher is better)
   - Uses cross-encoder model for accurate scoring

2. **Answer Groundedness (ans_groundedness_ce)**
   - Measures how well the answer is supported by retrieved contexts
   - Range: 0.0 to 1.0 (higher is better)
   - Prevents hallucination and ensures factual accuracy

3. **Language Consistency (language_consistency)**
   - Measures if the answer language matches the query language
   - 1.0 = Perfect match, 0.5 = Mixed language, 0.0 = Mismatch
   - Critical for multilingual systems

### **TruLens LLM Judge Metrics** (Optional - Requires OpenAI API Key)

1. **Answer Relevance**
   - AI-powered assessment of answer quality
   - Uses advanced reasoning for evaluation

2. **Context Relevance**
   - AI evaluation of retrieved context quality
   - More nuanced than cross-encoder metrics

3. **Groundedness with Chain-of-Thought**
   - Advanced hallucination detection
   - Provides reasoning for evaluation decisions

### **System Metrics**

- **Detected Language**: Automatically detected query language
- **Language Confidence**: Confidence in language detection (0.0-1.0)
- **Feedback Iterations**: Number of query refinement iterations used
- **Number of Contexts**: Count of retrieved document chunks

## 📁 Data Format

### **Input QA Format**

Your QA file should be JSON or JSONL with questions:

```json
[
  {
    "query": "What are the requirements to open a savings account?",
    "expected_language": "english",
    "category": "banking"
  },
  {
    "query": "সঞ্চয় হিসাব খুলতে কি কি প্রয়োজন?",
    "expected_language": "bengali", 
    "category": "banking"
  }
]
```

**Required Fields:**
- `query` or `question` or `input`: The question to evaluate

**Optional Fields:**
- `expected_language`: Expected language for validation
- `category`: Question category for analysis
- `positive`: Expected answer (for future recall metrics)

### **Output CSV Format**

```csv
sample_id,question,answer,detected_language,language_confidence,feedback_iterations,ctx_relevance_ce,ans_groundedness_ce,language_consistency,n_contexts,first_source
1,"What is a savings account?","A savings account is...","english",0.95,2,0.85,0.92,1.0,5,"{""source"": ""banking_guide.pdf""}"
```

## 🔧 Advanced Usage

### **Custom Evaluation Dataset**

Create your own evaluation dataset:

```python
import json

qa_pairs = [
    {
        "query": "How to apply for a loan?",
        "expected_language": "english",
        "category": "loans"
    },
    {
        "query": "ঋণের জন্য কিভাবে আবেদন করব?",
        "expected_language": "bengali",
        "category": "loans"
    }
]

with open("my_eval_set.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
```

### **Batch Evaluation Script**

```bash
#!/bin/bash
# evaluate_all.sh

echo "🚀 Running comprehensive evaluation..."

# Basic metrics
python TruLens_eval.py --qa dataqa/multilingual_eval_set.json --limit 50 --out reports/basic_eval.csv

# TruLens metrics (if OpenAI key available)
if [ ! -z "$OPENAI_API_KEY" ]; then
    python TruLens_eval.py --qa dataqa/multilingual_eval_set.json --limit 20 --trulens --out reports/trulens_eval.csv
fi

echo "✅ Evaluation completed!"
```

### **Performance Tuning**

For large evaluations, consider:

```bash
# Use smaller limits for testing
python TruLens_eval.py --qa data.json --limit 10

# Process in batches
python TruLens_eval.py --qa data.json --limit 50 --out batch1.csv
python TruLens_eval.py --qa data.json --limit 50 --out batch2.csv  # Skip first 50
```

## 📈 Interpreting Results

### **Good Performance Indicators**

- **Context Relevance > 0.7**: Retrieved contexts are relevant
- **Answer Groundedness > 0.8**: Answers are well-supported
- **Language Consistency = 1.0**: Perfect language matching
- **Low Feedback Iterations**: System finds good results quickly

### **Areas for Improvement**

- **Low Context Relevance**: Improve document indexing or retrieval
- **Low Answer Groundedness**: Tune prompt templates or LLM parameters
- **Poor Language Consistency**: Improve language detection or prompts
- **High Feedback Iterations**: Optimize query refinement strategies

### **Multilingual Analysis**

```python
import pandas as pd

# Load results
df = pd.read_csv("evaluation_results.csv")

# Language-specific performance
english_perf = df[df['detected_language'] == 'english']['ctx_relevance_ce'].mean()
bengali_perf = df[df['detected_language'] == 'bengali']['ctx_relevance_ce'].mean()

print(f"English performance: {english_perf:.3f}")
print(f"Bengali performance: {bengali_perf:.3f}")
```

## 🛠️ Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements_trulens.txt
   ```

2. **FAISS Index Not Found**
   ```bash
   # Rebuild your vector index
   python docadd.py
   ```

3. **GROQ API Errors**
   ```bash
   # Check your API key in the script
   # Ensure you have sufficient credits
   ```

4. **TruLens Dashboard Issues**
   ```bash
   # Install TruLens properly
   pip install trulens-eval
   
   # Set OpenAI key
   export OPENAI_API_KEY="your-key"
   ```

### **Performance Issues**

1. **Slow Evaluation**
   - Use `--limit` to test with fewer samples
   - Consider using CPU-only mode for cross-encoder
   - Reduce context length in config

2. **Memory Issues**
   - Lower batch sizes in cross-encoder
   - Use smaller embedding models
   - Process in smaller chunks

## 📊 Sample Reports

### **Basic Evaluation Report**

```
📊 EVALUATION RESULTS
============================================================

🔍 Sample Results:
   sample_id                    question detected_language  ctx_relevance_ce  ans_groundedness_ce  language_consistency
0          1  What is a savings account?           english             0.856                0.923                 1.000
1          2            ব্যাংক হিসাব কি?           bengali             0.792                0.887                 1.000

📈 Average Metrics:
  ctx_relevance_ce: 0.824
  ans_groundedness_ce: 0.905
  language_consistency: 1.000
  language_confidence: 0.943
  feedback_iterations: 1.500

🌍 Language Distribution:
  english: 10 samples (50.0%)
  bengali: 10 samples (50.0%)
```

## 🎯 Best Practices

1. **Start Small**: Begin with 10-20 samples to test the system
2. **Use Representative Data**: Include both English and Bangla queries
3. **Monitor Performance**: Track metrics over time as you improve the system
4. **Validate Results**: Manually review a few samples to ensure quality
5. **Iterate**: Use results to improve prompts, retrieval, and system configuration

## 🔗 Integration with Existing Workflow

The TruLens evaluation integrates seamlessly with your existing system:

- Uses your existing `language_utils.py` for language detection
- Leverages your `advanced_rag_feedback.py` for query processing
- Utilizes your `config.py` for system configuration
- Works with your existing FAISS index and embeddings

This ensures evaluation results reflect your actual system performance!

## 📞 Support

If you encounter issues:

1. Run `python test_trulens_eval.py` to diagnose problems
2. Check the logs in `logs/` directory
3. Verify your system setup with the existing demo scripts
4. Ensure all dependencies are installed correctly

The TruLens evaluation feature provides comprehensive insights into your multilingual RAG system's performance, helping you identify areas for improvement and track progress over time.
