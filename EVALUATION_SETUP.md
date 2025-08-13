# RAGAS Evaluation Setup Guide

## Problem Identified
The evaluation was returning NaN values because of an **invalid API key** causing authentication failures. All the API calls were failing, which resulted in empty/null metric values.

## Solution

### 1. Get a Valid Groq API Key
1. Go to [Groq Console](https://console.groq.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Generate a new API key

### 2. Set the Environment Variable
Before running the evaluation, set your API key:

```bash
export GROQ_API_KEY="your_actual_api_key_here"
```

### 3. Test the Simple Version First
Run the simple test to verify your API key works:

```bash
python test_eval.py
```

You should see:
- ✅ API Response: API working
- ✅ Evaluation completed!
- Actual numeric scores instead of NaN

### 4. Run the Full Evaluation
Once the simple test works, run the full evaluation:

```bash
python eval.py
```

## What Was Fixed

1. **Removed hardcoded API key** for security
2. **Fixed data format mapping** to match RAGAS expectations:
   - `question` → `user_input`
   - `contexts` → `retrieved_contexts`  
   - `answer` → `response`
   - `ground_truth` → `reference`
3. **Reduced sample size** to 3 samples for faster testing
4. **Reduced delay** to 5 seconds between requests
5. **Simplified LLM wrapper** to avoid complex rate limiting issues

## Expected Results
After fixing the API key, you should see results like:

```
📊 RAGAS EVALUATION RESULTS
==================================================
📈 SCORES:
   context_precision: 0.8500
   context_recall: 0.9200
   faithfulness: 0.7800
   answer_relevancy: 0.8900
```

Instead of all NaN values.

## Troubleshooting

If you still get NaN values:
1. Verify your API key is valid by running `test_eval.py`
2. Check that you have sufficient API credits/quota
3. Ensure your internet connection is stable
4. Try reducing `NUM_SAMPLES` to 1 for testing

## Environment Variables
You can customize the evaluation with these environment variables:

```bash
export GROQ_API_KEY="your_key"
export EVAL_NUM_SAMPLES="5"          # Number of samples to evaluate
export EVAL_DELAY_SECONDS="3"       # Delay between API requests
export EVAL_EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```
