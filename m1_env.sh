#!/bin/bash
# M1-Optimized Environment Settings
export TOKENIZERS_PARALLELISM="false"
export MPS_AVAILABLE="1"
export OMP_NUM_THREADS="1"
export EVAL_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export EVAL_NUM_SAMPLES="5"
export EVAL_DELAY_SECONDS="1"
export EVAL_MAX_TOKENS="1000"
export GROQ_API_KEY=""
