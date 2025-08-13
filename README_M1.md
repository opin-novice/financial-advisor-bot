# M1-Optimized RAGAS Evaluation

## Quick Start (M1/M2 MacBooks)

### 1. Environment Setup
```bash
# Load M1-optimized settings
source m1_env.sh

# Or set manually:
export GROQ_API_KEY="your-key-here"
export EVAL_NUM_SAMPLES=5
export EVAL_DELAY_SECONDS=1
```

### 2. Run Diagnostics
```bash
python diagnose_m1.py  # Full M1 system check
```

### 3. Run Evaluation
```bash
python eval_m1_optimized.py  # M1-optimized evaluation
```

## M1 Performance Benefits

- **2-3x faster** evaluation processing
- **Higher throughput** with increased k values (10-15 vs 5)
- **Better quality** with larger token limits (1000+ vs 500)
- **Thermal management** with intelligent retry logic

## Optimization Settings

| Setting | Standard | M1 Optimized | Benefit |
|---------|----------|--------------|---------|
| Batch size | 16 | 32-64 | Faster embeddings |
| Retrieval k | 5 | 10-15 | Better context |
| Max tokens | 500 | 1000+ | Better responses |
| Delay | 5s | 1-2s | Faster evaluation |

## Troubleshooting

### Memory Issues
- Reduce NUM_SAMPLES if seeing memory errors
- Use smaller embedding models for limited RAM

### Performance Issues  
- Check thermal throttling with Activity Monitor
- Ensure no other heavy processes running
- Use `diagnose_m1.py` for detailed analysis

### API Issues
- Verify GROQ_API_KEY is set correctly
- Check rate limits with reduced delay settings

## Files

- `eval_m1_optimized.py` - Main M1-optimized evaluation
- `diagnose_m1.py` - M1 system diagnostic  
- `m1_env.sh` - Environment settings
- `RAGAS_IMPROVEMENT_GUIDE.md` - Detailed optimization guide
