# 🍎 FAISS Quick Fix for M1 MacBook

## ⚡ Quick Commands (Try These First)

### Option 1: Automated Fix (Recommended)
```bash
python fix_faiss_m1.py
```
This script will automatically detect your system and try the best installation method.

### Option 2: Manual Conda Fix (Best for M1)
```bash
# Install conda if you don't have it
# Download: https://docs.conda.io/en/latest/miniconda.html

# Install FAISS via conda-forge (M1 optimized)
conda install -c conda-forge faiss-cpu -y
```

### Option 3: Manual Pip Fix
```bash
# Clean previous installations
pip uninstall faiss-cpu faiss-gpu faiss -y

# Update pip tools
pip install --upgrade pip setuptools wheel

# Install FAISS
pip install faiss-cpu

# If that fails, try with no cache:
pip install --no-cache-dir faiss-cpu
```

### Option 4: Alternative (hnswlib)
```bash
# If FAISS keeps failing, use this fast alternative
pip install hnswlib
```

## 🧪 Test Your Installation

```python
# Test FAISS
import faiss
import numpy as np

vectors = np.random.random((100, 64)).astype('float32')
index = faiss.IndexFlatL2(64)
index.add(vectors)

query = np.random.random((1, 64)).astype('float32')
distances, indices = index.search(query, k=5)
print("✅ FAISS working!")
```

## 🔧 Common M1 Issues & Solutions

### Error: "No module named 'faiss'"
```bash
# Solution 1: Use conda
conda install -c conda-forge faiss-cpu -y

# Solution 2: Check Python environment
which python  # Make sure you're using the right Python
pip list | grep faiss  # Check if installed
```

### Error: "Symbol not found" or "Library not loaded"
```bash
# This is common on M1 - use conda-forge version
conda install -c conda-forge faiss-cpu -y
```

### Error: "Building wheel for faiss failed"
```bash
# Install build dependencies
xcode-select --install
pip install cmake

# Or use pre-built conda version
conda install -c conda-forge faiss-cpu -y
```

### Still Not Working?
```bash
# Use hnswlib as alternative
pip install hnswlib

# Then modify your code - see FAISS_M1_FIX_GUIDE.md
```

## 🍎 M1-Specific Tips

1. **Always prefer conda-forge** for M1 compatibility
2. **Use CPU version** - works better than GPU on M1  
3. **Install XCode tools** if building from source
4. **Consider hnswlib** as a fast, M1-friendly alternative

## 📚 Full Documentation

- Run `python fix_faiss_m1.py` for automated fixing
- Check `FAISS_M1_FIX_GUIDE.md` for detailed troubleshooting
- See `README_M1.md` for complete M1 optimization guide

## ✅ Verification

After installation, run:
```bash
python -c "import faiss; print('FAISS version:', faiss.__version__)"
```

You should see something like: `FAISS version: 1.7.4` ✅
