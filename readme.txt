pip install langchain
pip install -U langchain-community
pip install sentence-transformers
pip install langchain_ollama
pip install -U langchain-huggingface

install ollama then:
ollama pull llama3.2:3b

for gpu
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
for cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


pip install faiss-cpu
pip install python-telegram-bot --upgrade
pip install pypdf