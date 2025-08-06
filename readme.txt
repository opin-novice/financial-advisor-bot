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

quantization8bit
pip install torch transformers accelerate bitsandbytes
pip install safetensors
debug
$env:CUDA_LAUNCH_BLOCKING=1

evaluation
pip install datasets
pip install ragas

ocr data sanitization

install tesseract ocr
 https://github.com/UB-Mannheim/tesseract/wiki
 from here
 then get bengali pack from here:
  https://github.com/tesseract-ocr/tessdata
  ben.traindata something
  put it in 
  C:\Program Files\Tesseract-OCR\tessdata\
  
pip install pytesseract pdf2image opencv-python transformers torch reportlab numpy PyMuPDF   

pip install opencv-python matplotlib
pip install pymupdf pillow numpy


pip install huggingface_hub
huggingface-cli login