pip install langchain
pip install -U langchain-community
pip install sentence-transformers
pip install langchain_ollama
pip install -U langchain-huggingface

install ollama then:
ollama pull llama3.1

for gpu
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-cpu
pip install pypdf
pip install datasets
//new version of sentence transformers(ST)
pip install -U sentence-transformers
//(v5 of ST)
pip install "accelerate>=0.26.0" --upgrade
---------------------------------------------------
//for query in the qa_paris.jsonl more fine the model will become because fine tune training will use that
---------------------------------------------------------------------------
for DeepEval
//when to use? For big model like gpt 4 this option is best
//but for small model it scores can get noisy and often invalid 
//because the model struggles to emit the strict JSON that DeepEval expects 
pip install deepeval 
-----------------------------------------------------------------------------------------------///
pip install ragas datasets
#since the model is too small 
pip install ragas==0.1.11   # last version that still supports pure-statistical mode
#Stay on Ragas but force it to use Ollama
#for evaluate_ollama.py (still no fit for 1b)
pip install -U ragas==0.1.11 langchain-ollama 
------------------------------->
#trying to run it for 1b 
pip install -U "langchain-ollama>=0.2.2" "langchain-core>=0.3,<0.4" "ragas>=0.1.9"








----------------------------------------------only for cypherQAchain
pip install neo4j
pip install spacy
after this, 
python -m spacy download en_core_web_sm
need to download neo4j
pip install langchain-neo4j
