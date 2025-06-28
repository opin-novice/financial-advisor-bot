import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Configuration
FAISS_INDEX_PATH = "faiss_index"  # Same as docadd.py
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Load FAISS index
print("[INFO] Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Setup retriever with top-k chunks retrieval
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Setup LLM
#llm = OllamaLLM(model="llama3.2:1b") 
#for better performance we are using llama3.2:3b
llm = OllamaLLM(model="llama3.2:3b")  # Make sure this matches your Ollama model 

# Build the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query loop
while True:
    query = input("Ask me anything (type 'exit' to quit):\n> ").strip()
    if query.lower() == "exit":
        print("[INFO] Exiting.")
        break
    if not query:
        print("[WARN] Please enter a non-empty question.")
        continue

    try:
        response = qa.invoke(query)

        # Print answer
        print("\n🔍 Answer\n" + "="*40)
        print(response['result'])

        # Print sources with page and source file
        print("\n📚 Sources")
        print("="*40)
        for i, doc in enumerate(response['source_documents'], 1):
            print(f"\n--- Source {i} ---")
            print(doc.page_content[:500])  # preview chunk text
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            print(f"[Page: {page}] [Source: {source}]")
            print("-" * 40)

    except Exception as e:
        print(f"[ERROR] {e}")
