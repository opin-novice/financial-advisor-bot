import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Configuration
FAISS_INDEX_PATH = "faiss_index_"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Load FAISS index
print("[INFO] Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"}  # use "cpu" if needed
)
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Set up LLM and retriever
retriever = vectorstore.as_retriever()
llm = OllamaLLM(model="llama3")  # or any other model available

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query loop
while True:
    query = input("Ask me anything (type 'exit' to quit):\n> ")
    if query.lower() == "exit":
        break
    try:
        response = qa.invoke(query)

        # Print the answer
        print("\n🔍 Answer:\n")
        print(response['result'])

        # Print the source documents
        print("\n📚 Sources:")
        for i, doc in enumerate(response['source_documents'], 1):
            print(f"\n--- Source {i} ---")
            print(doc.page_content[:500])  # show the first 500 characters of the chunk
            if 'metadata' in doc and 'page' in doc.metadata:
                print(f"[Page: {doc.metadata['page']}]")
            print("-" * 40)

    except Exception as e:
        print(f"[ERROR] {e}")
