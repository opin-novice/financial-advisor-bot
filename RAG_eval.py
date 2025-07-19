from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fuzzywuzzy import fuzz

# Configuration
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_MODEL = "llama3.2:3b"  # Change if needed

# Load embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Setup Ollama LLM
llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0,
    max_tokens=512
)

# Setup QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Evaluation samples: list of dicts with question & expected answer
eval_samples = [
    {
        "query": "What is the VAT rate in Bangladesh?",
        "expected_answer": "The VAT rate in Bangladesh is 15%."
    },
    {
        "query": "How much penalty is there for late tax submission?",
        "expected_answer": "The penalty is a fine or interest charged on late submissions."
    }
    # Add more test samples here...
]

def evaluate_rag(samples):
    correct = 0
    total = len(samples)
    
    for sample in samples:
        print(f"Question: {sample['query']}")
        response = qa_chain.invoke(sample["query"])
        generated_answer = response["result"]
        print(f"Generated Answer: {generated_answer}")
        print(f"Expected Answer: {sample['expected_answer']}")
        
        score = fuzz.token_sort_ratio(
            generated_answer.lower(), sample['expected_answer'].lower()
        )
        print(f"Similarity Score: {score}")
        
        # Threshold can be adjusted (e.g., 70)
        if score >= 70:
            correct += 1
            print("Result: ✅ Correct\n")
        else:
            print("Result: ❌ Incorrect\n")
    
    accuracy = (correct / total) * 100
    print(f"Final Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

if __name__ == "__main__":
    evaluate_rag(eval_samples)
