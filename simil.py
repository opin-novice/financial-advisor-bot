from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fuzzywuzzy import fuzz
import json

# --- Configuration ---
EVAL_DATA_PATH = "dataqa/eval_set.json"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_MODEL = "llama3.2:3b"

# --- Load Embeddings & Vector Store ---
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

# --- Setup Ollama LLM ---
llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0,
    max_tokens=512
)

# --- QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --- Load evaluation samples from JSON ---
def load_eval_samples(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        samples.append({
            "query": item["question"],
            "expected_answer": item["ground_truth"],
            "contexts": item.get("contexts", [])
        })
    return samples

# --- Evaluation Logic ---
def evaluate_rag(samples):
    correct = 0
    total = len(samples)
    
    print(f"Evaluating {total} questions...\n")
    
    for i, sample in enumerate(samples, 1):
        print(f"Question {i}/{total}: {sample['query']}")
        
        try:
            response = qa_chain.invoke(sample["query"])
            generated_answer = response["result"]
            print(f"Generated Answer: {generated_answer}")
            print(f"Expected Answer: {sample['expected_answer']}")
            
            score = fuzz.token_sort_ratio(
                generated_answer.lower(), sample['expected_answer'].lower()
            )
            print(f"Similarity Score: {score}")
            
            if score >= 70:
                correct += 1
                print("Result: ✅ Correct\n")
            else:
                print("Result: ❌ Incorrect\n")
                
        except Exception as e:
            print(f"Error processing question: {e}")
            print("Result: ❌ Error\n")
    
    accuracy = (correct / total) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

# --- Run ---
if __name__ == "__main__":
    eval_samples = load_eval_samples(EVAL_DATA_PATH)
    evaluate_rag(eval_samples)