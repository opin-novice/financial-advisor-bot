from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fuzzywuzzy import fuzz
import fitz  # PyMuPDF

# --- Configuration ---
PDF_PATH = "eval_questions.pdf"
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

# --- Extract Q&A pairs from PDF ---
def extract_eval_samples_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    lines = text.strip().split("\n")
    samples = []
    question, answer = "", ""

    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
            if question and answer:
                samples.append({"query": question, "expected_answer": answer})
                question, answer = "", ""
    return samples

# --- Evaluation Logic ---
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
        
        if score >= 70:
            correct += 1
            print("Result: ✅ Correct\n")
        else:
            print("Result: ❌ Incorrect\n")
    
    accuracy = (correct / total) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

# --- Run ---
if __name__ == "__main__":
    eval_samples = extract_eval_samples_from_pdf(PDF_PATH)
    evaluate_rag(eval_samples)
