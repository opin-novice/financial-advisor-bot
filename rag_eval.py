import os
import re
from datasets import Dataset
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
import time

# --- Configuration ---
INDEX_DIR = "faiss_index"
TEST_PDF_DIR = "test_questions"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 1  # Process one question at a time to reduce memory usage

def process_pdf_files():
    """Process PDF files in batches to reduce memory usage"""
    examples = []
    
    pdf_files = [
        os.path.join(TEST_PDF_DIR, f)
        for f in os.listdir(TEST_PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF test files found in {TEST_PDF_DIR}")

    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = " ".join([page.page_content for page in pages])
        
        # Clear pages from memory
        del pages

        qa_pairs = re.findall(
            r"(?i)(?:\d+\.\s*)?Question[:\s]+(.+?)\s+Answer[:\s]+(.+?)(?=(?:\n\s*\d+\.\s*Question:|\Z))",
            full_text,
            re.DOTALL
        )

        if not qa_pairs:
            print(f"[WARN] Could not parse any question/answer pairs from: {pdf_path}")
        else:
            for question, answer in qa_pairs:
                examples.append({
                    "question": question.strip(),
                    "ground_truth": answer.strip()
                })

        # Clear full text from memory
        del full_text

    return examples

def process_question(qa_chain, question):
    """Process a single question"""
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"] if isinstance(result, dict) and "result" in result else str(result)
        contexts = [doc.page_content for doc in result.get("source_documents", [])] if isinstance(result, dict) else []
        return answer, contexts
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return "", []

def evaluate_metrics(dataset):
    """Calculate evaluation metrics manually"""
    results = {}
    
    # Simple metric calculations
    total_questions = len(dataset)
    answered_questions = sum(1 for answer in dataset["answer"] if answer.strip())
    
    # Basic metrics
    results["response_rate"] = answered_questions / total_questions if total_questions > 0 else 0
    
    # Context usage
    context_used = sum(1 for contexts in dataset["contexts"] if contexts)
    results["context_usage"] = context_used / total_questions if total_questions > 0 else 0
    
    # Answer length analysis
    avg_answer_length = sum(len(answer.split()) for answer in dataset["answer"]) / total_questions if total_questions > 0 else 0
    results["avg_answer_length"] = avg_answer_length
    
    return results

def main():
    print("[INFO] Loading FAISS index...")
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'mps'}  # Use M1 GPU acceleration
    )
    
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}  # Reduce number of retrieved documents
    )

    print("[INFO] Parsing test PDFs...")
    examples = process_pdf_files()
    
    if not examples:
        raise ValueError("No valid question/answer pairs found in test PDFs.")

    # Create the dataset
    dataset = Dataset.from_list(examples)

    # Setup RAG pipeline with optimized settings
    llm = OllamaLLM(
        model="llama2",
        temperature=0.1,  # Reduce temperature for more focused responses
        num_ctx=512,  # Reduce context window
        num_thread=2  # Limit number of threads
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    print("[INFO] Running RAG pipeline...")
    all_answers, all_contexts = [], []
    
    # Process questions one at a time
    for i, item in enumerate(dataset):
        print(f"Processing question {i+1}/{len(dataset)}")
        
        # Add a small delay between questions to prevent overload
        if i > 0:
            time.sleep(0.5)
            
        answer, contexts = process_question(qa_chain, item["question"])
        all_answers.append(answer)
        all_contexts.append(contexts)

    # Add results to dataset
    dataset = dataset.add_column("answer", all_answers)
    dataset = dataset.add_column("contexts", all_contexts)

    print("[INFO] Calculating evaluation metrics...")
    results = evaluate_metrics(dataset)

    print("\n📊 Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.3f}")

    # Save results
    dataset.to_csv("evaluation_output.csv")
    print("\n✅ Results saved to evaluation_output.csv")

    # Print sample of questions and answers
    print("\n📝 Sample Q&A (first 2):")
    for i in range(min(2, len(dataset))):
        print(f"\nQ{i+1}: {dataset[i]['question'][:100]}...")
        print(f"A: {dataset[i]['answer'][:100]}...")
        print(f"Ground Truth: {dataset[i]['ground_truth'][:100]}...")

if __name__ == "__main__":
    main()
