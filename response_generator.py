from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

from config import config
from language_utils import LanguageDetector
from rag_utils import RAGUtils

class ResponseGenerator:
    """Handles response generation and language-specific formatting"""
    
    def __init__(self, language_detector: LanguageDetector):
        self.language_detector = language_detector
        self.rag_utils = RAGUtils()
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM"""
        try:
            # Initialize Ollama LLM
            self.llm = ChatOllama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=0.5,
                num_predict=1200,
                top_p=0.9
            )
            print(f"[INFO] ✅ Ollama LLM initialized with model: {config.OLLAMA_MODEL}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Ollama LLM: {e}")
            self.llm = None

    def create_language_specific_chain(self, detected_language: str):
        """Create retrieval chain with appropriate language-specific prompt"""
        # Determine response language
        response_language = self.language_detector.determine_response_language(detected_language)
        
        # Get language-specific prompt
        language_prompt = self.language_detector.get_language_specific_prompt(response_language)
        
        # Create document chain with language-specific prompt
        return create_stuff_documents_chain(self.llm, language_prompt)
    
    def create_english_chain(self):
        """Create retrieval chain with English prompt (for translation approach)"""
        # Always use English prompt for translation approach
        english_prompt = self.language_detector._get_english_prompt()
        
        # Create document chain with English prompt
        return create_stuff_documents_chain(self.llm, english_prompt)

    def process_query_traditional(self, query: str, category: str, original_language: str, 
                                retriever, ranker, use_remote_faiss: bool = False) -> Dict:
        """Traditional RAG processing without feedback loop with translation support"""
        print("[INFO] 🔄 Using traditional RAG approach...")
        
        # Get configuration for translation approach
        feedback_config = config.get_feedback_loop_config()
        use_translation = feedback_config.get("use_translation_approach", True)
        enable_response_translation = feedback_config.get("enable_response_translation", True)
        
        # Step 1: Retrieve documents
        if use_remote_faiss:
            # Use remote FAISS API
            print("[INFO] 🔍 Retrieving documents from remote FAISS API...")
            source_docs = self._query_remote_faiss(query, config.MAX_DOCS_FOR_RETRIEVAL * 2, retriever)
        else:
            # Use local FAISS
            print("[INFO] 🔍 Retrieving documents from local FAISS...")
            source_docs = retriever.get_relevant_documents(query)
        
        if not source_docs:
            return {"response": "I couldn't find relevant information in my database.", "sources": [], "contexts": []}
        
        # Step 2: Apply filtering and ranking
        filtered = ranker.rank_and_filter(source_docs, query)
        docs = ranker.prepare_docs(filtered)
        
        # Step 3: Generate answer using appropriate chain
        if use_translation:
            # Translation approach: use English chain
            qa_chain = self.create_english_chain()
            result = qa_chain.invoke({"input": query, "context": docs})
            answer = result
            
            # Translate answer back to Bangla if needed
            if original_language == 'bengali' and enable_response_translation:
                print("[INFO] 🔄 Translating answer back to Bangla...")
                answer = self.language_detector.translate_english_to_bangla(answer)
                print("[INFO] ✅ Answer translated to Bangla")
        else:
            # Traditional approach: use language-specific chain
            qa_chain = self.create_language_specific_chain(original_language)
            result = qa_chain.invoke({"input": query, "context": docs})
            answer = result
        
        # Step 4: Prepare sources and contexts
        sources = []
        contexts = []
        for doc in docs:
            source_info = {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0)
            }
            sources.append(source_info)
            contexts.append(doc.page_content)

        return {
            "response": answer,
            "sources": sources,
            "contexts": contexts
        }

    def _query_remote_faiss(self, query: str, top_k: int, retriever) -> List[Document]:
        """Query the remote FAISS API and return LangChain Documents"""
        # This is a simplified version - in a full implementation, this would make HTTP requests
        # For now, we'll just use the local retriever as a fallback
        print("[WARNING] Remote FAISS not implemented in refactored version, using local retriever")
        return retriever.get_relevant_documents(query)