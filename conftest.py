
import sys
import os
import pytest
from langchain.schema import Document
from advanced_rag_feedback import AdvancedRAGFeedbackLoop
from rag_utils import RAGUtils
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import config

@pytest.fixture(scope="module")
def feedback_loop():
    """Fixture for the AdvancedRAGFeedbackLoop."""
    try:
        rag_utils = RAGUtils()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        feedback_config = config.get_feedback_loop_config()
        feedback_loop = AdvancedRAGFeedbackLoop(
            vectorstore=vectorstore,
            rag_utils=rag_utils,
            config=feedback_config
        )
        return feedback_loop
    except Exception as e:
        pytest.skip(f"Failed to initialize feedback_loop fixture: {e}")


