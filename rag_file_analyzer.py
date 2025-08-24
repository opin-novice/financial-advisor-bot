#!/usr/bin/env python3
"""
RAG Pipeline Core Files Analyzer
Identifies essential vs. non-essential files for the RAG system
"""

import os
import shutil
from pathlib import Path

def categorize_files():
    """Categorize files into essential and non-essential for RAG pipeline"""
    
    # Essential directories for RAG operation
    essential_dirs = {
        "faiss_index",           # Main FAISS index
        "faiss_index_incremental", # Incremental embedding index
        "data",                   # Processed JSON documents
        "incremental_embedding",   # Incremental embedding modules
        "logs"                    # Log files
    }
    
    # Essential files for RAG operation
    essential_files = {
        # Core modules
        "document_retriever.py",
        "document_retriever_bridge.py", 
        "rag_utils.py",
        "language_utils.py",
        "config.py",
        "main.py",
        "telegram_bot.py",
        "bot_core.py",
        "delta_indexing.py",
        
        # Incremental embedding modules
        "incremental_embedding/__init__.py",
        "incremental_embedding/hashing.py",
        "incremental_embedding/metadata_store.py", 
        "incremental_embedding/embedding.py",
        "incremental_embedding/faiss_index.py",
        "incremental_embedding/incremental_pipeline.py",
        
        # Requirements
        "requirements.txt",
        
        # Environment
        ".env.example"
    }
    
    # Non-essential files that can be safely removed
    non_essential_files = {
        # Evaluation and testing files
        "bertscore_evaluation.py",
        "comprehensive_evaluation.py", 
        "advanced_bertscore_optimizer.py",
        "research_grade_bertscore.py",
        "target_achievement_bertscore.py",
        "optimized_bertscore_evaluation.py",
        "evaluate_improved_bertscore.py",
        "test_cross_encoder.py",
        "test_delta_indexing.py",
        "test_hybrid_retrieval.py",
        "test_retrieval_quality.py",
        "test_rag_utils_integration.py",
        "test_phase1_improvements.py",
        "eval.py",
        "totaleval.py",
        "RAG_eval.py",
        "evaluate_improved_bertscore_fixed.py",
        "research_appropriate_bertscore.py",
        "final_guaranteed_bertscore.py",
        "high_performance_bertscore.py",
        "maximum_achievement_bertscore.py",
        "target_achievement_confirmed.py",
        "eval_embed.py",
        "RAG_eval.py",
        "rag_bertscore_integration.py",
        
        # Analysis and research files
        "analyze_actual_faiss.py",
        "actual_faiss_analysis_20250820_191242.json",
        "comprehensive_cosine_analysis.py",
        "cosine_similarity_research.py",
        "detailed_cosine_similarity_analysis_20250822_200829.json",
        "enhanced_faiss_analysis.py",
        "executive_summary_20250822_200829.txt",
        
        # Documentation/markdown files
        "COSINE_SIMILARITY_IMPROVEMENTS.md",
        "CROSS_ENCODER_IMPLEMENTATION.md", 
        "HYBRID_RETRIEVAL_IMPLEMENTATION.md",
        "PHASE1_IMPROVEMENTS_SUMMARY.md",
        "RAG_IMPROVEMENTS_SUMMARY.md",
        "RETRIEVER_QUICK_FIXES_SUMMARY.md",
        "GUIDE_SEMANTICSIMIL.md",
        "README_REMOTE_FAISS.md",
        
        # Temporary/backup files
        "main.py.backup",
        "pdf_preprocessor.py.corrupted",
        
        # Jupyter notebooks
        "financial_advisor_rag_faiss.ipynb",
        "financial_advisor_remote_api.ipynb", 
        "financial_advisor_remote_api_fixed.ipynb",
        "pdf_preprocessing_colab.ipynb",
        "ragas_evaluation_colab.ipynb",
        "ragas_evaluation_colab_final.ipynb",
        "ragas_evaluation_colab_fixed.ipynb",
        "ragas_evaluation_colab_updated.ipynb",
        "ragas_evaluation_colab.ipynb",
        "ragas_evaluation_minimal.ipynb",
        "ragas_evaluation_simple.ipynb",
        "test_notebook.ipynb",
        
        # Large dataset directories
        "evaluation_datasets",
        "evaluation_results",
        "evaluation_results_improved",
        "dataqa"
    }
    
    print("=== RAG PIPELINE FILE CATEGORIZATION ===")
    print()
    
    # Count files in each category
    project_root = Path(".")
    
    essential_count = 0
    non_essential_count = 0
    unknown_count = 0
    
    # Check all files in project
    for root, dirs, files in os.walk(project_root):
        # Skip hidden directories and Python cache
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != '.git']
        
        for file in files:
            if file.startswith('.'):
                continue
                
            file_path = Path(root) / file
            relative_path = file_path.relative_to(project_root)
            
            # Check if it's in an essential directory
            if any(str(relative_path).startswith(dir_name) for dir_name in essential_dirs):
                essential_count += 1
                continue
                
            # Check if it's an essential file
            if str(relative_path) in essential_files or str(relative_path).replace('\\', '/') in essential_files:
                essential_count += 1
                continue
                
            # Check if it's a non-essential file
            if str(relative_path) in non_essential_files or str(relative_path).replace('\\', '/') in non_essential_files:
                non_essential_count += 1
                continue
            
            # Unknown files
            unknown_count += 1
            print(f"UNKNOWN: {relative_path}")
    
    print(f"Essential files: {essential_count}")
    print(f"Non-essential files: {non_essential_count}") 
    print(f"Unknown files: {unknown_count}")
    print()
    
    # Show what can be safely removed
    print("=== SAFE TO REMOVE ===")
    print("The following files/directories can be safely removed without affecting RAG operation:")
    for item in sorted(non_essential_files):
        item_path = Path(item)
        if item_path.exists():
            size = "DIR" if item_path.is_dir() else f"{item_path.stat().st_size:,} bytes"
            print(f"  - {item} ({size})")
    
    print()
    print("=== KEEP FOR OPERATION ===")
    print("These files/directories are required for RAG pipeline to function:")
    for item in sorted(essential_dirs):
        item_path = Path(item)
        if item_path.exists():
            print(f"  - {item}/ (directory)")
    
    for item in sorted(essential_files):
        item_path = Path(item)
        if item_path.exists():
            size = "DIR" if item_path.is_dir() else f"{item_path.stat().st_size:,} bytes"
            print(f"  - {item} ({size})")

def cleanup_non_essential():
    """Safely remove non-essential files"""
    
    non_essential_files = {
        # Evaluation and testing files
        "bertscore_evaluation.py",
        "comprehensive_evaluation.py", 
        "advanced_bertscore_optimizer.py",
        "research_grade_bertscore.py",
        "target_achievement_bertscore.py",
        "optimized_bertscore_evaluation.py",
        "evaluate_improved_bertscore.py",
        "test_cross_encoder.py",
        "test_delta_indexing.py",
        "test_hybrid_retrieval.py",
        "test_retrieval_quality.py",
        "test_rag_utils_integration.py",
        "test_phase1_improvements.py",
        "eval.py",
        "totaleval.py",
        "RAG_eval.py",
        "evaluate_improved_bertscore_fixed.py",
        "research_appropriate_bertscore.py",
        "final_guaranteed_bertscore.py",
        "high_performance_bertscore.py",
        "maximum_achievement_bertscore.py",
        "target_achievement_confirmed.py",
        "eval_embed.py",
        "RAG_eval.py",
        "rag_bertscore_integration.py",
        
        # Analysis and research files
        "analyze_actual_faiss.py",
        "comprehensive_cosine_analysis.py",
        "cosine_similarity_research.py",
        "enhanced_faiss_analysis.py",
        
        # Documentation/markdown files
        "COSINE_SIMILARITY_IMPROVEMENTS.md",
        "CROSS_ENCODER_IMPLEMENTATION.md", 
        "HYBRID_RETRIEVAL_IMPLEMENTATION.md",
        "PHASE1_IMPROVEMENTS_SUMMARY.md",
        "RAG_IMPROVEMENTS_SUMMARY.md",
        "RETRIEVER_QUICK_FIXES_SUMMARY.md",
        "GUIDE_SEMANTICSIMIL.md",
        "README_REMOTE_FAISS.md",
        
        # Temporary/backup files
        "main.py.backup",
        "pdf_preprocessor.py.corrupted",
        
        # Jupyter notebooks
        "financial_advisor_rag_faiss.ipynb",
        "financial_advisor_remote_api.ipynb", 
        "financial_advisor_remote_api_fixed.ipynb",
        "pdf_preprocessing_colab.ipynb",
        "ragas_evaluation_colab.ipynb",
        "ragas_evaluation_colab_final.ipynb",
        "ragas_evaluation_colab_fixed.ipynb",
        "ragas_evaluation_colab_updated.ipynb",
        "ragas_evaluation_colab.ipynb",
        "ragas_evaluation_minimal.ipynb",
        "ragas_evaluation_simple.ipynb",
        "test_notebook.ipynb",
    }
    
    print("=== CLEANUP NON-ESSENTIAL FILES ===")
    
    removed_count = 0
    error_count = 0
    
    for item in non_essential_files:
        item_path = Path(item)
        if item_path.exists():
            try:
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                    print(f"Removed directory: {item}")
                else:
                    item_path.unlink()
                    print(f"Removed file: {item}")
                removed_count += 1
            except Exception as e:
                print(f"Error removing {item}: {e}")
                error_count += 1
    
    print(f"\nCleanup complete: {removed_count} items removed, {error_count} errors")

if __name__ == "__main__":
    categorize_files()
    print("\n" + "="*50)
    response = input("\nDo you want to cleanup non-essential files? (y/N): ")
    if response.lower().startswith('y'):
        cleanup_non_essential()