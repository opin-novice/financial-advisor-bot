#!/usr/bin/env python3
"""
Simple script to run the complete data processing workflow
"""

import os
import sys

# Change to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# Add scripts to path
sys.path.append('scripts')

def main():
    print("🚀 Starting Financial Advisor Bot Data Processing Workflow")
    print("=" * 60)
    
    try:
        # Import and run the final process
        from final_process import run_workflow
        run_workflow()
        
        print("\n✅ All done! Your RAG system is ready to use.")
        print("\nNext steps:")
        print("- Run 'python scripts/test_index.py' to test the FAISS index")
        print("- Start your bot with 'python main.py'")
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
