# Codebase Conventions for Agentic Coding Agents

## Commands
- **Install dependencies**: `pip install -r requirements.txt`
- **Run tests**: `python -m pytest` (no specific test framework identified)
- **Run single test**: `python -m pytest path/to/test_file.py::test_function_name -v`
- **Run RAG evaluation**: `python eval.py`
- **Start Telegram bot**: `python main.py`
- **Add documents to index**: `python docadd.py`
- **Set up NLTK**: `python setup_nltk.py`

## Code Style Guidelines
- **Imports**: Use standard Python imports; group standard library, third-party, and local imports with blank lines
- **Naming**: Use snake_case for variables and functions, PascalCase for classes
- **Types**: Use type hints for function parameters and return values
- **Formatting**: Follow PEP 8 (indent with 4 spaces, max line length 88)
- **Error Handling**: Use try/except blocks with specific exception types when appropriate
- **Comments**: Avoid unnecessary comments; code should be self-explanatory
- **String Formatting**: Use f-strings for string interpolation
- **Logging**: Use the logging module with appropriate levels (INFO, WARNING, ERROR)

## Project Structure
- **main.py**: Core Telegram bot implementation
- **eval.py**: RAGAS evaluation script with Groq integration
- **docadd.py**: Document processing and FAISS index creation
- **data/**: Source PDF documents
- **faiss_index/**: Vector database files
- **logs/**: Log files and evaluation results

## Testing
- No specific testing framework configured; use pytest if adding tests
- Evaluation uses RAGAS metrics with custom Groq integration