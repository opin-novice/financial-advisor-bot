# AGENTS.md

## Build/Lint/Test
- Build: `pip install -r requirements.txt`
- Lint: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
- Test all: `pytest`
- Single test: `pytest tests/test_advanced_rag_feedback.py::test_feedback_loop`

## Code Style
- PEP8 with 2-space indents
- Group imports: stdlib → 3rd party → local
- Type hints required for public APIs
- Snake_case for functions/methods
- CamelCase for classes
- Handle errors with logging.exception()
- Use specific exceptions (no bare except)
- Docstrings for public modules/classes/functions