## CRUSH.md - Codebase Configuration

### Build Commands
- `python setup_multilingual.py`: Setup the multilingual environment
- `python multilingual_semantic_chunking.py`: Create multilingual vector index

### Test Commands
- `python multilingual_eval.py`: Run multilingual evaluation
- `python -c "from multilingual_main import MultilingualFinancialAdvisorBot; bot = MultilingualFinancialAdvisorBot(); print(bot.process_query('What is TIN number?'))"`: Test specific queries

### Lint Commands
- `flake8 financial-advisor-bot/`: Check code style
- `black --check financial-advisor-bot/`: Check code formatting

### Code Style Guidelines
- Use Python 3.9+ with type hints
- Follow PEP8 guidelines
- Use snake_case for variables
- Handle errors with try/except blocks
- Use Ollama for translation as per project requirements
- Keep documents in UTF-8 encoding
- Use Bangla Unicode (`\u0980-\u09FF`) for Bangla support
- Separate concerns between translation and financial logic
- Use logging for multilingual debugging
- Document all financial terms in both languages

### Additional Notes
- The project uses Ollama for translation
- The main bot is in multilingual_main.py
- Documents are processed in multilingual_semantic_chunking.py
- Evaluation is handled in multilingual_eval.py
- Setup is done via setup_multilingual.py
- Requirements are in requirements_multilingual.txt