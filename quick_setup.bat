@echo off
REM Financial Advisor Bot - Windows Setup Script

echo 🤖 Financial Advisor Telegram Bot - Windows Setup
echo ================================================

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)
echo ✅ Python found

REM Check if Ollama is installed
echo Checking Ollama installation...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama not found. Please install from https://ollama.ai/download
    pause
    exit /b 1
)
echo ✅ Ollama found

REM Create virtual environment
echo Setting up virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment and install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ✅ Dependencies installed

REM Create .env file
echo Setting up environment configuration...
if not exist ".env" (
    if exist ".env.template" (
        copy .env.template .env >nul
        echo ✅ Environment file created from template
        echo ⚠️ IMPORTANT: Edit .env file and add your Telegram bot token!
        echo ℹ️ Replace 'YOUR_TELEGRAM_BOT_TOKEN_HERE' with your actual bot token
    ) else (
        echo # Telegram Bot Token (REQUIRED) > .env
        echo TG_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE >> .env
        echo. >> .env
        echo # Optional configurations (these have defaults) >> .env
        echo FAISS_INDEX_PATH=faiss_index_multilingual >> .env
        echo EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2 >> .env
        echo OLLAMA_MODEL=gemma3n:e2b >> .env
        echo ✅ Environment file created
        echo ⚠️ IMPORTANT: Edit .env file and add your Telegram bot token!
    )
) else (
    echo ✅ Environment file already exists
)

REM Download model
echo Downloading AI model (this may take 10-15 minutes)...
echo ⚠️ Model size: ~5.6GB - ensure you have good internet connection
ollama pull gemma3n:e2b
echo ✅ Model downloaded

REM Final instructions
echo.
echo ================================================
echo 🎉 Setup completed successfully!
echo.
echo To start the bot:
echo   1. Start Ollama: ollama serve
echo   2. In another terminal, activate environment: venv\Scripts\activate.bat
echo   3. Start the bot: python main.py
echo.
echo To test the bot, send /start to your Telegram bot
echo ================================================
pause
