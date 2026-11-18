@echo off
REM SentimentIQ launcher for Windows: double-click this file.
REM First run: creates a virtual environment, installs dependencies and
REM trains the lightweight models. Later runs just start the app.
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv || goto :error
    echo Installing dependencies - this takes a few minutes the first time...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
)

if not exist models\lr_classifier.joblib (
    echo Training the TF-IDF sentiment model...
    venv\Scripts\python.exe -m src.train_model
)
if not exist models\hate_classifier.joblib (
    echo Training the TF-IDF hate-speech model...
    venv\Scripts\python.exe -m src.hate_speech
)

echo Starting SentimentIQ at http://localhost:8501 ...
venv\Scripts\python.exe -m streamlit run app.py
goto :eof

:error
echo.
echo Setup failed. Make sure Python 3.10+ is installed and on PATH, then try again.
pause
