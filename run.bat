@echo off
setlocal
REM ==========================================================================
REM  Senti-Q launcher for Windows: double-click this file.
REM  First run: creates a virtual environment, installs dependencies and trains
REM  the lightweight models (5-10 minutes). Later runs start in seconds.
REM  Your browser opens automatically once the app is ready.
REM ==========================================================================
cd /d "%~dp0"
title Senti-Q

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ from python.org and tick "Add to PATH".
    goto :error
)

if not exist venv\Scripts\python.exe (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 goto :error
)

venv\Scripts\python.exe -c "import streamlit, sklearn, vaderSentiment" >nul 2>nul
if errorlevel 1 (
    echo [2/3] Installing dependencies - the first time takes several minutes...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

if not exist models\lr_classifier.joblib (
    echo [3/3] Training the lightweight sentiment model...
    venv\Scripts\python.exe -m src.train_model
)
if not exist models\hate_classifier.joblib (
    echo [3/3] Training the lightweight hate-speech model...
    venv\Scripts\python.exe -m src.hate_speech
)

REM Open the browser as soon as the server answers (waits up to 2 minutes).
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 120;$i++){try{Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health -TimeoutSec 2 | Out-Null; Start-Process 'http://localhost:8501'; break}catch{Start-Sleep -Seconds 1}}"

echo.
echo  Senti-Q is starting at http://localhost:8501
echo  Your browser will open automatically. Keep this window open while you use
echo  the app - closing it stops the server.
echo.
venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
if errorlevel 1 goto :error
goto :eof

:error
echo.
echo Something went wrong - see the messages above.
pause
exit /b 1
