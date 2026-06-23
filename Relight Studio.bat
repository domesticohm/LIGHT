@echo off
REM ===== Relight Studio launcher =====
REM Starts the local AI backend (your GPU) and opens the studio in your browser.
REM Everything runs on your machine. First AI render downloads models once (~5GB).
cd /d "%~dp0"
echo Starting Relight Studio backend...
start "Relight Studio Server" ".venv\Scripts\python.exe" "backend\server.py"
echo Waiting for server...
timeout /t 4 /nobreak >nul
start "" "http://localhost:7860"
echo.
echo Relight Studio is running at http://localhost:7860
echo Keep the server window open while you use the app. Close it to quit.
