@echo off
echo 🌾 Rice Disease API - Local Server
echo ===================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo 🚀 Starting API server...
echo The server will be available at: http://localhost:8000
echo API docs will be at: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo.

python api.py

pause 