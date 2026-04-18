@echo off

call venv\Scripts\activate

echo ===============================
echo Running Prediction...
echo ===============================

:: Usage: run.bat <path_to_video>
:: Example: run.bat "C:\Videos\sample.mp4"

if "%~1"=="" (
    echo ERROR: Please provide a video path.
    echo Usage: run.bat "path\to\video.mp4"
    pause
    exit /b 1
)

python codeoptimization\predict_video.py "%~1"

pause
