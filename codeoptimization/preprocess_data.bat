@echo off
cd /d "%~dp0.."

echo ================================================
echo  Multimodal Deepfake Detection
echo  Full Dataset Preprocessing
echo ================================================
echo.
echo This will process the ENTIRE LAV-DF dataset.
echo Already-processed files will be skipped automatically.
echo.
echo Steps:
echo   1. Parse metadata
echo   2. Extract audio (.wav)
echo   3. Generate mel spectrograms (.npy)
echo   4. Extract face frames (.jpg)
echo.

call venv\Scripts\activate

:: Run all 4 steps
python codeoptimization\preprocess_data.py

echo.
echo ================================================
echo  Preprocessing complete.
echo  Next: train your models with
echo  python codeoptimization\train_pair.py --model M1
echo ================================================
pause
