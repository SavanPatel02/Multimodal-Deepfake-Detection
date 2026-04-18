@echo off

call venv\Scripts\activate

echo ===============================
echo Running Project...
echo ===============================

python scripts\predict_video.py

pause