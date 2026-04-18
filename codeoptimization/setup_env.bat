@echo off

echo ===============================
echo Creating Virtual Environment...
echo ===============================
python -m venv venv

echo ===============================
echo Activating Environment...
echo ===============================
call venv\Scripts\activate

echo ===============================
echo Upgrading pip...
echo ===============================
pip install --upgrade pip

echo ===============================
echo Installing PyTorch (CUDA 12.4)...
echo ===============================
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

echo ===============================
echo Installing Requirements...
echo ===============================
pip install -r requirements.txt

echo ===============================
echo Verifying GPU...
echo ===============================
python check_gpu.py

echo ===============================
echo Setup Completed Successfully!
echo ===============================
pause
