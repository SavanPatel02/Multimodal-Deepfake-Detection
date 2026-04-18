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
echo Installing Requirements...
echo ===============================
pip install -r requirements.txt

echo ===============================
echo Installing PyTorch (GPU)...
echo ===============================
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

echo ===============================
echo Setup Completed Successfully!
echo ===============================
pause