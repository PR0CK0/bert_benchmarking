@echo off
REM Fix PyTorch installation for CUDA support
REM Run this with virtual environment activated

echo ========================================
echo Upgrading to PyTorch 2.6+ for CUDA
echo ========================================
echo.
echo Your GPU: NVIDIA GeForce RTX 3080
echo CUDA Version: 12.6
echo.
echo This will:
echo   1. Uninstall old PyTorch version
echo   2. Upgrade NumPy (PyTorch 2.6+ compatible)
echo   3. Install PyTorch 2.6+ with CUDA 12.1 support
echo.
echo Why: PyTorch 2.6+ fixes CVE-2025-32434 security vulnerability
echo      and allows loading all model formats
echo.
pause

echo.
echo [1/3] Uninstalling old PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo [2/3] Upgrading NumPy...
pip install --upgrade "numpy>=1.24.0"

echo.
echo [3/3] Installing PyTorch 2.6+ nightly with CUDA 12.1 support...
echo Note: This installs the pre-release version with CVE fix and NumPy 2.x support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo If CUDA available = True, you're ready to go!
echo.
pause
