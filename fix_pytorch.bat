@echo off
REM Fix PyTorch installation for CUDA support
REM This script uses the venv pip automatically

echo ========================================
echo Upgrading to PyTorch 2.9.0 for CUDA 12.6
echo ========================================
echo.
echo Your GPU: NVIDIA GeForce RTX 3080
echo CUDA Version: 12.6
echo.
echo This will:
echo   1. Uninstall old PyTorch version
echo   2. Upgrade NumPy (PyTorch 2.9+ compatible)
echo   3. Install PyTorch 2.9.0 stable with CUDA 12.6 support
echo.
echo Why: PyTorch 2.6+ fixes CVE-2025-32434 security vulnerability
echo      and allows loading all model formats (pickle-based models)
echo.
pause

echo.
echo [1/3] Uninstalling old PyTorch...
venv\Scripts\pip.exe uninstall -y torch torchvision torchaudio

echo.
echo [2/3] Upgrading NumPy...
venv\Scripts\pip.exe install --upgrade "numpy>=1.24.0"

echo.
echo [3/3] Installing PyTorch 2.9.0 stable with CUDA 12.6 support...
echo Note: This installs the latest stable release with CVE fix and full CUDA support
venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Verifying installation...
venv\Scripts\python.exe -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo If CUDA available = True, you're ready to go!
echo.
pause
