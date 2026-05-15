@echo off
setlocal EnableDelayedExpansion
title Street Story Curator — Setup
cd /d "%~dp0"

echo.
echo  ================================================================
echo   Street Story Curator
echo  ================================================================
echo.

:: ── Python 3.12 check ─────────────────────────────────────────────
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  ERROR: Python 3.12 is required but was not found.
    echo.
    echo  Download from:
    echo    https://www.python.org/downloads/release/python-31210/
    echo.
    echo  Make sure to check "Add Python to PATH" during install.
    pause & exit /b 1
)

:: ── Node.js check ─────────────────────────────────────────────────
where npm >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  ERROR: Node.js is required but was not found.
    echo.
    echo  Download from: https://nodejs.org  (LTS version)
    echo.
    pause & exit /b 1
)

:: ── Skip setup if already done ────────────────────────────────────
if exist "venv\.setup_ok" goto :launch

:: ──────────────────────────────────────────────────────────────────
::  FIRST-RUN INSTALL
:: ──────────────────────────────────────────────────────────────────
echo  First launch — installing Street Story Curator.
echo  This takes 10-20 minutes depending on your connection.
echo  Do not close this window.
echo.

:: Remove stale venv if it exists
if exist "venv\Scripts\python.exe" (
    echo  Removing old environment...
    rmdir /s /q venv
)

:: Create venv
echo  [1/6] Creating Python environment...
py -3.12 -m venv venv
if %ERRORLEVEL% neq 0 (
    echo  ERROR: Could not create virtual environment.
    pause & exit /b 1
)
venv\Scripts\python.exe -m pip install --upgrade pip --quiet

:: ── Detect CUDA ───────────────────────────────────────────────────
set CUDA_AVAILABLE=0
set TORCH_EXTRA=
nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set CUDA_AVAILABLE=1
    :: Read CUDA version from nvidia-smi
    for /f "tokens=9" %%v in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VER=%%v
    echo  GPU detected  ^(CUDA !CUDA_VER!^)
) else (
    echo  No NVIDIA GPU detected — installing CPU-only PyTorch
)

:: ── Install PyTorch ───────────────────────────────────────────────
echo  [2/6] Installing PyTorch...
if !CUDA_AVAILABLE! equ 1 (
    venv\Scripts\pip.exe install "torch==2.5.1" "torchvision==0.20.1" ^
        --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    venv\Scripts\pip.exe install "torch==2.5.1" "torchvision==0.20.1" ^
        --index-url https://download.pytorch.org/whl/cpu --quiet
)
if %ERRORLEVEL% neq 0 (
    echo  ERROR: PyTorch install failed. Check your internet connection.
    pause & exit /b 1
)

:: ── Install AI model dependencies ─────────────────────────────────
echo  [3/6] Installing AI model libraries...
venv\Scripts\pip.exe install transformers bitsandbytes torchao --quiet
if %ERRORLEVEL% neq 0 (
    echo  WARNING: Some AI libraries failed to install. Grading may be limited.
)

:: ── Install remaining Python dependencies ─────────────────────────
echo  [4/6] Installing remaining dependencies...
venv\Scripts\pip.exe install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (
    echo  ERROR: Dependency install failed.
    pause & exit /b 1
)

:: ── Build frontend ────────────────────────────────────────────────
echo  [5/6] Building UI...
cd frontend
call npm install --silent 2>nul
call npm run build --silent 2>nul
cd ..
if not exist "frontend\dist\index.html" (
    echo  ERROR: UI build failed. Check Node.js installation.
    pause & exit /b 1
)

:: ── Create desktop shortcut ───────────────────────────────────────
echo  [6/6] Creating desktop shortcut...
set APP_DIR=%~dp0
set APP_DIR=!APP_DIR:~0,-1!
powershell -NoProfile -NonInteractive -Command ^
    "$ws = New-Object -ComObject WScript.Shell; " ^
    "$lnk = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Street Story Curator.lnk'); " ^
    "$lnk.TargetPath = 'wscript.exe'; " ^
    "$lnk.Arguments = '/b \"!APP_DIR!\launch_hidden.vbs\"'; " ^
    "$lnk.WorkingDirectory = '!APP_DIR!'; " ^
    "$lnk.IconLocation = '!APP_DIR!\icon.ico'; " ^
    "$lnk.Description = 'Street Story Curator — AI Photo Culler'; " ^
    "$lnk.Save(); Write-Host 'Shortcut created.'" 2>nul

echo setup_ok > venv\.setup_ok

echo.
echo  ================================================================
echo   Setup complete!
echo.
echo   A shortcut has been placed on your Desktop.
echo   Double-click it any time to launch the app.
echo  ================================================================
echo.
timeout /t 3 >nul

:: ──────────────────────────────────────────────────────────────────
:launch
:: ── Rebuild frontend if dist is missing (after a git pull etc.) ───
if not exist "frontend\dist\index.html" (
    where npm >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo  Rebuilding UI...
        cd frontend
        call npm install --silent 2>nul
        call npm run build --silent 2>nul
        cd ..
    )
)

:: Kill any stale processes on port 8000
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)

start "" venv\Scripts\pythonw.exe src\local_launcher.py
