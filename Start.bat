@echo off
setlocal EnableDelayedExpansion
title Street Story Curator

cd /d "%~dp0"

:: ── Python 3.12 check ─────────────────────────────────────────────
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo  Python 3.12 is required but was not found.
    echo  Download from: https://www.python.org/downloads/release/python-31210/
    echo.
    pause
    exit /b 1
)

:: ── Python version check — recreate venv if version changed ───────
if exist "venv\Scripts\python.exe" (
    for /f "tokens=2" %%v in ('venv\Scripts\python.exe --version 2^>^&1') do set VENV_PY=%%v
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set SYS_PY=%%v
    if not "!VENV_PY!"=="!SYS_PY!" (
        echo.
        echo  Python version changed ^(!VENV_PY! -^> !SYS_PY!^).
        echo  Recreating virtual environment...
        echo.
        rmdir /s /q venv
    )
)

:: ── First-run setup ───────────────────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo.
    echo  First launch: setting up environment.
    echo  This downloads ~1.5 GB of libraries and takes 5-10 minutes.
    echo  Subsequent launches will be instant.
    echo.

    py -3.12 -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo  ERROR: Could not create virtual environment with Python 3.12.
        echo  Make sure Python 3.12 is installed: https://www.python.org/downloads/
        pause & exit /b 1
    )

    venv\Scripts\python.exe -m pip install --upgrade pip --quiet

    echo  [1/3] Installing PyTorch (CPU)...
    venv\Scripts\pip.exe install torch torchvision ^
        --index-url https://download.pytorch.org/whl/cpu --quiet
    if %ERRORLEVEL% neq 0 (
        echo  ERROR: PyTorch install failed. Check your internet connection.
        pause & exit /b 1
    )

    echo  [2/3] Installing CLIP...
    venv\Scripts\pip.exe install ^
        "clip @ git+https://github.com/openai/CLIP.git" --quiet
    if %ERRORLEVEL% neq 0 (
        echo  WARNING: CLIP install failed (git may not be installed).
        echo  Some grading features may be limited.
    )

    echo  [3/3] Installing remaining dependencies...
    venv\Scripts\pip.exe install -r requirements.txt --quiet
    if %ERRORLEVEL% neq 0 (
        echo  ERROR: Dependency install failed.
        pause & exit /b 1
    )

    echo.
    echo  Setup complete.
    echo.
    echo setup_ok > venv\.setup_ok
)

:: ── Build frontend if dist is missing ────────────────────────────
if not exist "frontend\dist\index.html" (
    where npm >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo  Building UI...
        cd frontend
        call npm install --silent 2>nul
        call npm run build --silent 2>nul
        cd ..
    ) else (
        echo  WARNING: npm not found — cannot build frontend.
        echo  Install Node.js from https://nodejs.org then re-run.
        pause & exit /b 1
    )
)

:: ── Launch ────────────────────────────────────────────────────────
start "" venv\Scripts\pythonw.exe src\local_launcher.py
