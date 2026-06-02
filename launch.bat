@echo off
setlocal EnableDelayedExpansion
title Magnum Engine — Street Photo Curation AI
cd /d "%~dp0"

:: Enable ANSI colour support (Windows 10+)
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1

echo.
echo  ============================================================
echo   Magnum Engine — Street Photo Curation AI
echo  ============================================================
echo.

:: ── Python check ──────────────────────────────────────────────────────────
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    where py >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo  [ERROR] Python not found. Install from https://python.org
        echo         Make sure "Add Python to PATH" is checked.
        pause & exit /b 1
    )
    set PYTHON=py -3
) else (
    set PYTHON=python
)

:: ── Create venv if absent ─────────────────────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    if not exist "venv\Scripts\python3.exe" (
        echo  Creating Python virtual environment...
        %PYTHON% -m venv venv
        if %ERRORLEVEL% neq 0 (
            echo  [ERROR] Could not create virtual environment.
            pause & exit /b 1
        )
        echo  Virtual environment created.
        echo.
    )
)

:: ── Activate venv ─────────────────────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── Run the setup wizard (installs deps, checks Ollama, pulls models, launches server) ──
python scripts\setup_wizard.py %*

:: Keep the window open if the wizard exited with an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Setup wizard exited with code %ERRORLEVEL%.
    pause
)
