@echo off
setlocal EnableDelayedExpansion
title FrameGrade - One-Click Install
cd /d "%~dp0"

echo.
echo  ================================================================
echo    FrameGrade  -  One-Click Installer
echo  ================================================================
echo.

:: ── Already installed? Just launch. ───────────────────────────────
if exist "venv\.setup_ok" (
    echo  FrameGrade is already installed.
    echo  Launching...
    start "" wscript.exe //b "%~dp0launch_hidden.vbs"
    timeout /t 2 >nul
    exit /b 0
)

:: ── STEP 1: Python 3.12 ───────────────────────────────────────────
echo  [1/4] Checking for Python 3.12...
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% equ 0 goto :python_ok

echo        Python 3.12 not found - installing automatically via winget...
winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
if %ERRORLEVEL% neq 0 (
    echo.
    echo  ERROR: Automatic Python install failed.
    echo  Install it manually from: https://www.python.org/downloads/release/python-31210/
    echo  ^(tick "Add Python to PATH"^), then run this installer again.
    pause & exit /b 1
)

:: winget installed Python, but THIS window's PATH predates it. Add the
:: standard per-user + all-users locations so we can continue immediately.
set "PYHOME="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYHOME=%LocalAppData%\Programs\Python\Python312"
if exist "%ProgramFiles%\Python312\python.exe"              set "PYHOME=%ProgramFiles%\Python312"
if defined PYHOME set "PATH=%PYHOME%;%PYHOME%\Scripts\;%PATH%"
set "LAUNCHER=%LocalAppData%\Programs\Python\Launcher\py.exe"
if exist "%LAUNCHER%" set "PATH=%LocalAppData%\Programs\Python\Launcher;%PATH%"

py -3.12 --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    if defined PYHOME goto :python_ok
    echo.
    echo  ERROR: Python was installed but is not visible yet.
    echo  Close this window, open a NEW window, and run this installer again.
    pause & exit /b 1
)
:python_ok
for /f "tokens=2" %%v in ('py -3.12 --version 2^>nul') do echo        Found %%v

:: ── STEP 2: Node.js (ONLY when the UI needs building) ─────────────
echo  [2/4] Checking for the prebuilt UI...
if exist "frontend\dist\index.html" (
    echo        Prebuilt UI found - Node.js not needed.
    goto :node_done
)
where npm >nul 2>&1
if %ERRORLEVEL% equ 0 goto :node_done

echo        UI build needed and Node.js missing - installing via winget...
winget install -e --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
if %ERRORLEVEL% neq 0 (
    echo  ERROR: Automatic Node.js install failed.
    echo  Install the LTS from https://nodejs.org and run this installer again.
    pause & exit /b 1
)
set "PATH=%ProgramFiles%\nodejs;%PATH%"
where npm >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  ERROR: Node installed but not visible yet. Open a NEW window and retry.
    pause & exit /b 1
)
:node_done
echo        OK

:: ── STEP 3+4: full install + shortcut (delegates to Start.bat) ────
echo  [3/4] Installing FrameGrade components...
echo        (PyTorch, AI libraries, UI build - 10-20 minutes)
call "%~dp0Start.bat"
if not exist "venv\.setup_ok" (
    echo.
    echo  ERROR: Install did not complete. Re-run this installer.
    pause & exit /b 1
)

echo  [4/4] Done!
echo.
echo  ================================================================
echo   FrameGrade is installed and launching now.
echo   A shortcut has been placed on your Desktop -
echo   from now on, double-click THAT to start the app instantly.
echo  ================================================================
echo.
timeout /t 5 >nul