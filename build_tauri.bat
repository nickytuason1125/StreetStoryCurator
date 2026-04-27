@echo off
setlocal

echo ============================================================
echo  Street Story Curator — Tauri v2 Build
echo ============================================================

:: Set up MSVC linker environment (puts the correct link.exe first in PATH)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not load VS 2022 Build Tools.
    echo        Install them from: https://aka.ms/vs/17/release/vs_BuildTools.exe
    pause
    exit /b 1
)

:: Add Rust to PATH
set "PATH=C:\Users\Nicky Tuason\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin;C:\Users\Nicky Tuason\.cargo\bin;%PATH%"

rustc --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: rustc not found. Install Rust from https://rustup.rs
    pause
    exit /b 1
)

:: Build the React frontend first so server.py can serve it
cd /d "%~dp0frontend"
echo.
echo [1/2] Building React frontend ...
call npm run build
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: npm run build failed.
    pause
    exit /b 1
)

:: Build the Tauri native shell (compiles Rust, bundles NSIS installer)
echo.
echo [2/2] Building Tauri native shell ...
call npm run tauri -- build
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: tauri build failed (exit code %ERRORLEVEL%)
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Installer: frontend\src-tauri\target\release\bundle\nsis\
echo  Portable exe: frontend\src-tauri\target\release\street-story-curator.exe
echo ============================================================
pause
