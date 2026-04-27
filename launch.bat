@echo off
cd /d "%~dp0"

:: Force-release crash.log and kill zombie processes
taskkill /F /FI "IMAGENAME eq python.exe" 2>nul
taskkill /F /FI "IMAGENAME eq pythonw.exe" 2>nul
timeout /t 2 >nul
del /Q /F crash.log 2>nul

:: Kill any leftover Gradio on our ports
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":786" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: Launch local desktop app via pywebview (no external browser required)
start /b "" wscript.exe "%~dp0launch_hidden.vbs"
