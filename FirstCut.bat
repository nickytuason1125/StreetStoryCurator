@echo off
rem ============================================================
rem  FirstCut — simple reliable launcher (Lite mode)
rem  Kills stray processes, starts ONE backend directly (no
rem  exec chains), waits for it, opens the UI in the browser.
rem ============================================================
title FirstCut launcher
cd /d "%~dp0"
rem Lite-by-default on this 16 GB machine — see LITE_MODE.md. Full quality
rem needs 3.8-4.2 GB free; the Lite profile fits and is what has actually
rem been working. Remove these two lines to restore full mode.
set SIGLIP_TIER=low
set FIRSTCUT_LITE=1

echo [1/3] Clearing stale FirstCut processes...
taskkill /F /IM pythonw.exe /T >nul 2>&1
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/3] Starting the FirstCut backend (Lite mode)...
start "FirstCut backend" /min "venv\Scripts\python.exe" -m uvicorn server_impl:app --host 127.0.0.1 --port 8000

echo      Waiting for it to come up (can take ~1 min on a busy machine)...
set /a tries=0
:waitloop
timeout /t 3 /nobreak >nul
set /a tries+=1
curl -s -o nul -w "%%{http_code}" -H "X-Requested-With: FirstCut" http://127.0.0.1:8000/api/whoami 2>nul | findstr "200" >nul
if errorlevel 1 (
    if %tries% lss 40 goto waitloop
    echo ERROR: the backend did not come up in 2 minutes. Close some apps and run this again.
    pause
    exit /b 1
)

echo [3/3] Opening FirstCut in your browser...
start "" http://127.0.0.1:8000
echo FirstCut is running (Lite mode). Keep the minimized "FirstCut backend" window open while grading.
pause