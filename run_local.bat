@echo off
cd /d "%~dp0"

:: Kill stale backend (both python.exe and pythonw.exe) and WebView2
taskkill /F /FI "IMAGENAME eq python.exe"         >nul 2>&1
taskkill /F /FI "IMAGENAME eq pythonw.exe"        >nul 2>&1
taskkill /F /FI "IMAGENAME eq msedgewebview2.exe" >nul 2>&1

:: Free ports 8000 (API) and 5173 (Vite dev) if anything still holds them
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5173 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Hard-purge stale computed artifacts before launch
if exist "cache\catalog.json"        del /f /q "cache\catalog.json"
if exist "cache\archetype_embs.npy"  del /f /q "cache\archetype_embs.npy"
if exist "cache\archetype_embs.hash" del /f /q "cache\archetype_embs.hash"

start "" wscript.exe "%~dp0launch_hidden.vbs"
