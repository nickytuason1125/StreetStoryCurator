@echo off
rem FirstCut launcher — double-click to start the app.
rem The windowed launcher builds the frontend if needed, starts the
rem backend server, and opens the app window. Safe to run twice:
rem a second launch focuses the existing window instead of duplicating.
cd /d "%~dp0"
rem Lite-by-default on this 16 GB machine — see LITE_MODE.md. Full quality
rem needs 3.8-4.2 GB free; the Lite profile fits and is what has actually
rem been working. Remove these two lines to restore full mode.
set SIGLIP_TIER=low
set FIRSTCUT_LITE=1
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0src\local_launcher.py" %*