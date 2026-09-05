@echo off
rem FirstCut launcher — double-click to start the app.
rem The windowed launcher builds the frontend if needed, starts the
rem backend server, and opens the app window. Safe to run twice:
rem a second launch focuses the existing window instead of duplicating.
cd /d "%~dp0"
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0src\local_launcher.py" %*