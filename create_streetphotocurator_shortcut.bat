@echo off
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_streetphotocurator_shortcut.ps1"
pause
