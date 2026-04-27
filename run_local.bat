@echo off
cd /d "%~dp0"

:: One-click local launcher for Street Story Curator
:: Uses the hidden VBScript to start the local pywebview app without a visible console.

start "" wscript.exe "%~dp0launch_hidden.vbs"
