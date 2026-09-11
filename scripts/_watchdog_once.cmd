@echo off
rem One-shot detached watchdog launcher (used by tooling; not part of the app).
cd /d "C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator"
set FIRSTCUT_WATCHDOG_HOURS=6
start "" "venv\Scripts\pythonw.exe" scripts\run_watchdog.py
