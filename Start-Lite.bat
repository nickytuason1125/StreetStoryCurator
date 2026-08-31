@echo off
REM Lumara Lite — low-RAM / no-GPU profile.
REM Pins the small encoder and applies lite defaults before launching the server.
REM See LITE_MODE.md for what changes.

set SIGLIP_TIER=low
set LUMARA_LITE=1

echo Starting Lumara (Lite mode)...
python server.py %*