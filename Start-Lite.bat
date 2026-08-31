@echo off
REM Cullwise Lite — low-RAM / no-GPU profile.
REM Pins the small encoder and applies lite defaults before launching the server.
REM See LITE_MODE.md for what changes.

set SIGLIP_TIER=low
set CULLWISE_LITE=1

echo Starting Cullwise (Lite mode)...
python server.py %*