#!/usr/bin/env bash
# FrameGrade Lite — low-RAM / no-GPU profile (macOS + Linux).
# Pins the small encoder and applies lite defaults before launching the server.
# See LITE_MODE.md for what changes.

export SIGLIP_TIER=low
export FRAMEGRADE_LITE=1

echo "Starting FrameGrade (Lite mode)..."
exec python3 server.py "$@"