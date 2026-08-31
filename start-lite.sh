#!/usr/bin/env bash
# Lumara Lite — low-RAM / no-GPU profile (macOS + Linux).
# Pins the small encoder and applies lite defaults before launching the server.
# See LITE_MODE.md for what changes.

export SIGLIP_TIER=low
export LUMARA_LITE=1

echo "Starting Lumara (Lite mode)..."
exec python3 server.py "$@"