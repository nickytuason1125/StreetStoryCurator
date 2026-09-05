#!/bin/bash
# run_macos.sh — FirstCut bootstrap for macOS (Intel + Apple Silicon).
#
#   chmod +x run_macos.sh && ./run_macos.sh
#
# What it does: creates the venv, installs the machine-chosen builds the
# Windows installer normally picks (torch is MPS-capable on arm64 straight
# from PyPI), installs the pinned requirements (pywin32 is skipped by its
# sys_platform marker), builds the frontend if needed, starts the server,
# and opens the app window.
#
# Requirements: Python 3.10+, Node 18+ (only if frontend/src is present and
# dist/ is missing), CMake + Xcode CLT for llama-cpp-python (source build —
# the Windows prebuilt wheels do not exist for macOS).
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
VENV=venv

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found — install Python 3.10+ from python.org or brew" >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "[1/5] Creating venv…"
  "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "[2/5] Installing machine-chosen builds (torch MPS, onnxruntime CPU)…"
python -m pip install --upgrade pip
python -m pip install "torch==2.5.1" "torchvision==0.20.1"        # arm64 wheels ship MPS
python -m pip install "onnxruntime==1.22.0"                       # CPU tier; CoreML EP is a later upgrade
python -m pip install "llama-cpp-python==0.3.23" || \
  echo "WARN: llama-cpp-python failed — jury/critique need CMake + Xcode CLT"

echo "[3/5] Installing pinned requirements…"
python -m pip install -r requirements.txt

if [ ! -f frontend/dist/index.html ] && [ -d frontend/src ]; then
  echo "[4/5] Building frontend…"
  (cd frontend && npm install && npm run build)
fi

echo "[5/5] Starting server on http://127.0.0.1:8000 …"
python -m uvicorn server_impl:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
sleep 4
if command -v open >/dev/null 2>&1; then open "http://127.0.0.1:8000"; fi
echo "FirstCut is running (pid $SERVER_PID). Close this terminal or Ctrl-C to stop."
wait $SERVER_PID