#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  build-backend.sh  —  Build the Python FastAPI backend with PyInstaller
#  Run from the project root:  bash build-backend.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "[build-backend] Activating virtual environment..."
source venv/bin/activate

# Detect architecture: arm64 (Apple Silicon) maps to aarch64 for Tauri's triple
ARCH=$(python3 -c "import platform; m=platform.machine(); print('aarch64' if m=='arm64' else 'x86_64')")
echo "[build-backend] Detected arch: ${ARCH}"

echo "[build-backend] Running PyInstaller..."
pyinstaller \
    --onedir \
    --windowed \
    --name curator-api \
    --paths src \
    --hidden-import onnxruntime \
    --hidden-import onnxruntime.capi._pybind_state \
    --hidden-import cv2 \
    --hidden-import PIL._tkinter_finder \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.http.h11_impl \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    --hidden-import anyio._backends._asyncio \
    --hidden-import starlette.routing \
    --hidden-import multipart \
    --add-data "src:src" \
    --add-data "frontend/dist:frontend/dist" \
    server.py

echo "[build-backend] Moving bundle to Tauri binaries folder..."
mkdir -p "frontend/src-tauri/binaries"

DEST="frontend/src-tauri/binaries/curator-api-${ARCH}-apple-darwin"

# Remove old bundle if present
rm -rf "${DEST}"

mv "dist/curator-api" "${DEST}"

echo "[build-backend] Copying models alongside the bundle..."
if [ -d "models" ]; then
    cp -R models "${DEST}/models"
else
    echo "WARNING: models/ directory not found — skipping model copy."
fi

echo "[build-backend] Making binary executable..."
chmod +x "${DEST}/curator-api"

echo ""
echo "[build-backend] SUCCESS!"
echo "  Bundle: ${DEST}/"
echo ""
echo "  Next steps:"
echo "    1. cd frontend && npm run build   (build the React SPA)"
echo "    2. npm run tauri build            (package the Tauri app)"
