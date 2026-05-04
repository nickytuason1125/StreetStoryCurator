@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  build-backend.bat  —  Build the Python FastAPI backend with PyInstaller
REM  Run from the project root:  build-backend.bat
REM ─────────────────────────────────────────────────────────────────────────────

echo [build-backend] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate venv. Make sure venv\ exists.
    exit /b 1
)

echo [build-backend] Running PyInstaller...
pyinstaller ^
    --onedir ^
    --windowed ^
    --name curator-api ^
    --paths src ^
    --hidden-import onnxruntime ^
    --hidden-import onnxruntime.capi._pybind_state ^
    --hidden-import cv2 ^
    --hidden-import PIL._tkinter_finder ^
    --hidden-import uvicorn.logging ^
    --hidden-import uvicorn.loops ^
    --hidden-import uvicorn.loops.auto ^
    --hidden-import uvicorn.protocols ^
    --hidden-import uvicorn.protocols.http ^
    --hidden-import uvicorn.protocols.http.auto ^
    --hidden-import uvicorn.protocols.http.h11_impl ^
    --hidden-import uvicorn.protocols.websockets ^
    --hidden-import uvicorn.protocols.websockets.auto ^
    --hidden-import uvicorn.lifespan ^
    --hidden-import uvicorn.lifespan.on ^
    --hidden-import anyio._backends._asyncio ^
    --hidden-import starlette.routing ^
    --hidden-import multipart ^
    --add-data "src;src" ^
    --add-data "frontend/dist;frontend/dist" ^
    server.py

if errorlevel 1 (
    echo ERROR: PyInstaller failed.
    exit /b 1
)

echo [build-backend] Moving bundle to Tauri binaries folder...
if not exist "frontend\src-tauri\binaries" mkdir "frontend\src-tauri\binaries"

REM Remove old bundle if present
if exist "frontend\src-tauri\binaries\curator-api-x86_64-pc-windows-msvc" (
    rmdir /s /q "frontend\src-tauri\binaries\curator-api-x86_64-pc-windows-msvc"
)

move "dist\curator-api" "frontend\src-tauri\binaries\curator-api-x86_64-pc-windows-msvc"
if errorlevel 1 (
    echo ERROR: Could not move dist\curator-api.
    exit /b 1
)

echo [build-backend] Copying models alongside the bundle...
xcopy /E /I /Y models "frontend\src-tauri\binaries\curator-api-x86_64-pc-windows-msvc\models"
if errorlevel 1 (
    echo WARNING: xcopy of models\ failed or models\ does not exist.
)

echo.
echo [build-backend] SUCCESS!
echo   Bundle: frontend\src-tauri\binaries\curator-api-x86_64-pc-windows-msvc\
echo.
echo   Next steps:
echo     1. cd frontend ^&^& npm run build   (build the React SPA)
echo     2. npm run tauri build             (package the Tauri installer)
