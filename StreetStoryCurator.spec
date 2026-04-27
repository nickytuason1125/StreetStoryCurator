# -*- mode: python ; coding: utf-8 -*-
#
# StreetStoryCurator.spec
#
# Usage (run from the project root with venv active):
#   Windows:  pyinstaller StreetStoryCurator.spec
#   Mac:      pyinstaller StreetStoryCurator.spec
#
# Output lands in dist/StreetStoryCurator/
# The folder is self-contained — zip it and distribute.

import sys
from pathlib import Path

ROOT = Path(".").resolve()
SRC  = str(ROOT / "src")

block_cipher = None

a = Analysis(
    [str(ROOT / "src" / "local_launcher.py")],
    pathex=[SRC, str(ROOT)],
    binaries=[],
    datas=[
        # Pre-built React frontend
        (str(ROOT / "frontend" / "dist"),  "frontend/dist"),
        # ML models
        (str(ROOT / "models"),             "models"),
        # Python source modules (server.py lives at root)
        (str(ROOT / "server.py"),          "."),
        (str(ROOT / "src"),                "src"),
        # Pacing presets JSON
        (str(ROOT / "src" / "pacing_presets.json"), "src"),
        # App icon
        (str(ROOT / "icon.ico"),           "."),
    ],
    hiddenimports=[
        # ── App modules ───────────────────────────────────────────────
        "server",
        "lightweight_analyzer",
        "sequence_engine",
        "niche_engine",
        "niche_classifier",
        "vlm_niche_detector",
        "editorial_renderer",
        "engine_utils",
        "reference_bank",
        "model_loader",
        "fast_io",

        # ── FastAPI / ASGI stack ──────────────────────────────────────
        "fastapi", "fastapi.middleware.cors",
        "fastapi.staticfiles", "fastapi.responses",
        "starlette", "starlette.staticfiles",
        "uvicorn", "uvicorn.logging",
        "uvicorn.loops", "uvicorn.loops.auto",
        "uvicorn.protocols", "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan", "uvicorn.lifespan.on",
        "anyio", "httptools", "watchfiles", "websockets",

        # ── pywebview ─────────────────────────────────────────────────
        "webview",

        # ── Vision / ML ───────────────────────────────────────────────
        "cv2", "cv2.dnn",
        "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
        "numpy", "scipy", "scipy.spatial.distance",
        "sklearn", "sklearn.cluster", "sklearn.metrics.pairwise",
        "onnxruntime",
        "torch", "torchvision",
        "clip",

        # ── Misc ──────────────────────────────────────────────────────
        "fpdf", "fpdf.fpdf",
        "piexif",
        "tqdm",
        "joblib",
        "ftfy",
        "pydantic", "pydantic.v1",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "gradio", "gradio_client",
        "matplotlib", "IPython", "notebook",
        "pytest", "tkinter._test", "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StreetStoryCurator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "icon.ico"),
    # argv_emulation enables Mac app bundle to receive dropped files
    argv_emulation=sys.platform == "darwin",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="StreetStoryCurator",
)

# On Mac, wrap in an .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="StreetStoryCurator.app",
        icon=str(ROOT / "icon.ico"),
        bundle_identifier="com.streetstorycurator.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": "Camera access for photo import",
        },
    )
