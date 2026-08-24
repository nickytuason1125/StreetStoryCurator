# -*- mode: python ; coding: utf-8 -*-
#
# FrameGrade.spec
#
# Usage (run from the project root with venv active):
#   Windows:  pyinstaller FrameGrade.spec
#   Mac:      pyinstaller FrameGrade.spec
#
# Output lands in dist/FrameGrade/
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
        # Calibration fallback anchors. WITHOUT THIS a fresh install has no
        # anchors (cache/ is gitignored and never bundled), so _calibrate falls
        # back to batch-relative scoring — grading on a curve, where the same
        # photo is Weak or Strong depending on what it was culled alongside.
        (str(ROOT / "data" / "calibration_defaults.json"), "data"),
        # App icon
        (str(ROOT / "icon.ico"),           "."),
    ],
    hiddenimports=[
        # ── App modules ───────────────────────────────────────────────
        "server",
        # Imported lazily inside exif_reader, so PyInstaller's static scan
        # never sees it and the packaged build would return no RAW EXIF.
        "exifread",
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
    name="FrameGrade",
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
    name="FrameGrade",
)

# On Mac, wrap in an .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="FrameGrade.app",
        icon=str(ROOT / "icon.ico"),
        bundle_identifier="com.framegrade.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": "Camera access for photo import",
        },
    )
