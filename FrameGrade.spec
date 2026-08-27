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
        # ML models — enumerated, NOT the whole directory.
        #
        # models/ is 22 GB and bundling it whole made the installer 22 GB. What
        # ships is what a first cull actually loads; the rest is either an
        # accelerator for hardware not every user has, a fallback for a
        # checkpoint that already ships, or an opt-in feature. All of it stays
        # on disk — this list decides what goes in the BUNDLE, nothing is
        # deleted, and model_registry already treats every one of them as
        # optional ("None of these is required to grade").
        #
        # Deliberately NOT bundled, with the reason each:
        #   siglip2/        7.1 GB  open_clip fp32 fallback for a checkpoint we
        #                           already ship, measured 10.3 GB peak - more
        #                           RAM than the path it backs up. Reachable
        #                           only via SIGLIP_ENC_USE_OC=1.
        #   onnx/           3.5 GB  high-tier + GPU only (onnx_enabled() is
        #                           false on CPU: 6.5-7.6 GB, 11-14 s/img), so
        #                           every CPU user carries it for nothing.
        #   *Qwen*.gguf     3.1 GB  Deep Grade is already opt-in.
        #   LFM2.5-VL*.gguf 698 MB  ditto.
        #   _quarantine/    702 MB  quarantined artefacts, not a live path.
        #
        # 22 GB -> ~6.6 GB. The three encoder tiers all stay, so the RAM
        # degradation ladder is untouched: a weak machine still has somewhere
        # to fall back to, which is the one thing that must not become an
        # optional download.
        (str(ROOT / "models" / "siglip2_hf_fp16"),   "models/siglip2_hf_fp16"),
        (str(ROOT / "models" / "siglip2_L_hf_fp16"), "models/siglip2_L_hf_fp16"),
        (str(ROOT / "models" / "siglip2_B_hf_fp16"), "models/siglip2_B_hf_fp16"),
        (str(ROOT / "models" / "vision_probe"),      "models/vision_probe"),
        (str(ROOT / "models" / "dfine_nano"),        "models/dfine_nano"),
        (str(ROOT / "models" / "dpo_adapter"),       "models/dpo_adapter"),
        (str(ROOT / "models" / "ViT-B-32.pt"),       "models"),
        (str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "models"),
        # Python source modules (server.py lives at root)
        (str(ROOT / "server.py"),          "."),
        # server.py is a thin launcher: `from server_impl import app`. The app
        # itself, and the routers it mounts, live at the ROOT rather than in
        # src/, so bundling src/ does not reach them. Missing these is not a
        # degraded build, it is ModuleNotFoundError before the first pixel.
        (str(ROOT / "server_impl.py"),     "."),
        (str(ROOT / "routers"),            "routers"),
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
        "server_impl",
        # routers.mount_all() imports these INSIDE the function, deliberately,
        # so the app module is fully initialised before they bind. That is also
        # exactly the pattern PyInstaller's static scan cannot follow, so every
        # one has to be named. tests/test_packaging_spec.py discovers routers/
        # from disk and fails if a new one is added without being listed here.
        "routers",
        "routers.system", "routers.library", "routers.grading",
        "routers.creative", "routers.sequence", "routers.export",
        "routers.extras", "routers.misc",
        # Imported lazily inside request handlers and main(), same reason.
        "catalog_store",
        "system_check",
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
