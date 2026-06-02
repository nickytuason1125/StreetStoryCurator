"""
Model Loader - Updated for SpecVLM Pipeline

Removed legacy models:
- NIMA (replaced by Q-Align in SpecVLM)
- MobileViT (replaced by SigLIP-2)
- DINOv2-small (replaced by SigLIP-2)

New models:
- SigLIP-2 ViT-g/14 NaFlex (FP8)
- DeepSeek-R1-Distill-Qwen-1.5B (Draft)
- DeepSeek-R1-Distill-Qwen-7B (Verify)

Auto-download: call ensure_all_models_downloaded() at server startup to
pre-fetch all weights. get_download_status() returns the live status dict.
"""

import threading
from pathlib import Path

MODEL_DIR = Path("models/onnx")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Sentinel file written after all models are confirmed present.
_SENTINEL = Path("models/.models_ready")

# Live status map mutated by ensure_all_models_downloaded().
_DOWNLOAD_STATUS: dict = {
    "siglip2": "pending",
    "deepseek_draft": "pending",
    "deepseek_verify": "pending",
    "topiq_nr": "pending",
    "qwen_vlm": "pending",
}
_DOWNLOAD_LOCK = threading.Lock()

# Expected size of Qwen2.5-VL-3B-Instruct safetensors + config (~6.8 GB).
_QWEN_EXPECTED_BYTES = 6_800_000_000


# ── TOPIQ NR weight prefetch ──────────────────────────────────────────────────

def _download_topiq_nr_if_needed() -> bool:
    """
    Pre-cache TOPIQ NR weights by instantiating the metric once on CPU.
    pyiqa downloads weights to its cache dir on first create_metric() call.
    Returns True if the model loads successfully, False otherwise.
    """
    import threading
    _result: list = [False]
    _err:    list = [None]

    def _worker():
        try:
            import pyiqa
            m = pyiqa.create_metric("topiq_nr", device="cpu")
            del m
            _result[0] = True
        except Exception as e:
            _err[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=300)
    if t.is_alive():
        print("[model_loader] topiq_nr download timed out after 300s")
        return False
    if _err[0]:
        print(f"[model_loader] topiq_nr download failed: {_err[0]}")
        return False
    print("[model_loader] topiq_nr weights cached OK")
    return True


# ── Qwen2.5-VL-3B-Instruct prefetch ──────────────────────────────────────────

def _download_qwen_if_needed() -> bool:
    """
    Download Qwen2.5-VL-3B-Instruct weights to models/qwen_vlm/ via
    huggingface_hub.snapshot_download().  A background watcher thread updates
    _DOWNLOAD_STATUS["qwen_vlm"] with "downloading:<pct>" every 3 s so the
    frontend can show a live progress bar without polling the filesystem itself.
    """
    import time as _time
    qwen_dir = Path("models/qwen_vlm")

    # Fast-path: weights already present.
    if any(qwen_dir.rglob("*.safetensors")):
        print("[model_loader] Qwen2.5-VL already cached — skipping download")
        return True

    try:
        from huggingface_hub import snapshot_download

        qwen_dir.mkdir(parents=True, exist_ok=True)
        _DOWNLOAD_STATUS["qwen_vlm"] = "downloading:0"

        # Watcher thread: update progress every 3 s based on downloaded bytes.
        stop_evt = threading.Event()

        def _watch():
            while not stop_evt.is_set():
                try:
                    total = sum(
                        f.stat().st_size
                        for f in qwen_dir.rglob("*")
                        if f.is_file()
                    )
                    pct = min(99, int(total / _QWEN_EXPECTED_BYTES * 100))
                    _DOWNLOAD_STATUS["qwen_vlm"] = f"downloading:{pct}"
                except Exception:
                    pass
                stop_evt.wait(3)

        watcher = threading.Thread(target=_watch, daemon=True, name="qwen-dl-watch")
        watcher.start()

        print("[model_loader] Downloading Qwen2.5-VL-3B-Instruct (~6.8 GB) …")
        snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
            local_dir=str(qwen_dir),
            local_dir_use_symlinks=False,
            # Skip non-PyTorch weight formats to save ~2 GB.
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        stop_evt.set()
        watcher.join(timeout=5)

        _DOWNLOAD_STATUS["qwen_vlm"] = "ready"
        print("[model_loader] Qwen2.5-VL download complete")
        return True

    except Exception as exc:
        print(f"[model_loader] Qwen2.5-VL download failed: {exc}")
        _DOWNLOAD_STATUS["qwen_vlm"] = "failed"
        return False


# ── Central download orchestration ────────────────────────────────────────────


def ensure_all_models_downloaded(progress_cb=None) -> dict:
    """
    Download every SpecVLM model that is not already cached locally.

    Safe to call from a background thread on server startup — uses a lock
    to prevent concurrent downloads and a sentinel file to skip work on
    subsequent launches.

    Args:
        progress_cb: Optional callable(status_dict) invoked after each model
                     finishes so callers can stream progress.

    Returns:
        Dict mapping model names to "ready" | "failed".
    """
    global _DOWNLOAD_STATUS

    # Sentinel is only valid if Qwen weights are also present — older sentinels
    # were written before Qwen was added to the task list.
    _qwen_cached = any(Path("models/qwen_vlm").rglob("*.safetensors"))
    if _SENTINEL.exists() and _qwen_cached:
        ready = {k: "ready" for k in _DOWNLOAD_STATUS}
        _DOWNLOAD_STATUS.update(ready)
        return ready

    with _DOWNLOAD_LOCK:
        _qwen_cached = any(Path("models/qwen_vlm").rglob("*.safetensors"))
        if _SENTINEL.exists() and _qwen_cached:
            ready = {k: "ready" for k in _DOWNLOAD_STATUS}
            _DOWNLOAD_STATUS.update(ready)
            return ready

        # Invalidate stale sentinel (written before Qwen was in the task list).
        if _SENTINEL.exists() and not _qwen_cached:
            try:
                _SENTINEL.unlink()
            except OSError:
                pass

        from siglip2_encoder import _download_siglip2_if_needed
        from deepseek_model import (
            _download_model_if_needed,
            DRAFT_MODEL_ID,
            VERIFY_MODEL_ID,
            MODEL_CACHE_DIR as _DS_CACHE,
        )

        _model_tasks = [
            ("siglip2", lambda: _download_siglip2_if_needed()),
            ("deepseek_draft", lambda: _download_model_if_needed(
                DRAFT_MODEL_ID,
                _DS_CACHE / "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B",
            )),
            ("deepseek_verify", lambda: _download_model_if_needed(
                VERIFY_MODEL_ID,
                _DS_CACHE / "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
            )),
            ("topiq_nr", _download_topiq_nr_if_needed),
            ("qwen_vlm",  _download_qwen_if_needed),
        ]

        for key, fn in _model_tasks:
            _DOWNLOAD_STATUS[key] = "downloading"
            if progress_cb:
                progress_cb(_DOWNLOAD_STATUS.copy())
            try:
                ok = fn()
                _DOWNLOAD_STATUS[key] = "ready" if ok else "failed"
            except Exception as exc:
                print(f"⚠️  {key} download error: {exc}")
                _DOWNLOAD_STATUS[key] = "failed"
            if progress_cb:
                progress_cb(_DOWNLOAD_STATUS.copy())

        # Write sentinel only when every model succeeded.
        if all(v == "ready" for v in _DOWNLOAD_STATUS.values()):
            _SENTINEL.parent.mkdir(parents=True, exist_ok=True)
            _SENTINEL.touch()

        return _DOWNLOAD_STATUS.copy()


def get_download_status() -> dict:
    """
    Return the current model download status without triggering any download.

    Returns:
        Dict mapping model names to "pending" | "downloading" | "ready" | "failed".
    """
    if _SENTINEL.exists():
        return {k: "ready" for k in _DOWNLOAD_STATUS}
    return _DOWNLOAD_STATUS.copy()


def get_sessions() -> dict:
    """
    Load ONNX inference sessions for the V1 LightweightStreetScorer.

    Returns a dict with keys "aesthetic", "composition", and optionally "nima".
    Raises RuntimeError if the required model files are missing.
    """
    import onnxruntime as ort

    _ONNX = Path("models/onnx")
    sessions: dict = {}

    aesthetic_path = _ONNX / "mobilevit_aesthetic.onnx"
    composition_path = _ONNX / "dinov2_small.onnx"
    nima_path = _ONNX / "nima.onnx"

    if not aesthetic_path.exists():
        raise RuntimeError(f"Missing model: {aesthetic_path}")
    if not composition_path.exists():
        raise RuntimeError(f"Missing model: {composition_path}")

    sessions["aesthetic"] = ort.InferenceSession(
        str(aesthetic_path), providers=["CPUExecutionProvider"]
    )
    sessions["composition"] = ort.InferenceSession(
        str(composition_path), providers=["CPUExecutionProvider"]
    )
    if nima_path.exists():
        sessions["nima"] = ort.InferenceSession(
            str(nima_path), providers=["CPUExecutionProvider"]
        )

    return sessions


def get_siglip2_encoder():
    """
    Get SigLIP-2 ViT-g/14 NaFlex encoder.
    
    Returns:
        SigLIP2Encoder instance
    """
    from siglip2_encoder import SigLIP2Encoder
    return SigLIP2Encoder()


def get_specvlm_pipeline():
    """
    Get SpecVLM pipeline with Draft-and-Verify models.
    
    Returns:
        SpecVLMPipeline instance
    """
    from specvlm_pipeline import SpecVLMPipeline
    return SpecVLMPipeline()


def get_deepseek_draft():
    """
    Get DeepSeek-R1-Distill-Qwen-1.5B draft model.
    
    Returns:
        DraftModel instance
    """
    from deepseek_model import DraftModel
    return DraftModel()


def get_deepseek_verify():
    """
    Get DeepSeek-R1-Distill-Qwen-7B verify model.
    
    Returns:
        VerifyModel instance
    """
    from deepseek_model import VerifyModel
    return VerifyModel()


def get_vram_manager():
    """
    Get VRAM manager for memory management.
    
    Returns:
        VRAMManager instance
    """
    from vram_manager import VRAMManager
    return VRAMManager()


def get_priority_gate():
    """
    Get Priority-Gate controller.
    
    Returns:
        PriorityGate instance
    """
    from priority_gate import PriorityGate
    return PriorityGate()


def get_nsga3_sequencer():
    """
    Get NSGA-III sequencer for photo optimization.
    
    Returns:
        NSGA3Sequencer instance
    """
    from nsga3_sequencer import NSGA3Sequencer
    return NSGA3Sequencer()


def get_lance_migration():
    """
    Get LanceDB migration utilities.
    
    Returns:
        Module with migration functions
    """
    import lance_migration
    return lance_migration


# ── Legacy Model Loading (Deprecated) ──────────────────────────────────────────


def get_legacy_sessions():
    """
    Get legacy ONNX model sessions (DEPRECATED).
    
    These models are replaced by SpecVLM pipeline:
    - DINOv2-small → SigLIP-2 ViT-g/14
    - MobileViT → SigLIP-2 ViT-g/14
    - NIMA → Q-Align (in SpecVLM)
    
    Returns:
        Empty dict - legacy models no longer used
    """
    print("⚠️  WARNING: get_legacy_sessions() is deprecated.")
    print("   Use get_siglip2_encoder() and get_specvlm_pipeline() instead.")
    return {}


def get_onnx_sessions():
    """
    Get ONNX model sessions (DEPRECATED).
    
    Returns:
        Empty dict - legacy models no longer used
    """
    print("⚠️  WARNING: get_onnx_sessions() is deprecated.")
    print("   Use get_siglip2_encoder() and get_specvlm_pipeline() instead.")
    return {}
