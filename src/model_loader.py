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
}
_DOWNLOAD_LOCK = threading.Lock()


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

    if _SENTINEL.exists():
        ready = {k: "ready" for k in _DOWNLOAD_STATUS}
        _DOWNLOAD_STATUS.update(ready)
        return ready

    with _DOWNLOAD_LOCK:
        # Re-check after acquiring the lock (another thread may have finished).
        if _SENTINEL.exists():
            ready = {k: "ready" for k in _DOWNLOAD_STATUS}
            _DOWNLOAD_STATUS.update(ready)
            return ready

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
