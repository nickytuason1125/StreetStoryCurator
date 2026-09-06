"""Reproduce the encoder in the REAL chain environment: detached pythonw WITH
suppress_console's global Popen patch active (as grade_worker has it)."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import suppress_console  # noqa: F401  — patches subprocess/multiprocessing/asyncio
sys.path.insert(0, str(ROOT / "src"))

out = []
try:
    import glob
    from siglip2_encoder import get_siglip2_encoder
    out.append("constructing encoder …")
    enc = get_siglip2_encoder()
    out.append("constructed OK")
    img = sorted(glob.glob(r"C:\Users\Nicky Tuason\Desktop\LAS\*.jpg"))[0]
    embs = enc.encode_images([img])
    out.append(f"encode OK: {embs.shape}")
    enc.unload()
except Exception:
    out.append("FAILED:\n" + traceback.format_exc())

Path(ROOT / "detached_encoder_result.txt").write_text("\n".join(out), encoding="utf-8")

