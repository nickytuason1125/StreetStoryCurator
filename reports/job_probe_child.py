import sys, threading
sys.path.insert(0, r"C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\src")
import torch, onnxruntime
print("torch+ort imported", flush=True)
from raw_support import RAW_EXTS, extract_embedded_preview
res = {}

def work():
    try:
        res["img"] = extract_embedded_preview(r"F:\DCIM\100MSDCF\DSC07551.ARW", "RGB")
    except Exception as e:
        res["err"] = f"{type(e).__name__}: {e}"

t = threading.Thread(target=work, daemon=True)
t.start()
t.join(30)
print("thread decode:", res.get("img").size if res.get("img") else res.get("err"), flush=True)

# also try the exact worker entry: _decode_with_timeout
import encode_worker as ew
img = ew._decode_with_timeout(r"F:\DCIM\100MSDCF\DSC07551.ARW")
print("_decode_with_timeout:", img.size if img else ("FAILED: " + str(ew_err if (ew_err := getattr(sys.modules[__name__], '_x', None)) else 'none')), flush=True)
print("last err:", ew._DECODE_LAST_ERR, flush=True)
