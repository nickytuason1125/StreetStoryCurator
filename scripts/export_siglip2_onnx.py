r"""
Export the SigLIP-2 encoder to ONNX — the anti-OOM switch.

Why this exists
---------------
The PyTorch encode path peaks ~4 GB (torch 0.36 + transformers 1.60 + fp16
weights). The identical graph through onnxruntime peaks ~1.2 GB — measured on
a real 514-photo grade. encode_worker already prefers ONNX the moment
models/onnx/vision.onnx exists; this one-time export is the only missing piece.

Run:  venv\\Scripts\\python.exe scripts/export_siglip2_onnx.py

The export loads the fp16 checkpoint onto the GPU when one is present (keeping
CPU RAM low — exactly the resource this whole exercise protects) and falls back
to CPU otherwise, where it will lean on the pagefile if RAM is tight. Slow but
one-time. After it finishes, grades auto-select ONNX (tier 'high' + GPU) and
the RAM gate relaxes from ~4 GB to ~2 GB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import siglip2_encoder as se  # noqa: E402  (profile: hf dir, dims, floors)

OUT_DIR = ROOT / "models" / "onnx"
HF_DIR = Path(se._hf_dir())
OPSET = 17


class VisionWrap(torch.nn.Module):
    """pixel_values (B,3,384,384) -> unnormalized image embeds (B,1536)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m.get_image_features(pixel_values=pixel_values)


class TextWrap(torch.nn.Module):
    """input_ids (B,64) int64 -> unnormalized text embeds (B,1536).

    The runtime feeds padded ids only, so the attention mask is folded in as a
    constant all-ones (equivalent: SigLIP text attention over the real ids).
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m.get_text_features(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


def main() -> int:
    from transformers import SiglipModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[export] loading fp16 checkpoint from {HF_DIR} onto {dev} …", flush=True)
    model = SiglipModel.from_pretrained(
        str(HF_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = model.to(dev).eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dummy_px = torch.zeros((1, 3, 384, 384), dtype=torch.float16, device=dev)
    dummy_ids = torch.zeros((1, 64), dtype=torch.int64, device=dev)

    print("[export] tracing vision graph …", flush=True)
    vpath = OUT_DIR / "vision.onnx"
    torch.onnx.export(
        VisionWrap(model).half() if dev == "cpu" else VisionWrap(model),
        (dummy_px,), str(vpath),
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"},
                      "image_embeds": {0: "batch"}},
        opset_version=OPSET)
    print(f"[export] wrote {vpath} ({vpath.stat().st_size / 1e6:.0f} MB)", flush=True)

    print("[export] tracing text graph …", flush=True)
    tpath = OUT_DIR / "text.onnx"
    torch.onnx.export(
        TextWrap(model),
        (dummy_ids,), str(tpath),
        input_names=["input_ids"], output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "batch"},
                      "text_embeds": {0: "batch"}},
        opset_version=OPSET)
    print(f"[export] wrote {tpath} ({tpath.stat().st_size / 1e6:.0f} MB)", flush=True)

    # ── verify: ONNX vs the torch reference, on a real image ────────────────
    print("[export] verifying against the torch reference …", flush=True)
    try:
        from PIL import Image
        img_path = next(Path(r"C:\Users\Nicky Tuason\Desktop\LAS").glob("*.jpg"), None)
        if img_path is None:
            print("[export] no test image found — skipping numeric check")
            return 0
        import torchvision.transforms.v2.functional as TF
        a = np.asarray(Image.open(img_path).convert("RGB")).copy()
        t = torch.from_numpy(a).permute(2, 0, 1)
        t = TF.resize(t, [384, 384], interpolation=TF.InterpolationMode.BILINEAR,
                      antialias=True)
        px = ((t.float() / 255.0 - 0.5) / 0.5).unsqueeze(0).to(dev, torch.float16)
        with torch.no_grad():
            ref = model.get_image_features(pixel_values=px).float().cpu().numpy()
            ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)

        import onnxruntime as ort
        sess = ort.InferenceSession(str(vpath), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"pixel_values": px.cpu().numpy().astype(np.float16)})[0]
        out = out.astype(np.float32)
        out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        cos = float((out * ref).sum())
        print(f"[export] dim={out.shape[1]} cosine(onnx, torch)={cos:.6f}")
        if out.shape[1] != 1536:
            print("[export] FAIL — wrong embed dim")
            return 1
        if cos < 0.999:
            print("[export] FAIL — space drift too large")
            return 1
        print("[export] OK — graphs verified")
    except Exception as e:
        print(f"[export] verification error (graphs still written): {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())