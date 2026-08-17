"""
Export the vision encoder to ONNX so inference does not need PyTorch.

Why
---
Profiling showed where the encoder's ~2.7 GB of system RAM actually goes:

    torch                 0.36 GB
    transformers          1.60 GB   <- the library, not the model
    CUDA context          0.11 GB
    model weights        ~0.01 GB   <- mmap'd straight to VRAM, effectively free
    inference working     0.40 GB

The MODEL is not the cost; the research framework around it is. onnxruntime-gpu
imports for 0.03 GB and runs the same graph, so moving inference off PyTorch is
the only change that can lower the floor without shrinking the model (which was
measured to cost grading accuracy — see the tier A/B).

This exports two graphs, matching how the pipeline actually uses the model:
    vision.onnx : pixel_values (B,3,384,384) -> image embedding (B,1536)
    text.onnx   : input_ids   (B,64)         -> text  embedding (B,1152->proj)

Run:  venv\\Scripts\\python.exe scripts/export_encoder_onnx.py
Nothing in the live pipeline changes; this only writes files under models/onnx/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
_ROOT = Path(__file__).resolve().parent.parent
_HF = _ROOT / "models" / "siglip2_hf_fp16"
_OUT = _ROOT / "models" / "onnx"
_OPSET = 17


def main() -> int:
    import torch
    from transformers import AutoModel

    if not (_HF / "config.json").exists():
        print(f"no checkpoint at {_HF}", file=sys.stderr)
        return 1
    _OUT.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    print(f"[export] loading checkpoint ({dev}, {dt})…", flush=True)
    model = AutoModel.from_pretrained(_HF, dtype=dt, low_cpu_mem_usage=True).to(dev).eval()

    class VisionTower(torch.nn.Module):
        """Wraps get_image_features so the ONNX graph is exactly what we call."""
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, pixel_values):
            e = self.m.get_image_features(pixel_values=pixel_values)
            return e / (e.norm(dim=-1, keepdim=True) + 1e-9)   # L2 norm baked in

    class TextTower(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids):
            e = self.m.get_text_features(input_ids=input_ids)
            return e / (e.norm(dim=-1, keepdim=True) + 1e-9)

    ok = True

    # ── vision ──────────────────────────────────────────────────────────────
    vpath = _OUT / "vision.onnx"
    try:
        dummy = torch.randn(1, 3, 384, 384, dtype=dt, device=dev)
        print("[export] tracing vision tower…", flush=True)
        torch.onnx.export(
            VisionTower(model), (dummy,), str(vpath),
            input_names=["pixel_values"], output_names=["embedding"],
            dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=_OPSET, do_constant_folding=True,
        )
        print(f"[export] vision.onnx  {vpath.stat().st_size/1e9:.2f} GB", flush=True)
    except Exception as exc:
        print(f"[export] vision export FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        ok = False

    # ── text ────────────────────────────────────────────────────────────────
    tpath = _OUT / "text.onnx"
    try:
        dummy_ids = torch.ones(1, 64, dtype=torch.int64, device=dev)
        print("[export] tracing text tower…", flush=True)
        torch.onnx.export(
            TextTower(model), (dummy_ids,), str(tpath),
            input_names=["input_ids"], output_names=["embedding"],
            dynamic_axes={"input_ids": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=_OPSET, do_constant_folding=True,
        )
        print(f"[export] text.onnx    {tpath.stat().st_size/1e9:.2f} GB", flush=True)
    except Exception as exc:
        print(f"[export] text export FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        ok = False

    # ── consolidate external weights ────────────────────────────────────────
    # torch.onnx.export spills initialisers as ONE FILE PER TENSOR when the model
    # exceeds protobuf's 2 GB limit — 656 loose files for this model. It loads,
    # but session creation took 13 s and the directory is unmanageable. Re-saving
    # with all_tensors_to_one_file collapses them into a single .weights blob.
    if ok:
        try:
            import onnx
            for name in ("vision", "text"):
                f = _OUT / f"{name}.onnx"
                if not f.exists():
                    continue
                print(f"[export] consolidating {name} weights…", flush=True)
                m = onnx.load(str(f), load_external_data=True)
                for stale in _OUT.glob(f"m.{name.replace('vision','vision_model')}*"):
                    try: stale.unlink()
                    except Exception: pass
                onnx.save_model(
                    m, str(f), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=f"{name}.weights",
                    size_threshold=1024, convert_attribute=False,
                )
                del m
                print(f"[export] {name}: graph {f.stat().st_size/1e6:.1f} MB + "
                      f"weights {(_OUT / (name + '.weights')).stat().st_size/1e9:.2f} GB",
                      flush=True)
            # sweep any remaining per-tensor files
            loose = [f for f in _OUT.iterdir()
                     if f.is_file() and f.suffix not in (".onnx", ".weights")]
            for f in loose:
                try: f.unlink()
                except Exception: pass
            if loose:
                print(f"[export] removed {len(loose)} loose tensor files", flush=True)
        except Exception as exc:
            print(f"[export] consolidation skipped ({type(exc).__name__}: {exc})",
                  file=sys.stderr)

    print("[export] done" if ok else "[export] finished with failures")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
