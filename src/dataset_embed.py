"""Encode an arbitrary folder of images through the production SigLIP-2
encoder and cache the result. Shared by aadb_setup.py and unsplash_setup.py
so both one-time dataset scripts use the exact same embedding space grading
already uses — never a foreign feature space.

SigLIP2Encoder already isolates all GPU work in its own subprocess
(src/siglip2_encoder.py), so this module — and any script that imports it —
never touches torch.cuda directly.
"""
from pathlib import Path
import numpy as np

from siglip2_encoder import SigLIP2Encoder


def encode_folder(image_paths: list, out_npz: "Path | str", progress=None) -> np.ndarray:
    """Encode image_paths, write {paths, embeddings} to out_npz, return the
    (N, D) embedding array. Re-encodes every call — callers that want caching
    across runs should check out_npz.exists() themselves before calling."""
    out_npz = Path(out_npz)
    enc = SigLIP2Encoder(device="auto")
    embeddings = enc.encode_images(list(image_paths), progress=progress)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, paths=np.array(image_paths), embeddings=embeddings)
    return embeddings
