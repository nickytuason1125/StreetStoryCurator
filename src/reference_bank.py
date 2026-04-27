"""
Reference embedding bank.

Index a folder of award-winning / exemplar photos once.  At grade time,
compute cosine similarity to the nearest exemplars and use it to nudge
the raw score toward what the bank considers "quality work."

Only DINOv2 is used — the same ONNX session already running for composition.
No extra dependencies, no GPU required.

Usage
-----
bank = ReferenceBank()
bank.build(folder, ort_session, input_name)   # one-time indexing
sim  = bank.score(embedding)                  # per-image at grade time
"""

import threading
import numpy as np
import cv2
from pathlib import Path

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff",
         ".JPG", ".JPEG", ".PNG", ".WEBP"}

# ImageNet normalisation — same as in lightweight_analyzer._analyze
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class ReferenceBank:
    BANK_PATH = Path("models/reference_bank.npz")

    def __init__(self):
        self._embs: np.ndarray | None = None   # (N, 384) float32, L2-normalised
        self._lock = threading.Lock()
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if self.BANK_PATH.exists():
            try:
                data = np.load(str(self.BANK_PATH), allow_pickle=False)
                embs = data["embs"].astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                self._embs = embs / (norms + 1e-9)
            except Exception:
                self._embs = None

    @property
    def count(self) -> int:
        with self._lock:
            return 0 if self._embs is None else int(len(self._embs))

    # ── indexing ──────────────────────────────────────────────────────────────

    def build(self, folder, ort_session, input_name, progress=None) -> int:
        """
        Extract a DINOv2 CLS embedding for every image in `folder` (recursive)
        and persist a compressed bank to disk.

        Returns the number of images successfully indexed.
        `progress(done, total)` is called after each image if provided.
        """
        paths = [p for p in Path(folder).rglob("*") if p.suffix in _EXTS]
        if not paths:
            return 0

        embs = []
        for i, p in enumerate(paths):
            emb = _embed(str(p), ort_session, input_name)
            if emb is not None:
                embs.append(emb)
            if progress:
                progress(i + 1, len(paths))

        if not embs:
            return 0

        matrix = np.stack(embs).astype(np.float32)   # (N, 384)
        self.BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self.BANK_PATH), embs=matrix)

        with self._lock:
            self._embs = matrix

        return len(embs)

    def add(self, folder, ort_session, input_name, progress=None) -> tuple[int, int]:
        """
        Add new exemplars from `folder` to the existing bank without replacing it.
        Near-duplicate embeddings (cosine sim > 0.97 to anything already stored)
        are skipped so the bank doesn't bloat with burst-shot variants.

        Returns (added, skipped).
        """
        paths = [p for p in Path(folder).rglob("*") if p.suffix in _EXTS]
        if not paths:
            return 0, 0

        with self._lock:
            existing = self._embs.copy() if self._embs is not None else None

        new_embs: list[np.ndarray] = []
        skipped = 0

        for i, p in enumerate(paths):
            emb = _embed(str(p), ort_session, input_name)
            if emb is None:
                skipped += 1
                if progress:
                    progress(i + 1, len(paths))
                continue

            # Deduplicate: skip if too similar to anything already in the bank
            # or anything we've already added in this batch.
            is_dup = False
            check_against = []
            if existing is not None:
                check_against.append(existing)
            if new_embs:
                check_against.append(np.stack(new_embs))
            for mat in check_against:
                if float((mat @ emb).max()) > 0.97:
                    is_dup = True
                    break

            if is_dup:
                skipped += 1
            else:
                new_embs.append(emb)

            if progress:
                progress(i + 1, len(paths))

        if not new_embs:
            return 0, skipped

        parts = []
        if existing is not None:
            parts.append(existing)
        parts.append(np.stack(new_embs).astype(np.float32))
        matrix = np.concatenate(parts, axis=0)

        self.BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self.BANK_PATH), embs=matrix)

        with self._lock:
            self._embs = matrix

        return len(new_embs), skipped

    def clear(self):
        """Remove all exemplars from the bank."""
        with self._lock:
            self._embs = None
        if self.BANK_PATH.exists():
            self.BANK_PATH.unlink()

    # ── scoring ───────────────────────────────────────────────────────────────

    def score(self, embedding, top_k: int = 5) -> float | None:
        """
        Return the mean cosine similarity to the `top_k` nearest exemplars.

        `embedding` — L2-normalised (384,) float array (same as stored in cache).
        Returns None if the bank is empty so callers can skip the nudge entirely.

        Interpretation guide (typical DINOv2 within-genre similarities):
          ≥ 0.78  very close to exemplar work  →  +0.05 nudge
            0.65  neutral                      →   0.00 nudge
          ≤ 0.52  different genre/style        →  −0.03 nudge
        """
        with self._lock:
            if self._embs is None or len(self._embs) == 0:
                return None
            embs = self._embs

        emb  = np.asarray(embedding, dtype=np.float32)
        sims = embs @ emb                              # (N,) dot product of unit vecs
        k    = min(top_k, len(sims))
        return float(np.partition(sims, -k)[-k:].mean())


# ── module-level helper (not a method so it can be called without self) ───────

def _embed(path: str, ort_session, input_name) -> np.ndarray | None:
    """Run DINOv2 on a single image and return the L2-normalised CLS token."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb224 = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        inp    = ((np.transpose(rgb224, (2, 0, 1)) - _MEAN) / _STD)[np.newaxis, ...]
        out    = ort_session.run(None, {input_name: inp})[0]
        cls    = out[0, 0, :].astype(np.float32)       # CLS token (384,)
        norm   = np.linalg.norm(cls) + 1e-9
        return cls / norm
    except Exception:
        return None
