"""
niche_classifier.py — ONNX-backed niche classifier for Street Story Curator.

Uses DINOv2 (384-dim) embeddings already computed during grading.
Prototype anchors are built from the graded-photo cache — no text encoder
or separate model download required.

Design decisions vs. the original NicheClassifier:
  • _encode_text removed — there is no CLIP/SigLIP text encoder in this stack.
    Visual prototype averaging is more accurate on real photo batches anyway.
  • classify() accepts a pre-computed embedding, not a raw image tensor,
    so the existing ONNX session is never duplicated.
  • classify_image() is kept for callers that only have a raw tensor.
  • Softmax uses TEMPERATURE = 8.0 so well-matched niches score clearly
    above noise (raw cosine ≈ 0.6 → prob > 0.85 with T=8; without T the
    spread collapses to near-uniform at 0.08 per class).
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional


# ── Niche metadata ────────────────────────────────────────────────────────────
# Kept for documentation; NOT used for encoding.
NICHE_LABELS = [
    "Portrait/People", "Street/Urban", "Travel/Tourism", "Architecture",
    "Real Estate", "Food/Culinary", "Product/Commercial", "Night/Nocturnal",
    "Landscape/Nature", "Wedding/Event", "Sports/Action", "Macro/Detail",
    "General/Mixed",
]

# Dimension-based seed scores used to identify prototype examples from cache.
# Each niche is seeded from cached photos that score highest on these signals.
# (c=comp, t=tech, h=human, l=light, a=auth — key-set proxies)
_NICHE_SEED_FNS: dict[str, callable] = {
    # Mirrors the gate + score logic in _detect_top_niches so build_anchors
    # seeds each prototype from the same exemplar population.
    # Returns 0.0 when hard gate fails so mismatched photos don't dilute anchors.
    "Portrait/People":    lambda c,t,h,l,a: h*0.48 + t*0.30 + l*0.22
                              if h > 0.48 and t > 0.38 and t >= a - 0.12 else 0.0,
    "Street/Urban":       lambda c,t,h,l,a: a*0.46 + h*0.30 + c*0.24
                              if a > 0.36 and h > 0.25 and a >= t - 0.08 else 0.0,
    "Travel/Tourism":     lambda c,t,h,l,a: h*0.34 + a*0.34 + l*0.32
                              if h > 0.32 and a > 0.36 and l > 0.33 else 0.0,
    "Architecture":       lambda c,t,h,l,a: c*0.56 + t*0.26 + l*0.18
                              if c > 0.55 and h < 0.28 and c >= l else 0.0,
    "Real Estate":        lambda c,t,h,l,a: c*0.38 + l*0.44 + t*0.18
                              if c > 0.48 and l > 0.48 and h < 0.22 else 0.0,
    "Food/Culinary":      lambda c,t,h,l,a: t*0.38 + l*0.40 + c*0.22
                              if t > 0.50 and l > 0.48 and h < 0.22 else 0.0,
    "Product/Commercial": lambda c,t,h,l,a: t*0.52 + c*0.34 + l*0.14
                              if t > 0.70 and c > 0.58 and h < 0.18 else 0.0,
    "Night/Nocturnal":    lambda c,t,h,l,a: a*0.52 + c*0.28 + (1.0 - l)*0.20
                              if l < 0.38 and a > 0.46 else 0.0,
    "Landscape/Nature":   lambda c,t,h,l,a: l*0.56 + c*0.26 + a*0.18
                              if l > 0.38 and h < 0.28 and l >= c else 0.0,
    "Wedding/Event":      lambda c,t,h,l,a: h*0.48 + l*0.28 + c*0.24
                              if h > 0.58 and l > 0.35 else 0.0,
    "Sports/Action":      lambda c,t,h,l,a: t*0.50 + a*0.28 + h*0.22
                              if t > 0.62 and h > 0.36 else 0.0,
    "Macro/Detail":       lambda c,t,h,l,a: c*0.46 + t*0.42 + (1.0 - h)*0.12
                              if c > 0.60 and t > 0.56 and h < 0.18 else 0.0,
    "General/Mixed":      lambda c,t,h,l,a: 0.28,
}


class NicheClassifier:
    """
    Classifies a photo's niche using cosine similarity against per-niche
    prototype embeddings.

    Typical usage
    -------------
    1. After grading:
        clf = NicheClassifier(onnx_path="models/onnx/composition.onnx")
        clf.build_anchors(analyzer.cache, analyzer._COMP_KEYS, ...)
    2. Per-photo:
        probs = clf.classify(analyzer.cache[path]["embedding"])
        top, conf = clf.top_niche(analyzer.cache[path]["embedding"])
    """

    TEMPERATURE = 8.0     # softmax sharpness — higher → more decisive
    MIN_SAMPLES = 3       # minimum photos per niche to form a prototype

    def __init__(self, onnx_path: Optional[str] = None) -> None:
        self._session: Optional[ort.InferenceSession] = None
        if onnx_path and Path(onnx_path).exists():
            self._session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
        self._anchors:      dict[str, np.ndarray] = {}
        self._anchor_sizes: dict[str, int]        = {}

    # ── Anchor building ───────────────────────────────────────────────────────

    def build_anchors(
        self,
        cache: dict,
        comp_keys:  frozenset,
        tech_keys:  frozenset,
        human_keys: frozenset,
        light_keys: frozenset,
        auth_keys:  frozenset,
        min_samples: int = MIN_SAMPLES,
    ) -> int:
        """
        Build per-niche prototype vectors from the graded-photo cache.

        For each photo, computes a seed score against every niche function,
        assigns the photo to its best-matching niche, then averages the
        embeddings of the top-N photos per niche to form a prototype.

        Returns the number of niches for which a prototype was built.
        """
        def _dv(b, keys):
            return next((v for k, v in b.items() if k in keys), 0.0)

        # Gather (seed_score_per_niche, embedding) for every cached photo
        buckets: dict[str, list[tuple[float, np.ndarray]]] = {n: [] for n in _NICHE_SEED_FNS}

        for path, data in cache.items():
            emb = data.get("embedding")
            b   = data.get("breakdown", {})
            if not emb or not b:
                continue
            c = _dv(b, comp_keys)
            t = _dv(b, tech_keys)
            h = _dv(b, human_keys)
            l = _dv(b, light_keys)
            a = _dv(b, auth_keys)

            best_niche, best_score = "General/Mixed", 0.0
            for niche, fn in _NICHE_SEED_FNS.items():
                s = fn(c, t, h, l, a)
                if s > best_score:
                    best_score, best_niche = s, niche

            if best_score > 0.40:
                buckets[best_niche].append((best_score, np.array(emb, dtype=np.float32)))

        built = 0
        for niche, items in buckets.items():
            if len(items) < min_samples:
                continue
            # Weight average by seed score so the cleanest examples dominate
            items.sort(key=lambda x: x[0], reverse=True)
            top_items = items[:max(min_samples, len(items) // 3)]
            weights   = np.array([s for s, _ in top_items], dtype=np.float32)
            weights  /= weights.sum()
            prototype = sum(w * e for w, (_, e) in zip(weights, top_items))
            norm      = np.linalg.norm(prototype)
            self._anchors[niche]      = prototype / (norm + 1e-9)
            self._anchor_sizes[niche] = len(items)
            built += 1

        return built

    # ── Classification ────────────────────────────────────────────────────────

    def classify(self, img_emb) -> dict[str, float]:
        """
        Classify a pre-computed embedding (list or 1-D ndarray).
        Returns niche → probability dict sorted descending by probability.
        Falls back to {"General/Mixed": 1.0} if no anchors have been built.
        """
        if not self._anchors:
            return {"General/Mixed": 1.0}

        arr  = np.array(img_emb, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm < 1e-9:
            return {"General/Mixed": 1.0}
        arr = arr / norm

        # Raw cosine similarities
        names  = list(self._anchors.keys())
        raw    = np.array([float(np.dot(arr, self._anchors[n])) for n in names],
                          dtype=np.float32)

        # Temperature-scaled softmax for stable, discriminative probabilities
        scaled = raw * self.TEMPERATURE
        scaled -= scaled.max()                       # numerical stability
        exp    = np.exp(scaled)
        probs  = (exp / (exp.sum() + 1e-9)).tolist()

        return dict(sorted(zip(names, probs), key=lambda x: x[1], reverse=True))

    def classify_image(self, image_tensor: np.ndarray) -> dict[str, float]:
        """
        Run ONNX inference on a (C,H,W) float32 tensor, then classify.
        Only needed when an embedding is not already cached.
        """
        if self._session is None:
            return {"General/Mixed": 1.0}
        inp = self._session.get_inputs()[0].name
        out = self._session.get_outputs()[0].name
        emb = self._session.run(
            [out], {inp: image_tensor[np.newaxis].astype(np.float32)}
        )[0].flatten()
        return self.classify(emb)

    def top_niche(self, img_emb) -> tuple[str, float]:
        """Returns (niche_name, probability) for the highest-scoring niche."""
        probs = self.classify(img_emb)
        best  = next(iter(probs))            # already sorted descending
        return best, probs[best]

    def batch_classify(self, embeddings: list) -> list[dict[str, float]]:
        """Classify multiple embeddings efficiently (vectorised cosine)."""
        if not self._anchors or not embeddings:
            return [{"General/Mixed": 1.0}] * len(embeddings)

        mat    = np.array(embeddings, dtype=np.float32)
        norms  = np.linalg.norm(mat, axis=1, keepdims=True)
        mat   /= (norms + 1e-9)

        names     = list(self._anchors.keys())
        anchor_mat = np.stack([self._anchors[n] for n in names])  # (K, D)
        sims       = mat @ anchor_mat.T                           # (N, K)

        scaled = sims * self.TEMPERATURE
        scaled -= scaled.max(axis=1, keepdims=True)
        exp    = np.exp(scaled)
        probs  = exp / (exp.sum(axis=1, keepdims=True) + 1e-9)   # (N, K)

        return [
            dict(sorted(zip(names, row.tolist()), key=lambda x: x[1], reverse=True))
            for row in probs
        ]

    @property
    def anchor_info(self) -> dict[str, int]:
        """Returns {niche: n_photos_used} for diagnostics."""
        return dict(self._anchor_sizes)
