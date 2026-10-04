"""Person detection and subject sharpness must decode a chunk IN PARALLEL.

2026-10-04, measured on a 4,300-photo card (61 MP Sony JPEGs + ARWs): the
IQA phase spent ~31 s/200 photos in person detection and ~45 s/200 in subject
sharpness, while D-FINE itself costs ~19 ms/img on the GPU. Both loops decoded
one photo at a time (100-270 ms per 61 MP JPEG). Decoding is pure, so a thread
pool gives the same pixels in the same order — identical grades, all cores.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import dfine_detector  # noqa: E402
import subject_sharpness  # noqa: E402


class _Probe:
    """A fake decoder that records peak concurrency."""

    def __init__(self, result):
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.result = result

    def __call__(self, path):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        time.sleep(0.05)
        with self.lock:
            self.live -= 1
        return self.result(path)


def test_sharpness_decodes_chunk_concurrently_and_in_order(monkeypatch):
    probe = _Probe(lambda p: np.full((64, 64, 3), int(p), np.uint8))
    monkeypatch.setattr(subject_sharpness, "_decode", probe)
    seen = []

    def fake_detect(items, conf=0.5):
        seen.extend(k for k, _ in items)
        return {k: [] for k, _ in items}

    monkeypatch.setattr(dfine_detector, "detect_subjects_from_arrays", fake_detect)
    paths = [str(i) for i in range(16)]
    out = subject_sharpness.score_paths(paths)
    assert probe.peak >= 4, f"decoded serially (peak concurrency {probe.peak})"
    assert seen == paths, "decode order changed — detection batches would differ"
    assert set(out) == set(paths)


def test_person_detection_decodes_chunk_concurrently(monkeypatch):
    from PIL import Image

    probe = _Probe(lambda p: Image.new("RGB", (64, 64)))
    monkeypatch.setattr(dfine_detector, "_open_rgb", probe)
    batches = []

    class _Proc:
        def __call__(self, images, return_tensors):
            batches.append(len(images))
            raise RuntimeError("stop after decode")   # the chunk is caught and left empty

    monkeypatch.setattr(dfine_detector, "_load", lambda: (object(), _Proc()))
    dfine_detector.detect_persons([f"{i}.jpg" for i in range(16)])
    assert probe.peak >= 4, f"decoded serially (peak concurrency {probe.peak})"
    assert sum(batches) == 16


def test_rawpy_is_never_entered_concurrently(monkeypatch):
    """LibRaw is not thread-safe (0xC0000005 history): the thread-pooled
    decoders must still reach rawpy one call at a time."""
    import types
    import raw_support

    probe = _Probe(lambda p: None)

    class _Raw:
        def __init__(self, path): pass
        def __enter__(self):
            probe(None)            # records concurrency inside the libraw section
            return self
        def __exit__(self, *a): return False
        def extract_thumb(self): raise RuntimeError("no preview")

    monkeypatch.setitem(sys.modules, "rawpy", types.SimpleNamespace(imread=_Raw))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: raw_support.extract_embedded_preview(f"{i}.arw"), range(16)))
    assert probe.peak == 1, f"rawpy entered by {probe.peak} threads at once"


def test_one_lock_whichever_name_imported_it():
    import raw_support
    sys.path.insert(0, str(_ROOT))
    import src.raw_support as alias
    assert alias.RAWPY_LOCK is raw_support.RAWPY_LOCK


# ── One detector pass for routing AND subject sharpness (2026-10-04) ─────────

def _fake_detector(monkeypatch, sizes_seen):
    import torch

    class _Model:
        config = type("C", (), {"id2label": {0: "person", 1: "bicycle"}})()
        def __call__(self, **kw):
            return "out"

    class _Proc:
        def __call__(self, images, return_tensors):
            sizes_seen.extend(im.size for im in images)
            class _In(dict):
                def to(self, _d): return self
            return _In()
        def post_process_object_detection(self, out, target_sizes, threshold):
            res = []
            for _ in range(len(target_sizes)):
                dets = [(0.9, 0, [0, 0, 10, 10]), (0.52, 0, [0, 0, 5, 5]), (0.6, 1, [1, 1, 4, 4]),
                        (0.3, 1, [0, 0, 2, 2])]
                keep = [d for d in dets if d[0] >= threshold]
                res.append({"scores": torch.tensor([d[0] for d in keep]),
                            "labels": torch.tensor([d[1] for d in keep]),
                            "boxes": torch.tensor([d[2] for d in keep], dtype=torch.float32)})
            return res

    monkeypatch.setattr(dfine_detector, "_load", lambda: (_Model(), _Proc()))
    monkeypatch.setattr(dfine_detector, "_person_id", 0)


def test_person_pass_caches_every_subject_for_sharpness(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_DFINE_PRESHRINK", "1")
    monkeypatch.setenv("FIRSTCUT_DFINE_SHARED_PASS", "1")
    from PIL import Image
    sizes = []
    _fake_detector(monkeypatch, sizes)
    monkeypatch.setattr(dfine_detector, "_open_rgb", lambda p: Image.new("RGB", (1616, 1080)))
    dfine_detector._SUBJECT_CACHE.clear()
    out = dfine_detector.detect_persons(["a.arw"], conf=0.55)
    assert [round(d["conf"], 2) for d in out["a.arw"]] == [0.9]          # persons >= 0.55 only
    cached = dfine_detector.cached_subjects("a.arw")
    assert sorted((d["label"], round(d["conf"], 2)) for d in cached) == \
        [("bicycle", 0.6), ("person", 0.52), ("person", 0.9)]             # all classes >= 0.5
    assert all(max(s) <= 640 for s in sizes), f"detector fed full-size frames: {sizes}"


def test_sharpness_reuses_cached_detections(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_DFINE_PRESHRINK", "1")
    monkeypatch.setenv("FIRSTCUT_DFINE_SHARED_PASS", "1")
    calls = []
    monkeypatch.setattr(subject_sharpness, "_decode", lambda p: np.zeros((64, 64, 3), np.uint8))
    monkeypatch.setattr(dfine_detector, "detect_subjects_from_arrays",
                        lambda items, conf=0.5: calls.append([k for k, _ in items]) or {k: [] for k, _ in items})
    dfine_detector._SUBJECT_CACHE.clear()
    dfine_detector._SUBJECT_CACHE["a"] = [{"bbox": [0.1, 0.1, 0.5, 0.5], "conf": 0.9, "label": "person"}]
    out = subject_sharpness.score_paths(["a", "b"])
    assert calls == [["b"]], "re-detected a photo the person pass already covered"
    assert out["a"]["labels"] == ["person"]


def test_cached_only_sharpness_never_touches_the_detector(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_DFINE_PRESHRINK", "1")
    monkeypatch.setenv("FIRSTCUT_DFINE_SHARED_PASS", "1")
    """The overlap thread in iqa_worker must stay off the GPU: it measures only
    photos the person pass already boxed and hands the rest back."""
    monkeypatch.setattr(subject_sharpness, "_decode", lambda p: np.zeros((64, 64, 3), np.uint8))
    monkeypatch.setattr(dfine_detector, "detect_subjects_from_arrays",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU from side thread")))
    dfine_detector._SUBJECT_CACHE.clear()
    dfine_detector._SUBJECT_CACHE["a"] = [{"bbox": [0.1, 0.1, 0.5, 0.5], "conf": 0.9, "label": "dog"}]
    out, misses = subject_sharpness.score_paths_cached(["a", "b"])
    assert set(out) == {"a"} and misses == ["b"]
    assert out["a"]["living"] is True


def test_person_pass_signals_completion(monkeypatch):
    from PIL import Image
    _fake_detector(monkeypatch, [])
    monkeypatch.setattr(dfine_detector, "_open_rgb", lambda p: Image.new("RGB", (100, 80)))
    dfine_detector.PERSON_PASS_DONE.clear()
    dfine_detector.detect_persons(["x.jpg"])
    assert dfine_detector.PERSON_PASS_DONE.is_set()


def test_drift_prone_speedups_are_off_by_default(monkeypatch):
    """Grades must not drift for speed (2026-10-04, 600 ARWs): skipping the
    LANCZOS shrink moved 39 smeared flags; pre-shrink / shared detector pass
    moved 1-2. All three stay opt-in."""
    for k in ("FIRSTCUT_DFINE_PRESHRINK", "FIRSTCUT_DFINE_SHARED_PASS",
              "FIRSTCUT_SHARP_NEAR_RESIZE_SKIP"):
        monkeypatch.delenv(k, raising=False)
    from PIL import Image
    big = Image.new("RGB", (1616, 1080))
    assert dfine_detector._model_input(big) is big
    calls = []
    monkeypatch.setattr(subject_sharpness, "_decode", lambda p: np.zeros((64, 64, 3), np.uint8))
    monkeypatch.setattr(dfine_detector, "detect_subjects_from_arrays",
                        lambda items, conf=0.5: calls.append(len(items)) or {k: [] for k, _ in items})
    dfine_detector._SUBJECT_CACHE["a"] = [{"bbox": [0, 0, 1, 1], "conf": 0.9, "label": "person"}]
    subject_sharpness.score_paths(["a"])
    assert calls == [1], "shared detector pass used without opt-in"
