"""Regression: grade_worker must slim each folder's gallery PER FOLDER.

2026-09-07 SD-upload OOM: a "Re-grade everything" over a ~100k-photo library
held every folder's gallery entries — each carrying a 1536-float embedding
list (~49 KB) — in combined_gallery until after the whole loop. Peak memory
scaled with the ENTIRE library and the runner died mid-run. The fix pops
embeddings (cull mode) inside the per-folder loop, so peak memory is one
folder's gallery, not the whole library's.

Run:  venv\\Scripts\\python.exe -m pytest tests\\test_grade_worker_slim.py -q
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import grade_worker  # noqa: E402


class _FakeQ:
    def __init__(self):
        self.msgs = []

    def put(self, m):
        self.msgs.append(m)


def test_embeddings_popped_before_next_folder_is_graded(monkeypatch, tmp_path):
    import catalog_store
    import grade_pipeline_v2

    # Record what run_v2 sees about the PREVIOUS folder's gallery when it is
    # called for the next one — the heart of the per-folder slimming fix.
    seen_at_call = []
    folders = [str(tmp_path / "f1"), str(tmp_path / "f2")]

    def fake_run_v2(fp, **kw):
        return {
            "gallery": [
                {"path": f"{fp}\\a.jpg", "grade": "Strong", "score": 0.9,
                 "embedding": [0.5] * 1536},
                {"path": f"{fp}\\b.jpg", "grade": "Weak", "score": 0.2,
                 "embedding": [0.5] * 1536},
            ],
            "total": 2,
        }

    _returned = []   # references to folder 1's gallery entries
    orig_append = seen_at_call.append

    def run_v2_spy(fp, **kw):
        if _returned:
            # second folder: folder 1's entries must already be slimmed
            orig_append(["emb" if "embedding" in p else "slim"
                         for p in _returned[0]])
        r = fake_run_v2(fp, **kw)
        _returned.append(r["gallery"])
        return r

    monkeypatch.setattr(grade_pipeline_v2, "run_v2", run_v2_spy)
    merge_calls = []
    monkeypatch.setattr(catalog_store, "merge_write",
                        lambda *a, **k: merge_calls.append(a and a[0] or k))

    q = _FakeQ()
    grade_worker.grade_worker_main(
        q, folders, "street", True, False,
        str(tmp_path / "catalog.json"), str(tmp_path),
        mogco_target=0,
    )

    errs = [m for m in q.msgs if "error" in m]
    assert not errs, f"worker errored: {errs}"

    done = [m for m in q.msgs if m.get("done")]
    assert done and done[0]["total"] == 4
    # The result payload carries no embeddings.
    assert all("embedding" not in p for p in done[0]["data"])

    # Folder 1's embeddings were already gone when folder 2 started grading —
    # the peak-memory fix (previously they survived until after the loop).
    assert seen_at_call == [["slim", "slim"]], (
        f"folder 1 gallery still held embeddings when folder 2 was graded: "
        f"{seen_at_call}")

    # The catalog merge never sees embeddings either.
    assert merge_calls and all(
        "embedding" not in p for batch in merge_calls for p in (batch or []))


def test_story_mode_keeps_embeddings(monkeypatch, tmp_path):
    """Story mode (mogco_target>0) needs embeddings — they must survive."""
    import catalog_store
    import grade_pipeline_v2

    monkeypatch.setattr(grade_pipeline_v2, "run_v2", lambda fp, **kw: {
        "gallery": [{"path": f"{fp}\\a.jpg", "grade": "Strong", "score": 0.9,
                     "embedding": [0.5] * 4}],
        "total": 1,
    })
    monkeypatch.setattr(catalog_store, "merge_write", lambda *a, **k: None)

    q = _FakeQ()
    grade_worker.grade_worker_main(
        q, [str(tmp_path / "f1")], "street", True, False,
        str(tmp_path / "catalog.json"), str(tmp_path),
        mogco_target=5,
    )
    errs = [m for m in q.msgs if "error" in m]
    assert not errs, f"worker errored: {errs}"
