"""Regression test for the `name '_p' is not defined` crash in
run_creative_direction (fixed 2026-09-19: the progress callback was
defined as `p` but called as `_p`, so the very first progress update
killed the whole styling run).

Stubs the GGUF agent so no models load, then drives the pipeline
through the subject-filter / peg / rerank / rule-set stages — the
exact call sites of `_p(...)` that previously raised NameError.
Run:  venv\\Scripts\\python.exe tests\\test_creative_progress_callback.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import types
import numpy as np
import creative_director as cd

calls: list[tuple[float, str]] = []
PROGRESS = lambda f, d: calls.append((f, d))          # noqa: E731


class _FakeAgent(types.ModuleType):
    """Replaces creative_director_agent: records that we reached the
    rule-set stage, then raises a sentinel to stop the pipeline before
    any real model work."""
    @staticmethod
    def generate_rule_set(style_prompt, rag_phrases=None):
        raise RuntimeError("STOP-TEST reached generate_rule_set")

    @staticmethod
    def generate_director_brief(style_prompt):
        raise RuntimeError("no agent in test")


sys.modules["creative_director_agent"] = _FakeAgent("creative_director_agent")

# Keep the text-semantic rerank cheap: no SigLIP text tower in the test.
cd._brief_vector = lambda s, M=None: np.zeros(
    M.shape[1] if M is not None else 1536, dtype=np.float32)
cd._embed_texts = lambda texts: np.stack(
    [np.zeros(1536, dtype=np.float32) for _ in texts])


def _run(label, embeddings):
    calls.clear()
    result = {}
    try:
        result = cd.run_creative_direction(
            strong_paths=[f"img_{i}.jpg" for i in range(8)],
            embeddings=embeddings,
            anchor_path="img_0.jpg",
            output_dir=str(ROOT / "output"),
            scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            aspect_scores_list=[None] * 8,
            n_target=4,
            style_prompt="vehicles only",        # hard subject filter → _p path
            progress=PROGRESS,
            mode="story",
        )
    except RuntimeError as e:
        if "STOP-TEST" not in str(e):
            raise
    except NameError as e:
        print(f"\nFAIL [{label}] — NameError still present: {e}")
        sys.exit(1)
    assert calls, f"[{label}] no progress callbacks fired"
    print(f"\nPASS [{label}] — {len(calls)} progress callbacks, no NameError")
    for f, d in calls:
        print(f"  {f:.2f}  {d[:70]}")


# Scenario 1: no embeddings → subject-filter `elif` branch (old crash line 1478)
_run("no-embeddings", [])

# Scenario 2: 1536-dim embeddings → subject-filter try branch + rerank _p
rng = np.random.default_rng(0)
_embs = [rng.normal(size=1536).astype(np.float32) for _ in range(8)]
_run("with-embeddings", _embs)

print("\nALL PASS — every previously-crashing _p call site executed.")
sys.exit(0)
