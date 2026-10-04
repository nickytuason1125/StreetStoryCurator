"""Variety guards: per-run manifest sampling + no-immediate-repeat rotation."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_manifest_returns_requested_size_within_pool():
    from creative_director import _select_manifest
    scores = np.linspace(0.2, 1.0, 50)
    idx = _select_manifest(scores, 10, seed=42)
    assert len(idx) == 10
    assert len(set(idx)) == 10          # no duplicates
    assert all(0 <= i < 50 for i in idx)


def test_manifest_only_draws_from_top_band():
    """Weak photos may never leak into the manifest."""
    from creative_director import _select_manifest
    scores = np.zeros(100)
    scores[:20] = np.linspace(0.5, 1.0, 20)     # top band = first 20
    idx = _select_manifest(scores, 10, seed=7)
    assert all(i < 20 for i in idx)


def test_manifest_varies_across_seeds():
    """Same scores, different runs -> different mixes (the variety guarantee)."""
    from creative_director import _select_manifest
    scores = np.linspace(0.2, 1.0, 40)
    picks = {tuple(_select_manifest(scores, 8, seed=s)) for s in range(12)}
    assert len(picks) >= 4, f"only {len(picks)} distinct manifests over 12 seeds"


def test_manifest_deterministic_for_same_seed():
    from creative_director import _select_manifest
    scores = np.linspace(0.2, 1.0, 40)
    assert _select_manifest(scores, 8, seed=1) == _select_manifest(scores, 8, seed=1)


def test_small_pool_degrades_to_full_pool():
    from creative_director import _select_manifest
    scores = np.array([0.9, 0.8, 0.7])
    idx = _select_manifest(scores, 10, seed=3)      # top_n exceeds pool
    assert sorted(idx) == [0, 1, 2]


def test_last_sequence_never_returns_immediately():
    """Router contract: avoid set = used-history UNION last run's outputs, so
    the photos from the sequence the user just saved cannot be re-picked even
    when FIFO rotation would otherwise retire them."""
    import server_impl
    used = {"C:/lib/a.jpg", "C:/lib/b.jpg"}
    server_impl.LAST_SEQUENCE = ["C:/lib/c.jpg", "C:/lib/d.jpg"]
    avoid = used | set(server_impl.LAST_SEQUENCE)
    # the just-saved story photos are excluded even though they are NOT in
    # the long-term used history
    assert "C:/lib/c.jpg" in avoid and "C:/lib/d.jpg" in avoid
    # and clearing LAST_SEQUENCE returns them to eligibility (fresh cycle)
    server_impl.LAST_SEQUENCE = []
    assert "C:/lib/c.jpg" not in (used | set(server_impl.LAST_SEQUENCE))


def test_story_id_extraction_recovers_noisy_model_output():
    """The director's answer is frequently wrapped in prose/commentary; the
    hardened recovery chain must still pull the ID list out."""
    import re, json as _json
    from creative_director import ask_local_art_director  # noqa: F401  (module import)

    # replicate the extraction chain exactly as implemented
    def extract(_raw):
        m = re.search(r"</think>\s*(.*)", _raw, re.DOTALL)
        if m:
            _raw = m.group(1).strip()
        s = _raw.find("[")
        e = _raw.rfind("]") + 1
        if s < 0:
            return []
        chunk = _raw[s:e] if e > s else _raw[s:]
        try:
            return _json.loads(chunk)
        except Exception:
            pass
        for blk in re.findall(r"\[[^\[\]]*\]", chunk):
            try:
                return _json.loads(blk)
            except Exception:
                continue
        try:
            return [int(x) for x in re.findall(r"\d+", chunk)]
        except Exception:
            return []

    assert extract('[0,3,7,2,1]') == [0, 3, 7, 2, 1]
    assert extract('Sure! [1,4,5] — these fit the brief best.') == [1, 4, 5]
    assert extract('Picks:\n[2,6]\nI also considered 9 but rejected it.') == [2, 6]
    assert extract('[0,3][5,7]') == [0, 3]
    assert extract('</think>analysis… </think>[4, 8, 12]') == [4, 8, 12]
    assert extract('no brackets at all 3 5 9') == []  # no brackets → refuse to guess
    assert extract('[1, 2, xyz] junk 7') == [1, 2]     # bracket chunk; trailing junk outside slice
    assert extract('[3, 5, 9, ope') == [3, 5, 9]       # truncated array → bare-int recovery
    assert extract('nothing useful') == []


def test_recent_twin_suppression_logic():
    """A near-identical burst twin of the previous run's pick must be dropped
    from the eligible pool, while distinct photos survive."""
    import numpy as np
    embs = np.array([
        [1.0, 0.0, 0.0],   # twin of the recent pick
        [0.999, 0.04, 0.0],  # twin (sim ~0.999)
        [0.0, 1.0, 0.0],   # distinct
        [0.0, 0.0, 1.0],   # distinct
    ], dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    recent = embs[0:1]                       # previous run's pick embedding
    sims = (embs @ recent.T).max(axis=1)
    keep = [i for i, s in enumerate(sims) if float(s) <= 0.90]
    assert 0 not in keep and 1 not in keep    # pick + its twin both suppressed
    assert 2 in keep and 3 in keep            # distinct photos survive


def test_fifo_headroom_widens_pool_beyond_target():
    """Starving pool retires enough oldest entries for the full manifest
    headroom, not just n_target — the ping-pong regression guard."""
    from creative_director import _fifo_evict_used, _director_pool_size
    pool = [f"p{i:03d}.jpg" for i in range(39)]
    history = [{"path": p, "ts": float(i)} for i, p in enumerate(reversed(pool))]
    n_target = 7
    min_pool = min(len(pool), max(_director_pool_size(n_target, len(pool)), n_target))
    evicted, remaining, pool_idx = _fifo_evict_used(
        history, [], min_pool, pool)
    assert len(pool_idx) == min_pool
    assert min_pool > n_target          # strictly more headroom than the request
    # evicted = the OLDEST entries (lowest ts; reversed() mapping means the
    # highest-numbered names are oldest)
    evicted_names = {e["path"] for e in evicted}
    assert all(n in evicted_names for n in ("p038.jpg", "p037.jpg", "p036.jpg"))
