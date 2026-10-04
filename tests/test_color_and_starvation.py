"""Colour-brief honouring + early-avoid starvation rescue (2026-09-22)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Fix 2: colour extraction ─────────────────────────────────────────────────

def test_orange_and_reds_matches_red_orange_family():
    """The user's exact failing brief: 'orange and reds' previously matched
    nothing (_COLOR_KW had no red/orange family) so blues were returned."""
    from creative_director_agent import extract_color_terms
    assert extract_color_terms("orange and reds") == ["warm/red-orange"]


def test_rule_set_carries_color_target():
    """Fix 3: the rule set must expose the detected colour families so the
    UI 'brief read as' chips agree with what selection applied. Previously
    colour lived only in DirectorBrief and the chips said 'neutral' for an
    orange/yellow brief."""
    from creative_director_agent import _keyword_rule_set
    rs = _keyword_rule_set("RETRO architecture, orange/yellow, city")
    assert "warm/red-orange" in rs["COLOR_TARGET"]
    assert rs["GEOMETRIC_PRIORITY"] == "High"     # untouched by colour work


def test_rule_set_color_target_survives_gguf_merge():
    """The GGUF refinement doesn't know colour — update() must not clobber."""
    import creative_director_agent as agent
    from creative_director_agent import generate_rule_set
    original = agent._gguf_refine_rule_set
    agent._gguf_refine_rule_set = lambda *a, **k: {"HARD_FILTER_PEOPLE": True,
                                                   "EXCLUDE": []}
    try:
        rs = generate_rule_set("vibrant orange and reds, no people")
    finally:
        agent._gguf_refine_rule_set = original
    assert "warm/red-orange" in rs["COLOR_TARGET"]
    assert rs["HARD_FILTER_PEOPLE"] is True


def test_rule_set_empty_brief_has_no_color_target():
    from creative_director_agent import _keyword_rule_set
    assert _keyword_rule_set("rainy overcast streets")["COLOR_TARGET"] == []


def test_multiple_colour_families_all_detected():
    from creative_director_agent import extract_color_terms
    hits = set(extract_color_terms("blues and warm oranges"))
    # "warm" also legitimately hits the golden family — every named family
    # is returned so the boost can honour the whole request.
    assert hits == {"cool/blue", "warm/red-orange", "warm/golden"}


def test_no_substring_false_positives():
    """'red' must not match inside 'hundred'/'colored' (word boundaries)."""
    from creative_director_agent import extract_color_terms
    assert extract_color_terms("a hundred colored walls") == []


def test_empty_brief_has_no_colour_families():
    from creative_director_agent import extract_color_terms
    assert extract_color_terms("") == []
    assert extract_color_terms("   ") == []


def test_director_brief_reads_orange_reds():
    from creative_director_agent import _keyword_director_brief
    db = _keyword_director_brief("orange and reds, geometric lines")
    assert db.color_profile_target == "warm/red-orange"
    assert db.thematic_niche == "architecture"


# ── Fix 1: early-avoid starvation FIFO ───────────────────────────────────────

def _mk(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"photo_{i}.jpg"
        p.write_bytes(b"x")
        paths.append(str(p))
    return paths


def test_starvation_retire_frees_oldest_used(tmp_path):
    """pool=135 avoid=129 kept=4 for a 6-photo story: the helper must retire
    the OLDEST used entries until the manifest headroom is met."""
    from creative_director import _starvation_retire
    paths = _mk(tmp_path, 6)
    avoid_res = {str(Path(p).resolve()) for p in paths[1:]}       # 5 used
    used = [{"path": paths[i], "ts": 1000 + i} for i in range(1, 6)]
    addback, evicted, remaining = _starvation_retire(
        used, avoid_res, paths, eligible_idx=[0], cap=4)
    assert len(addback) == 3                    # retire just enough for cap=4
    # oldest first: photos 1, 2, 3 come back; 4 and 5 stay retired
    assert set(addback) == {1, 2, 3}
    assert {Path(e["path"]).name for e in remaining} == {
        "photo_4.jpg", "photo_5.jpg"}
    assert all(str(Path(paths[i]).resolve()) in evicted for i in (1, 2, 3))


def test_starvation_retire_noop_when_history_empty(tmp_path):
    from creative_director import _starvation_retire
    paths = _mk(tmp_path, 4)
    addback, evicted, remaining = _starvation_retire(
        [], set(), paths, eligible_idx=[0, 1], cap=4)
    assert addback == [] and evicted == set() and remaining == []


def test_starvation_retire_ignores_entries_not_in_avoid(tmp_path):
    """Entries whose path was never excluded free nothing."""
    from creative_director import _starvation_retire
    paths = _mk(tmp_path, 3)
    used = [{"path": paths[0], "ts": 1}]        # in pool, not in avoid set
    addback, evicted, remaining = _starvation_retire(
        used, {str(Path(paths[2]).resolve())}, paths, eligible_idx=[0, 1], cap=3)
    assert addback == [] and evicted == set()
    assert len(remaining) == 1


def test_starvation_retire_cap_already_met(tmp_path):
    from creative_director import _starvation_retire
    paths = _mk(tmp_path, 4)
    avoid_res = {str(Path(p).resolve()) for p in paths[2:]}
    used = [{"path": paths[2], "ts": 1}, {"path": paths[3], "ts": 2}]
    addback, evicted, remaining = _starvation_retire(
        used, avoid_res, paths, eligible_idx=[0, 1], cap=2)
    assert addback == [] and evicted == set() and len(remaining) == 2