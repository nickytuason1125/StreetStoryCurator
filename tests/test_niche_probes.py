"""
Niche-specific probes must ADD to the base set, never replace it.

niche_registry has tailored (pos, neg) probes for 20 niches and its own
docstring claims "grade_pipeline_v2.py -> niche_clip_probes() returns (pos, neg)
for pre-filter". That integration was never built: the function had no callers,
so every folder was graded against street-photography prompts regardless of what
it contained.

Replacing the base set would be worse than leaving it unwired. The score is
max(similarity) over the positive set, so a niche with 7 probes competes against
street's 74 and loses on set size alone - measured: landscape probes scored a
landscape folder slightly WORSE than street probes did, because there were ten
times fewer of them. Augmenting keeps every existing way to score well and adds
genre-appropriate ones.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_niche_probes.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import niche_registry as nr          # noqa: E402


BASE_POS = ["base positive one", "base positive two"]
BASE_NEG = ["base negative one"]


def test_niche_probes_are_added_not_substituted():
    pos, neg = nr.augment_probes("landscape", BASE_POS, BASE_NEG)
    for p in BASE_POS:
        assert p in pos, "an existing way to score well must never be removed"
    for n in BASE_NEG:
        assert n in neg
    assert len(pos) > len(BASE_POS), "landscape probes were not added"


def test_landscape_adds_landscape_vocabulary():
    pos, _ = nr.augment_probes("landscape", BASE_POS, BASE_NEG)
    joined = " ".join(pos).lower()
    assert "landscape" in joined


def test_unknown_preset_is_a_no_op_not_a_crash():
    """A preset typo must not empty the probe set and fail every photo."""
    pos, neg = nr.augment_probes("not_a_real_niche_xyz", BASE_POS, BASE_NEG)
    assert pos == BASE_POS
    assert neg == BASE_NEG


def test_empty_preset_is_a_no_op():
    pos, neg = nr.augment_probes("", BASE_POS, BASE_NEG)
    assert pos == BASE_POS and neg == BASE_NEG


def test_no_duplicates_are_introduced():
    """Duplicated probes are wasted encode time and skew nothing usefully."""
    pos, neg = nr.augment_probes("classic_street", BASE_POS, BASE_NEG)
    assert len(pos) == len(set(pos))
    assert len(neg) == len(set(neg))


def test_calling_twice_is_stable():
    """run_v2 may build probes more than once; the set must not grow each time."""
    p1, n1 = nr.augment_probes("night", BASE_POS, BASE_NEG)
    p2, n2 = nr.augment_probes("night", p1, n1)
    assert p1 == p2 and n1 == n2


def test_the_base_lists_are_not_mutated():
    before_pos, before_neg = list(BASE_POS), list(BASE_NEG)
    nr.augment_probes("landscape", BASE_POS, BASE_NEG)
    assert BASE_POS == before_pos and BASE_NEG == before_neg


# ── the union: one shared rubric, no genre gets a private scale ──────────────

def test_union_contains_every_niche_vocabulary():
    """Every genre's idea of 'good' must be reachable by every photo.

    Grading each folder only against its OWN genre would normalise every genre
    to the same spread - fine art ~30% Strong, street ~30% Strong - regardless
    of whether the work is actually comparable. That is the batch-relative
    curve resurrected one level up, and it flatters every folder.
    """
    pos, neg = nr.union_probes(BASE_POS, BASE_NEG)
    for niche in nr.REGISTRY:
        npos, nneg = nr.niche_clip_probes(niche)
        for p in npos:
            assert p in pos, f"{niche} vocabulary missing from the shared rubric"
        for n in nneg:
            assert n in neg


def test_union_keeps_the_base_set():
    pos, neg = nr.union_probes(BASE_POS, BASE_NEG)
    assert all(p in pos for p in BASE_POS)
    assert all(n in neg for n in BASE_NEG)


def test_union_is_deterministic():
    """Probe ORDER feeds the cache key and the anchor fingerprint. If it varied
    run to run, every cull would re-encode and every anchor would read stale."""
    a = nr.union_probes(BASE_POS, BASE_NEG)
    b = nr.union_probes(BASE_POS, BASE_NEG)
    assert a == b


def test_union_has_no_duplicates():
    pos, neg = nr.union_probes(BASE_POS, BASE_NEG)
    assert len(pos) == len(set(pos))
    assert len(neg) == len(set(neg))


def test_union_massively_widens_the_negative_side():
    """The base had FIVE generic negatives, so max(neg) was near-constant and
    the discriminant was effectively positive-only."""
    _, neg = nr.union_probes(BASE_POS, BASE_NEG)
    assert len(neg) > 50, "the negative side must be able to discriminate"


def test_union_does_not_mutate_inputs():
    bp, bn = list(BASE_POS), list(BASE_NEG)
    nr.union_probes(BASE_POS, BASE_NEG)
    assert BASE_POS == bp and BASE_NEG == bn


@pytest.mark.parametrize("niche", sorted(nr.REGISTRY))
def test_every_registered_niche_yields_usable_probes(niche):
    """All 20 niches must work, not just the ones that happen to be exercised."""
    pos, neg = nr.augment_probes(niche, BASE_POS, BASE_NEG)
    assert len(pos) >= len(BASE_POS)
    assert all(isinstance(p, str) and p.strip() for p in pos)
    assert all(isinstance(n, str) and n.strip() for n in neg)
