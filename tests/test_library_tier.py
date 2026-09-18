"""The library tier is PERSISTED STATE (src/library_tier.py, 2026-09-13).

The bug class this guards: embedding tables and probe caches are tier-
partitioned, and the tier used to be re-derived per run from a free-RAM
measurement. A flip between runs made the incremental encode cache look
empty — the whole library re-encoded, the job ballooned, and a memory-
constrained machine that funded the first pass died on the second.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import library_tier  # noqa: E402


@pytest.fixture()
def tier_file(tmp_path, monkeypatch):
    """A scratch tier file + clean env for one test."""
    p = tmp_path / "library_tier.json"
    monkeypatch.setenv("FIRSTCUT_LIB_TIER_FILE", str(p))
    monkeypatch.delenv("SIGLIP_TIER", raising=False)
    return p


def test_roundtrip_and_invalid_refused(tier_file):
    assert library_tier.get() is None, "fresh library has no tier yet"
    assert library_tier.set("mid", "test") is True
    assert library_tier.get() == "mid"
    # A caller bug must not poison the file: only real tiers persist.
    assert library_tier.set("turbo") is False
    assert library_tier.get() == "mid"
    library_tier.clear()
    assert library_tier.get() is None


def test_corrupt_file_reads_as_no_tier(tier_file):
    tier_file.write_text("{not json at all", encoding="utf-8")
    assert library_tier.get() is None, "a corrupt cache file must never stop a grade"
    # And it self-heals: the next set overwrites it.
    assert library_tier.set("high") is True
    assert library_tier.get() == "high"


def test_apply_honours_the_library_tier_over_free_ram(tier_file, monkeypatch):
    """THE regression: auto-selection re-measures RAM per run. With a persisted
    library tier, a RAM dip must NOT flip the tier (that is what re-encoded
    whole libraries on this 16 GB machine)."""
    import tier_select
    library_tier.set("high", "test")
    # 1 GB free would auto-select the smallest tier — the library tier wins.
    tier, lbl, reason = tier_select.apply()
    assert tier == "high", f"a RAM dip must not flip the library tier: {reason}"
    assert "library" in reason.lower()
    assert tier_select.label("high") == lbl


def test_apply_persists_the_first_auto_selection(tier_file, monkeypatch):
    """First grade on a fresh library: auto-select runs, and the choice is
    persisted so the NEXT grade (maybe with less RAM free) inherits it."""
    import tier_select
    # Enough RAM for the top installed tier — whatever select() picks here,
    # apply() must persist exactly that.
    monkeypatch.setenv("FIRSTCUT_ASSUME_GPU", "1")
    tier, lbl, reason = tier_select.apply()
    assert library_tier.get() == tier, "the first auto-selection must persist"
    # And a second apply() on a starved machine returns the SAME tier.
    tier2, _, reason2 = tier_select.apply()
    assert tier2 == tier, f"library tier drifted between runs: {reason} vs {reason2}"


def test_explicit_env_override_wins_and_is_not_persisted(tier_file, monkeypatch):
    """SIGLIP_TIER in the env is a deliberate per-run choice (tests, A/B, a
    user forcing a tier) — it wins over the file but must NOT rewrite it."""
    import tier_select
    library_tier.set("low", "test")
    monkeypatch.setenv("SIGLIP_TIER", "high")
    tier, _, reason = tier_select.apply()
    assert tier == "high"
    assert library_tier.get() == "low", "an env override must not rewrite library state"


def test_run_profile_current_uses_the_library_tier(tier_file, monkeypatch):
    """Import-order landmine: modules that read run_profile BEFORE
    tier_select.apply() ran (lance_store pins its table name at import) must
    still see the library's tier — not a machine-RAM guess."""
    import run_profile as rp
    library_tier.set("low", "test")
    monkeypatch.delenv("SIGLIP_TIER", raising=False)
    prof = rp.current(refresh=True)
    assert prof.tier == "low"
    assert prof.spec.embed_dim == 768, "the profile must match the library tier's model"
    # The cache key must track the file, so a tier change rebuilds the profile.
    library_tier.set("high")
    prof2 = rp.current(refresh=True)
    assert prof2.tier == "high"