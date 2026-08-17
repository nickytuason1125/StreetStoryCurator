"""
Automatic encoder-quality selection.

The pipeline must run on whatever the machine has right now, using the best
encoder that actually fits, instead of demanding the largest one and refusing.

Rules under test:
  1. Pick the highest-quality INSTALLED tier that fits in free RAM.
  2. Never pick a tier whose weights are absent (the runtime is offline —
     that would relocate the failure, not avoid it).
  3. An explicit SIGLIP_TIER always wins over auto-selection.
  4. When nothing fits, still return a tier so the encoder's own RAM floor
     raises ONE clear error rather than this module inventing a second.
  5. Each tier uses its OWN LanceDB table, so switching quality never purges
     another tier's grades.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_tier_select.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import tier_select  # noqa: E402


@pytest.fixture
def all_installed(monkeypatch):
    monkeypatch.setattr(tier_select, "available", lambda t: True)
    monkeypatch.setattr(tier_select, "_has_hf", lambda t: t == "high")


# ── 1. best tier that fits ───────────────────────────────────────────────────

@pytest.mark.parametrize("free_gb,expect", [
    # Only 'high' has a lean fp16 checkpoint today, so it costs 3.0 while
    # mid/low are costed on the HEAVY open_clip path (4.0 / 2.0). That is why
    # 2.5 GB skips 'Balanced' entirely — it is genuinely the more expensive
    # option until a lean checkpoint exists for it.
    (16.0, "high"),   # plenty → best quality
    (6.0,  "high"),   # lean 'Pro' needs 3.0
    (3.0,  "high"),   # exactly at the requirement
    (2.5,  "low"),    # 'Pro' 3.0 no, 'Balanced' 4.0 (heavy) no → 'Fast' 2.0
    (1.5,  "low"),    # still the smallest installed
])
def test_picks_best_fitting_tier(all_installed, free_gb, expect):
    tier, lbl, reason = tier_select.select(free_gb=free_gb)
    assert tier == expect, f"{free_gb} GB free → {tier} ({reason})"
    assert lbl == tier_select.label(expect)


def test_lean_checkpoints_would_unlock_the_middle_tier(monkeypatch):
    """With a lean checkpoint for 'Balanced' too, 2.5 GB should pick it.

    Documents the payoff of building the mid-tier fp16 checkpoint: today that
    RAM band falls straight through to 'Fast'.
    """
    monkeypatch.setattr(tier_select, "available", lambda t: True)
    monkeypatch.setattr(tier_select, "_has_hf", lambda t: t in ("high", "mid"))
    assert tier_select.select(free_gb=2.5)[0] == "mid"


def test_labels_are_user_facing():
    """Labels must be plain quality words, not model internals."""
    assert tier_select.label("high") == "Pro"
    assert tier_select.label("mid") == "Balanced"
    assert tier_select.label("low") == "Fast"
    for t in ("high", "mid", "low"):
        lbl = tier_select.label(t).lower()
        for leak in ("vit", "siglip", "clip", "qwen", "b/16", "patch"):
            assert leak not in lbl, f"label {lbl!r} leaks a model name"


# ── 2. never select absent weights ───────────────────────────────────────────

def test_skips_tiers_without_weights(monkeypatch):
    """Only 'low' installed → pick it even when RAM would allow 'high'."""
    monkeypatch.setattr(tier_select, "available", lambda t: t == "low")
    monkeypatch.setattr(tier_select, "_has_hf", lambda t: False)
    tier, lbl, _ = tier_select.select(free_gb=32.0)
    assert tier == "low" and lbl == "Fast"


def test_falls_back_to_next_installed(monkeypatch):
    """High is absent → Balanced is used even with ample RAM."""
    monkeypatch.setattr(tier_select, "available", lambda t: t in ("mid", "low"))
    monkeypatch.setattr(tier_select, "_has_hf", lambda t: False)
    tier, _, reason = tier_select.select(free_gb=32.0)
    assert tier == "mid", reason


def test_no_weights_at_all_is_survivable(monkeypatch):
    monkeypatch.setattr(tier_select, "available", lambda t: False)
    tier, lbl, reason = tier_select.select(free_gb=8.0)
    assert tier in tier_select._TIERS and "no encoder weights" in reason


# ── 3. heavy vs lean loader changes the requirement ──────────────────────────

def test_requirement_reflects_the_actual_loader(monkeypatch):
    """Without the lean checkpoint, 'Pro' costs the heavy fp32 figure."""
    monkeypatch.setattr(tier_select, "_has_hf", lambda t: False)
    assert tier_select.ram_need_gb("high") == 10.3
    monkeypatch.setattr(tier_select, "_has_hf", lambda t: t == "high")
    assert tier_select.ram_need_gb("high") == 3.0


def test_heavy_loader_downgrades_when_lean_checkpoint_absent(monkeypatch):
    """5 GB free fits lean 'Pro' (3.0) but not heavy 'Pro' (10.3)."""
    monkeypatch.setattr(tier_select, "available", lambda t: True)
    monkeypatch.setattr(tier_select, "_has_hf", lambda t: False)
    assert tier_select.select(free_gb=5.0)[0] == "mid"


# ── 4. nothing fits → still return something actionable ──────────────────────

def test_returns_smallest_when_nothing_fits(all_installed):
    tier, lbl, reason = tier_select.select(free_gb=0.2)
    assert tier == "low"
    assert "below every installed encoder" in reason


# ── 5. explicit override wins ────────────────────────────────────────────────

def test_explicit_env_wins(monkeypatch):
    monkeypatch.setenv("SIGLIP_TIER", "low")
    tier, lbl, reason = tier_select.apply()
    assert tier == "low" and "explicitly" in reason


def test_apply_publishes_env_for_later_imports(monkeypatch):
    monkeypatch.delenv("SIGLIP_TIER", raising=False)
    monkeypatch.setattr(tier_select, "available", lambda t: True)
    monkeypatch.setattr(tier_select, "_has_hf", lambda t: t == "high")
    monkeypatch.setattr(tier_select, "free_ram_gb", lambda: 1.2)
    import os
    tier, lbl, _ = tier_select.apply()
    assert tier == "low"
    assert os.environ["SIGLIP_TIER"] == "low", "downstream modules read this at import"


# ── 6. per-tier tables: a quality switch must not purge grades ───────────────

@pytest.mark.parametrize("tier,table,dim", [
    ("high", "photos",     1536),
    ("mid",  "photos_mid", 1024),
    ("low",  "photos_low",  768),
])
def test_each_tier_has_its_own_table(tier, table, dim):
    """Run in a clean process — lance_store fixes both at module import."""
    code = (
        "import os, sys; os.environ['SIGLIP_TIER']=%r;"
        "sys.path.insert(0, r'%s');"
        "import lance_store as ls;"
        "print(ls._TBL_NAME, ls._EMBED_DIM)" % (tier, str(_ROOT / "src"))
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(_ROOT), timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    assert f"{table} {dim}" in out.stdout, out.stdout


def test_high_tier_keeps_the_original_table_name():
    """Existing grades live in 'photos' — the default tier must not orphan them."""
    code = (
        "import os, sys; os.environ.pop('SIGLIP_TIER', None);"
        "sys.path.insert(0, r'%s');"
        "import lance_store as ls; print(ls._TBL_NAME)" % str(_ROOT / "src")
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(_ROOT), timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    assert "photos" in out.stdout and "photos_mid" not in out.stdout


# ── 7. the selector itself must stay lightweight ─────────────────────────────

def test_selector_does_not_import_heavy_modules():
    """It runs before the pipeline loads; pulling torch/numpy would defeat it."""
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import tier_select;"
        "print('HEAVY' if ('torch' in sys.modules or 'numpy' in sys.modules) else 'LIGHT')"
        % str(_ROOT / "src")
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(_ROOT), timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    assert "LIGHT" in out.stdout, out.stdout
