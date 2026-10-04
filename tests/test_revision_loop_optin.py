r"""
The contact-sheet critique must not run by default.

Measured through the real /api/creative-direction/stream endpoint on a folder of
graded photos: the request did not finish in TEN MINUTES. The server log shows
why -- the revision loop renders the chosen set as one contact sheet and pushes
it through a vision model on CPU:

    encoding image slice ...          170,524 ms
    decoding image batch 1/3 ...        5,300 ms
    decoding image batch 2/3 ...       20,985 ms
    decoding image batch 3/3 ...        6,110 ms

~200 seconds per iteration, up to _MAX_ITERS iterations, for a feature that
suggests at most one slot swap.

Contract (2026-09 update): an explicit FIRSTCUT_STORY_REVISION setting always
wins; otherwise it auto-enables only when CUDA has >= 4 GB free (the vision pass
is cheap there), and any failure in the settings or GPU probe falls back to
False without raising.

Run:  venv\Scripts\python.exe -m pytest tests/test_revision_loop_optin.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import creative_director as cd  # noqa: E402
import run_profile  # noqa: E402


def test_setting_is_declared():
    assert "FIRSTCUT_STORY_REVISION" in run_profile.SETTINGS


def test_explicit_setting_wins(monkeypatch):
    monkeypatch.setattr(run_profile, "setting",
                        lambda name: "1" if name == "FIRSTCUT_STORY_REVISION" else None)
    assert cd._revision_enabled() is True
    monkeypatch.setattr(run_profile, "setting",
                        lambda name: "0" if name == "FIRSTCUT_STORY_REVISION" else None)
    assert cd._revision_enabled() is False


def test_off_when_no_gpu(monkeypatch):
    """No CUDA context → the CPU vision pass is too expensive → off."""
    import torch
    monkeypatch.setattr(run_profile, "setting", lambda name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cd._revision_enabled() is False


def test_off_when_gpu_is_tight(monkeypatch):
    """CUDA present but under 4 GB free → off (the 2B critic must not push
    the machine into swap mid-run)."""
    import torch
    monkeypatch.setattr(run_profile, "setting", lambda name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _tight():
        return (3.0 * 2**30, 8.0 * 2**30)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _tight)
    assert cd._revision_enabled() is False


def test_on_when_gpu_has_headroom(monkeypatch):
    import torch
    monkeypatch.setattr(run_profile, "setting", lambda name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _roomy():
        return (6.0 * 2**30, 8.0 * 2**30)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _roomy)
    assert cd._revision_enabled() is True


def test_probe_never_raises(monkeypatch):
    """A settings failure must not be what stops a Story run: the probe is
    swallowed and the decision falls through to the GPU check."""
    def boom(_):
        raise RuntimeError("no run_profile")
    monkeypatch.setattr(run_profile, "setting", boom)
    # GPU probe broken too → every path fails → False, never an exception.
    import torch
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: (_ for _ in ()).throw(RuntimeError("no cuda")))
    assert cd._revision_enabled() is False
