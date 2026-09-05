"""
H1 regression: PersonalHead's singleton (_head/_opt) is mutated from real
threadpool threads — UI star-ratings, the every-25-ratings auto-retrain, and
manual retrains all race. An RLock now serialises _get_head/update/fit/score.

These tests assert the corruption-freedom properties the lock guarantees:
no exceptions under concurrency, the optimizer always bound to the LIVE head
(fit() reassigns _head; an orphaned optimizer silently no-ops training),
finite weights, and the weights file written. They run on a tiny 16-dim head
with the weights file redirected to a temp dir, so the real
cache/personal_head.pt and the caller's taste model are never touched.

Note on what is deliberately NOT asserted: cross-thread floating-point
equality with a serial run — thread interleaving changes apply order, and
the lock's job is corruption-freedom, not FP determinism.
"""
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import personal_head as ph  # noqa: E402


@pytest.fixture()
def isolated_head(monkeypatch, tmp_path):
    """Redirect the weights file to a temp dir and reset the singleton;
    restore the caller's real singleton afterwards."""
    monkeypatch.setattr(ph, "_WEIGHTS_PATH", tmp_path / "personal_head.pt")
    saved = (ph._head, ph._opt)
    ph._head = None
    ph._opt = None
    yield
    ph._head, ph._opt = saved


def _rand(dim: int = 16):
    return np.random.randn(dim).astype(np.float32)


def _assert_coherent():
    head, opt = ph._get_head()
    assert head is not None and opt is not None
    head_ids = {id(p) for p in head.parameters()}
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert head_ids == opt_ids, "optimizer must be bound to the LIVE head"
    for p in head.parameters():
        assert torch.isfinite(p).all(), "weights corrupted under concurrency"


def test_concurrent_updates_do_not_corrupt(isolated_head):
    errors: list[Exception] = []

    def worker():
        try:
            for _ in range(8):
                ph.update(_rand(), "Strong ✅", _rand(), "Weak ❌")
        except Exception as e:  # noqa: BLE001 — collecting, not swallowing
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"racing updates raised: {errors[:3]}"
    _assert_coherent()
    assert ph._WEIGHTS_PATH.exists(), "updates must persist through the lock"


def test_fit_and_update_interleave_safely(isolated_head):
    """The dangerous pair: fit() reassigns _head/_opt outright while update()
    is mid-backward. Serialised, neither may observe a half-swapped state."""
    errors: list[Exception] = []
    samples = [(_rand(), "Strong ✅"), (_rand(), "Mid ⚠️"), (_rand(), "Weak ❌")] * 10

    def fitter():
        try:
            for _ in range(4):
                ph.fit(samples, epochs=1, pairs_per_combo=20)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def updater():
        try:
            for _ in range(12):
                ph.update(_rand(), "Strong ✅", _rand(), "Weak ❌")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=fitter), threading.Thread(target=updater)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"fit/update interleave raised: {errors[:3]}"
    _assert_coherent()


def test_score_during_fit_stays_valid(isolated_head):
    """score() reads the head forward-only while fit() rebuilds it — under
    the lock every score must come from a whole, finite model."""
    errors: list[Exception] = []
    samples = [(_rand(), "Strong ✅"), (_rand(), "Mid ⚠️"), (_rand(), "Weak ❌")] * 10

    def fitter():
        try:
            for _ in range(4):
                ph.fit(samples, epochs=1, pairs_per_combo=20)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def scorer():
        try:
            for _ in range(20):
                scores = ph.score(np.stack([_rand() for _ in range(4)]))
                assert np.isfinite(scores).all()
                assert (scores >= 0.0).all() and (scores <= 1.0).all()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=fitter), threading.Thread(target=scorer),
               threading.Thread(target=scorer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"score/fit race raised: {errors[:3]}"
    _assert_coherent()