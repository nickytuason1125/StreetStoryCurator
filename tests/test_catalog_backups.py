"""
Rotating catalog backups (2026-09-07 'once and for all' plan, Phase 1).

The catalog wipe that started this was only recoverable from a MANUAL
state_backup folder. merge_write() now rotates catalog.json.bak.1..3 on every
checkpoint so a corrupt write is always one rollback away. These tests pin:

  1. A merge_write over an existing catalog copies it to .bak.1.
  2. Successive writes rotate .bak.1 -> .bak.2 -> .bak.3 and keep at most 3.
  3. A first-ever write with no existing catalog creates no backups.
  4. Every .bak.N is a valid, loadable catalog — rotation never stores garbage.

Hermetic — tmp_path only, no models, no server. Run:

    venv\\Scripts\\python.exe -m pytest tests/test_catalog_backups.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src import catalog_store  # noqa: E402


def _write_seed(p: Path, marker: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"photos": [{"path": f"x/{marker}.jpg"}],
                             "folders": ["x"]}), encoding="utf-8")


def _read_bak(p: Path, n: int) -> dict:
    return json.loads((p.with_name(p.name + f".bak.{n}")).read_text(encoding="utf-8"))


def test_first_merge_rotates_existing_catalog_to_bak1(tmp_path):
    p = tmp_path / "catalog.json"
    _write_seed(p, "old")
    catalog_store.merge_write([{"path": "x/new.jpg"}], path=p)
    bak = _read_bak(p, 1)
    assert bak["photos"][0]["path"] == "x/old.jpg"
    live = json.loads(p.read_text(encoding="utf-8"))
    assert {"x/old.jpg", "x/new.jpg"} == {ph["path"] for ph in live["photos"]}


def test_successive_writes_rotate_and_cap_at_three(tmp_path):
    p = tmp_path / "catalog.json"
    for i, marker in enumerate(("v1", "v2", "v3", "v4", "v5")):
        _write_seed(p, marker)
        catalog_store.merge_write([{"path": f"x/extra{i}.jpg"}], path=p)
    baks = sorted(tmp_path.glob("catalog.json.bak.*"))
    assert len(baks) == 3, "rotation must keep exactly `keep` backups"
    assert _read_bak(p, 1)["photos"][0]["path"] == "x/v5.jpg"
    assert not (tmp_path / "catalog.json.bak.4").exists()


def test_first_ever_write_needs_no_backup(tmp_path):
    p = tmp_path / "catalog.json"
    n = catalog_store.merge_write([{"path": "x/a.jpg"}], path=p)
    assert n == 1
    assert list(tmp_path.glob("catalog.json.bak.*")) == []


def test_backups_are_loadable_catalogs(tmp_path):
    p = tmp_path / "catalog.json"
    _write_seed(p, "seed")
    catalog_store.merge_write([{"path": "x/b.jpg"}], path=p)
    d = catalog_store.load(p.with_name(p.name + ".bak.1"))
    assert isinstance(d.get("photos"), list)
