"""
FRONTIER 2026 — Legacy graders permanently removed.

The following source files have been renamed to *.legacy_backup and can no
longer be imported:

    src/qalign_grader.py          → qalign_grader.py.legacy_backup
    src/onealign_scorer.py        → onealign_scorer.py.legacy_backup
    src/lightweight_analyzer.py   → lightweight_analyzer.py.legacy_backup
      (contained: _NIMAFallback, _V1LightweightFallback, LightweightStreetScorer,
                  MobileViT scorer, DINOv2-small ops)

Any attempt to import from this module raises RuntimeError.
Use grade_pipeline_v2.run_v2() which routes through SigLIP-2 + SpecVLM.
"""
from __future__ import annotations


def __getattr__(name: str):
    _REMOVED = {
        "QAlignGrader", "OneAlignScorer", "OneAlign",
        "_NIMAFallback", "_V1LightweightFallback", "LightweightStreetScorer",
    }
    if name in _REMOVED:
        raise RuntimeError(
            f"'{name}' was permanently removed in Frontier 2026. "
            "Source backed up at src/<module>.legacy_backup. "
            "Use grade_pipeline_v2.run_v2() instead."
        )
    raise AttributeError(f"module 'deprecated' has no attribute {name!r}")


__all__: list = []
