"""
Claim-vs-signal hallucination validator.

Checks jury/critique claims that cite a specific structured evidence field
(aspect name + value, at a given slot) against the REAL computed value in
that slot's breakdown dict. A claim citing evidence that doesn't exist, or
whose claimed value diverges too far from the real one, is treated as
hallucinated and rejected — this is what makes the "evidence-first" design
(the cited_aspect/cited_value fields forced by the jury/swap grammars)
load-bearing rather than decorative: a model that names a fact must be
checkable against what was actually computed, not merely plausible-sounding.

Used by src/jury_engine.py (per persona verdict, before it counts toward
the self-consistency spread or the final narrative) and src/contact_sheet.py
(a swap request whose cited evidence fails validation is downgraded to
"accept" rather than executed — a hallucinated number is rejectable, not
correctable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Claim:
    text: str
    cited_aspect: Optional[str] = None
    cited_value: Optional[float] = None
    cited_slot: Optional[int] = None


@dataclass
class ValidationResult:
    passed: bool
    reason: str = ""


def validate_claims(
    claims: list[Claim],
    aspects_by_slot: list[dict],
    tolerance: float = 0.15,
    bboxes: Optional[list[dict]] = None,
) -> ValidationResult:
    """
    A claim with no cited evidence (cited_aspect is None) passes through
    unvalidated — the grammar forces evidence-first for anything
    quantitative, but a purely qualitative claim ("the sequence flows
    well") needs no evidence field to be legitimate.

    A claim WITH cited evidence fails validation if:
      - cited_slot is missing or out of range for aspects_by_slot, or
      - cited_aspect doesn't exist in that slot's real breakdown data (the
        model named a field that was never computed — invented evidence), or
      - the claimed value diverges from the real value by more than
        `tolerance` (the model's number doesn't match reality).

    bboxes is a documented extension point for future bbox-grounded claim
    validation (checking claims like "person in frame" against real D-FINE
    detections) — unused for now; the jury/critique here operate at the
    sequence level (which slot, which aspect), not the pixel-region level
    that run_audit_annotation in critique_engine.py already owns separately.
    """
    for claim in claims:
        if claim.cited_aspect is None or claim.cited_value is None:
            continue
        if claim.cited_slot is None or not (0 <= claim.cited_slot < len(aspects_by_slot)):
            return ValidationResult(False, f"cited_slot {claim.cited_slot!r} out of range for {len(aspects_by_slot)} slots")
        real = aspects_by_slot[claim.cited_slot].get(claim.cited_aspect)
        if real is None or not isinstance(real, (int, float)):
            return ValidationResult(
                False,
                f"cited_aspect '{claim.cited_aspect}' not present in slot {claim.cited_slot}'s real data",
            )
        if abs(float(real) - claim.cited_value) > tolerance:
            return ValidationResult(
                False,
                f"cited_value {claim.cited_value} diverges from real {real:.3f} by more than {tolerance}",
            )
    return ValidationResult(True)
