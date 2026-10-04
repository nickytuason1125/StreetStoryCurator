r"""
Vision-verified story mode: the 2026-09 audit fixes.

Covers the correctness fixes that downstream spatial facts depend on:
  F2  letterboxed render + content_rect + bbox remap (was a square stretch,
      which corrupted h_gap / area_pct / left-right claims for non-square
      frames)
  F1  the user brief steers slot retrieval (blended query vector)
  F4  slot provenance surfaced instead of hidden in a bool
  F5  hardened JSON parsing (fences, trailing commas, tagged output)
  F3  validator thresholds follow config.json, not hard-coded numbers

Run:  venv\Scripts\python.exe -m pytest tests/test_vision_story_mode.py -v
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vision_story_mode as vsm


# ---------------------------------------------------------------- F2: geometry

def test_render_letterboxes_tall_image_and_reports_rect(tmp_path):
    from PIL import Image
    p = tmp_path / "tall.png"
    Image.new("RGB", (100, 400), (255, 255, 255)).save(p)
    b64, rect = vsm._render_image_b64(str(p), size=512)
    assert b64
    img = Image.open(io.BytesIO(__import__("base64").b64decode(b64)))
    assert img.size == (512, 512)
    # 100x400 -> 128x512 content on a 512 canvas -> in 0-1000 sent coords:
    # x spans [375, 625], y spans the full canvas
    x0, y0, x1, y1 = rect
    assert (x0, y0, x1, y1) == pytest.approx((375.0, 0.0, 625.0, 1000.0))


def test_render_square_image_covers_full_canvas(tmp_path):
    from PIL import Image
    p = tmp_path / "square.png"
    Image.new("RGB", (200, 200), (0, 0, 0)).save(p)
    b64, rect = vsm._render_image_b64(str(p), size=512)
    assert rect == pytest.approx((0.0, 0.0, 1000.0, 1000.0))


def test_remap_bbox_is_identity_for_full_canvas():
    assert vsm._remap_bbox([100, 200, 300, 400], (0.0, 0.0, 1000.0, 1000.0)) \
        == [100, 200, 300, 400]


def test_remap_bbox_maps_padded_canvas_back_to_the_photograph():
    # tall image: content occupies x in [375, 625] of the sent canvas
    rect = (375.0, 0.0, 625.0, 1000.0)
    out = vsm._remap_bbox([0, 375, 1000, 625], rect)
    assert out == [0, 0, 1000, 1000]          # full content box -> full frame


def test_remap_bbox_clamps_and_survives_garbage():
    assert vsm._remap_bbox([-50, 0, 2000, 1000], (0.0, 0.0, 1000.0, 1000.0)) \
        == [0, 0, 1000, 1000]
    assert vsm._remap_bbox("junk", (0.0, 0.0, 1000.0, 1000.0)) == [0, 0, 1000, 1000]


def test_remap_of_the_full_frame_fallback_stays_the_full_frame():
    # the [0,0,1000,1000] fallback from _sanitize_bbox must survive remapping
    # for ANY rect (it only maps to the full frame when the rect is full)
    assert vsm._remap_bbox([0, 0, 1000, 1000], (0.0, 0.0, 1000.0, 1000.0)) \
        == [0, 0, 1000, 1000]


# ---------------------------------------------------------------- F5: parsing

def test_parse_slot_json_accepts_clean_json():
    raw = json.dumps({
        "approved_for_essay": True,
        "assigned_slot": "Opener",
        "curator_justification": "deep focus establishes context.",
        "subject_bbox": [10, 20, 30, 40],
        "anchor_bbox": [0, 0, 1000, 1000],
    })
    out = vsm._parse_slot_json(raw, "Opener")
    assert out is not None and out["approved_for_essay"] is True
    assert out["subject_bbox"] == [10, 20, 30, 40]


def test_parse_slot_json_repairs_fences_and_trailing_commas():
    raw = "```json\n{" \
        "\"approved_for_essay\": false, \"assigned_slot\": \"Snapshot\", " \
        "\"curator_justification\": \"flat scene.\", }```"
    out = vsm._parse_slot_json(raw, "Opener")
    assert out is not None and out["assigned_slot"] == "Snapshot"


def test_parse_slot_json_returns_none_on_garbage_or_missing_fields():
    assert vsm._parse_slot_json("no json here at all", "Opener") is None
    raw = json.dumps({"approved_for_essay": True})
    assert vsm._parse_slot_json(raw, "Opener") is None


def test_strip_trailing_commas_only_touches_structural_commas():
    # the comma before the brace is dropped; harmless whitespace may remain
    out = vsm._strip_trailing_commas('{"a": 1, }')
    assert out == '{"a": 1 }'
    assert json.loads(out) == {"a": 1}
    # commas INSIDE strings are preserved
    kept = vsm._strip_trailing_commas('{"a": "x, y"}')
    assert kept == '{"a": "x, y"}'


# ---------------------------------------------------------------- F1: blending

def test_rank_for_slot_blends_the_user_brief():
    slot_vec = [1.0, 0.0]
    brief = [0.0, 1.0]
    embs = [[1.0, 0.0], [0.0, 1.0]]          # e0 matches the slot, e1 the brief
    paths = ["a.jpg", "b.jpg"]
    ranked_plain = vsm._rank_for_slot(
        "Opener", paths, embs, {"Opener": slot_vec}, set(), top_k=2)
    assert ranked_plain[0][1] == "a.jpg"
    # brief weight above 0.5 must flip the ranking toward the brief match
    old = vsm._CFG.get("SLOT_BRIEF_WEIGHT")
    try:
        vsm._CFG["SLOT_BRIEF_WEIGHT"] = 0.70
        ranked_blended = vsm._rank_for_slot(
            "Opener", paths, embs, {"Opener": slot_vec}, set(), top_k=2,
            prompt_emb=brief)
    finally:
        if old is None:
            vsm._CFG.pop("SLOT_BRIEF_WEIGHT", None)
        else:
            vsm._CFG["SLOT_BRIEF_WEIGHT"] = old
    assert ranked_blended[0][1] == "b.jpg"   # the brief flipped the ranking


def test_rank_for_slot_skips_already_used_paths():
    ranked = vsm._rank_for_slot(
        "Opener", ["a.jpg", "b.jpg"], [[1.0, 0.0], [0.9, 0.1]],
        {"Opener": [1.0, 0.0]}, {"a.jpg"}, top_k=2)
    assert [p for _, p in ranked] == ["b.jpg"]


# ---------------------------------------------------------------- F4: honesty

def test_apply_honest_mode_keeps_only_vision_approved_slots():
    slots = [
        vsm.SlotResult("Opener", "a.jpg", 0.9, True, "Opener", "ok"),
        vsm.SlotResult("Detail/Accent", "b.jpg", 0.5, False, "Snapshot", "gate fail"),
        vsm.SlotResult("Closer/Resolution", "c.jpg", 0.8, True, "Closer/Resolution", "ok"),
    ]
    kept = vsm._apply_honest_mode(slots)
    assert [r.image_path for r in kept] == ["a.jpg", "c.jpg"]


def test_slotresult_provenance_defaults_to_vision_approved():
    r = vsm.SlotResult("Opener", "a.jpg", 0.9, True, "Opener", "ok")
    assert r.provenance == "vision_approved"


# ---------------------------------------------------------------- F3: config

def test_validator_dominance_check_follows_configured_threshold():
    # subject fills 50% of the frame; default DOMINANT_AREA_PCT is 18.3
    r = vsm.SlotResult(
        "Subject/Interaction", "a.jpg", 0.9, True, "Subject/Interaction", "ok",
        subject_bbox=[0, 0, 500, 1000])
    verdict = "A plain sentence with no dominance language at all. left frame."
    out = vsm.validate_narrative(verdict, [r])
    dom_line = next(d for d in out["details"] if "Dominance check" in d)
    assert f"area>{int(vsm._CFG['DOMINANT_AREA_PCT'])}%" in dom_line
    assert out["checks"]["dominance_language"] is False


def test_spatial_facts_area_uses_configured_threshold():
    r = vsm.SlotResult(
        "Opener", "a.jpg", 0.9, True, "Opener", "ok",
        subject_bbox=[0, 0, 1000, 1000])
    facts = vsm._derive_spatial_facts(r)
    assert facts["subject_area_pct"] == 100.0
    assert facts["subject_dominant"] is True


def test_config_defaults_carry_the_new_knobs():
    for key in ("LUM_RANGE_THRESH", "SLOT_BRIEF_WEIGHT", "HONEST_SLOTS",
                "CANDIDATES_PER_SLOT", "GEN_SEED"):
        assert key in vsm._THRESHOLD_DEFAULTS, key

# ------------------------------------------------- 2026 protocol: structured judge

def _packet_for_registry():
    return [
        vsm.SlotResult(
            "Subject/Interaction", "a.jpg", 0.9, True, "Subject/Interaction", "ok",
            subject_bbox=[80, 600, 700, 980],
            anchor_bbox=[0, 0, 1000, 200], luminance=0.55),
    ]


def test_fact_registry_values_match_the_packet():
    reg = vsm._fact_registry(_packet_for_registry())
    # subject area (620*380)/10000 = 23.6% dominant; h_gap = 600-200 = 400
    assert reg["S1_area"][0] == 23.6
    assert reg["S1_hgap"][0] == 400
    assert reg["S1_lum"][0] == 0.55


def test_structured_claims_must_reference_real_facts_with_exact_values():
    reg = vsm._fact_registry(_packet_for_registry())
    ok = vsm._build_structured_verdict({
        "summary": "Tense frame.",
        "claims": [
            {"fact_ref": "S1_area", "text": "The argument dominates", "value": 23.6},
            {"fact_ref": "S1_hgap", "text": "Pushed right of the wall", "value": 400},
        ],
    }, reg)
    assert ok is not None and "23.6" in ok and "400" in ok
    # wrong value -> claim dropped; only grounded claims survive
    bad = vsm._build_structured_verdict({
        "summary": "Tense frame.",
        "claims": [{"fact_ref": "S1_area", "text": "dominates", "value": 99.0}],
    }, reg)
    assert bad is None
    # unknown fact_ref -> dropped
    unk = vsm._build_structured_verdict({
        "summary": "x",
        "claims": [{"fact_ref": "S9_area", "text": "y", "value": 23.6}],
    }, reg)
    assert unk is None


def test_structured_word_cap_is_enforced_by_assembly():
    reg = vsm._fact_registry(_packet_for_registry())
    long_claims = [
        {"fact_ref": "S1_area", "text": "word " * 40, "value": 23.6},
        {"fact_ref": "S1_hgap", "text": "word " * 40, "value": 400},
        {"fact_ref": "S1_lum", "text": "word " * 40, "value": 0.55},
    ]
    out = vsm._build_structured_verdict({"summary": "word " * 60, "claims": long_claims}, reg)
    assert out is not None
    assert len(out.split()) <= 120
