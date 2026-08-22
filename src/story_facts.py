"""
story_facts.py — measured facts about a candidate photo.

Why this exists
---------------
The Art Director's payload was {"id", "score", "style", "profile"} per photo,
and `profile` was always "" because nothing in this repo ever wrote
semantic_profile (read in 3 places, written in 0). So a "Magnum photo editor"
prompt about pacing and negative space was being applied to a spreadsheet.

Where each fact comes from, and why
-----------------------------------
FRAMING comes from the LENS, not the embedding. The first design derived shot
type from SigLIP zero-shot probes; it was validated against 488 real Strong
photos and rejected. The "close" class returned a waterfall, then a wide-angle
sun-flare landscape; "portrait" could not win a single frame out of 488, every
margin negative. Rewriting the prompts with explicit scale words and shared
negatives did not fix it. CLIP-family models are strong on CONTENT and weak on
FRAMING. Focal length is not an inference — 24mm IS wide, and the camera wrote
it down.

SigLIP keeps what it is good at: brief match, mood, content. That is what the
existing text-semantic rerank already uses it for, successfully.

SESSIONS come from capture time, which is what turns a pile of files into a
shoot with a beginning.

TASTE comes from personal_score — the PersonalHead trained on the user's own
ratings. It is computed for every photo and then discarded by the director.

Every fact is allowed to be None. A missing fact carries a `reason`, because a
photo silently defaulted into a narrative slot on invented evidence is the bug
this module replaces.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

# 35mm-equivalent millimetres. Conventional photographic boundaries: 35mm and
# wider is a wide lens, 70mm and longer is a portrait/tele lens. Named because a
# boundary that drifts silently changes every story the app ever builds.
WIDE_MAX_MM = 35.0
MEDIUM_MAX_MM = 70.0

# A half-hour gap ends a shoot. Street work happens in bursts; the gaps between
# them are where a photographer moved on to somewhere else.
SESSION_GAP_S = 1800.0

# Fraction of frame AREA covered by the largest detected face.
#
# Subject scale outranks focal length when the two disagree, because a face
# filling the frame is narratively a close-up whatever lens shot it -- a 24mm
# portrait at arm's length is optically wide and editorially close, and the
# story cares about the second reading.
#
# It also covers the gap that killed the focal-only design: 68% of Strong
# photos in this library are Lightroom exports with the lens data stripped,
# but a face is measurable in any pixels.
CLOSE_FACE_FRAC = 0.08
MEDIUM_FACE_FRAC = 0.01

_NUM = re.compile(r"[0-9]+(?:[.][0-9]+)?")


def _to_mm(value: Any) -> Optional[float]:
    """Focal length as a number, or None. Never raises.

    exif_reader formats focal as '24mm', so this has to accept the string it
    actually produces, not the float we might wish it produced.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        mm = float(value)
        return mm if mm > 0 else None
    m = _NUM.search(str(value))
    if not m:
        return None
    try:
        mm = float(m.group())
    except ValueError:
        return None
    return mm if mm > 0 else None


def framing_from_subject(frac: Any) -> Optional[str]:
    """Framing from how much of the frame the subject occupies, or None.

    None when there is no subject to measure -- absence of a face is not
    evidence of a wide shot, it is absence of evidence.
    """
    try:
        f = float(frac)
    except (TypeError, ValueError):
        return None
    if f <= 0.0:
        return None
    if f >= CLOSE_FACE_FRAC:
        return "close"
    if f >= MEDIUM_FACE_FRAC:
        return "medium"
    return "wide"


def framing_from_focal(value: Any) -> Optional[str]:
    """'wide' | 'medium' | 'close', or None when the lens is unknown.

    None is deliberate. RAW files, scans and phone exports often carry no focal
    length, and guessing 'medium' would place an unknown frame into a narrative
    slot on evidence that does not exist.
    """
    mm = _to_mm(value)
    if mm is None:
        return None
    if mm <= WIDE_MAX_MM:
        return "wide"
    if mm <= MEDIUM_MAX_MM:
        return "medium"
    return "close"


def sessions_from_timestamps(ts: "list[float]",
                             gap_s: float = SESSION_GAP_S) -> "list[int]":
    """Group frames into shoots by gaps in capture time.

    Returns one session index per input, IN INPUT ORDER. Input is not assumed
    sorted: the pool arrives ranked by score, not chronologically.
    """
    if not ts:
        return []
    order = sorted(range(len(ts)), key=lambda i: float(ts[i] or 0.0))
    out = [0] * len(ts)
    cur = 0
    prev: Optional[float] = None
    for i in order:
        t = float(ts[i] or 0.0)
        if prev is not None and (t - prev) > gap_s:
            cur += 1
        out[i] = cur
        prev = t
    return out


@dataclass(frozen=True)
class PhotoFacts:
    """What is actually known about one candidate. None means unknown, and
    `reason` says which facts are missing so the caller can report it rather
    than pretend."""
    path: str
    framing: Optional[str] = None
    focal_mm: Optional[float] = None
    shot_at: Optional[float] = None
    session: Optional[int] = None
    luminance: Optional[float] = None
    subject_scale: Optional[float] = None
    framing_source: Optional[str] = None
    score: Optional[float] = None
    personal_score: Optional[float] = None
    reason: Optional[str] = None

    def terse(self) -> str:
        """The one-line form for an LLM manifest.

        Terse on purpose: a verbose payload took generation from 2.6s to 22.5s
        on the target laptop, and 25 verbose candidates cost 36.1s against 4.7s
        for 12. CPU prefill is superlinear, so every token here is paid for.
        """
        bits = []
        if self.framing:
            bits.append(self.framing)
        if self.focal_mm:
            bits.append("%dmm" % int(self.focal_mm))
        if self.luminance is not None:
            bits.append("lum%d" % int(self.luminance))
        if self.session is not None:
            bits.append("s%d" % self.session)
        return " ".join(bits)


def _first(row: dict, *keys: str) -> Any:
    for k in keys:
        v = row.get(k)
        if v not in (None, "", 0):
            return v
    return None


def facts_for_pool(rows: "list[dict]") -> "list[PhotoFacts]":
    """Facts for a candidate pool, in input order.

    `rows` are LanceDB-shaped dicts. Sessions are computed across the WHOLE
    pool, because a session only means something relative to its neighbours.
    """
    if not rows:
        return []

    stamps = [float(r.get("exif_ts") or 0.0) for r in rows]
    sessions = sessions_from_timestamps(stamps)

    out = []
    for r, sess in zip(rows, sessions):
        # focal_35mm first: it is the crop-corrected number, and 24mm on APS-C
        # is not a wide shot.
        raw_focal = _first(r, "focal_35mm", "focal")
        mm = _to_mm(raw_focal)

        # Subject scale first, lens second. When they disagree the subject
        # wins; when there is no subject the lens is all we have.
        frac = r.get("largest_face_frac")
        framing = framing_from_subject(frac)
        source = "subject" if framing else None
        if framing is None:
            framing = framing_from_focal(raw_focal)
            source = "focal" if framing else None
        ts = float(r.get("exif_ts") or 0.0)

        missing = []
        if framing is None:
            missing.append("no subject detected and no focal length in EXIF")
        if not ts:
            missing.append("no capture time")

        out.append(PhotoFacts(
            path=str(r.get("path", "")),
            framing=framing,
            focal_mm=mm,
            shot_at=ts or None,
            session=sess,
            luminance=r.get("luminance"),
            subject_scale=(float(frac) if frac not in (None, "") else None),
            framing_source=source,
            score=r.get("score"),
            personal_score=r.get("personal_score"),
            reason="; ".join(missing) if missing else None,
        ))
    return out
