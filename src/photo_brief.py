"""photo_brief — the Brief object: one parse of the user's brief, consumed
everywhere (2026-09-16).

Before this module the brief was a raw string that five layers each
re-interpreted independently: keyword tables, SigLIP rerank, GGUF rule set,
the "<subject> only" regex, and the Art Director prompt. They could disagree
and nothing could prove they didn't.

This module owns the parse. Every consumer — the /interpret preview, the
subject filter, the model prompts, the post-run display — reads the same
Brief. Adding a new brief concept = one field here, not five edits.

No heavy imports: this module must stay cheap to import from tests and the
frontend-facing router (the pipeline's model code imports FROM here, never
the reverse)."""

from dataclasses import dataclass, field, asdict
import re

from creative_director_agent import _keyword_rule_set

# ── Subject parsing ──────────────────────────────────────────────────────────

SUBJECT_SYNONYMS = {
    "vehicles":    "a car or motorcycle or scooter on the street",
    "vehicle":     "a car or motorcycle or scooter on the street",
    "cars":        "a car parked on the street",
    "car":         "a car parked on the street",
    "motorcycles": "a motorcycle or scooter parked on the street",
    "motorcycle":  "a motorcycle or scooter parked on the street",
    "scooters":    "a scooter parked on the street",
    "bicycles":    "a bicycle on the street",
    "bicycle":     "a bicycle on the street",
    "people":      "a person walking on the street",
    "person":      "a person walking on the street",
    "pedestrians": "a person walking on the street",
    "architecture": "buildings and architecture",
    "buildings":   "buildings and architecture",
    "signs":       "neon signs and shop signs",
    "signage":     "neon signs and shop signs",
    "reflections": "reflections in windows and puddles",
    "food":        "street food and food stalls",
    "night":       "the street at night lit by artificial light",
    "crowds":      "a dense crowd of people",
}

_SUBJECT_STOP = {"something", "anything", "everything", "nothing", "one",
                 "you", "it", "they", "there", "here", "way", "thing",
                 "things", "shot"}

_AUDIENCE_PATTERNS = [
    ("instagram", r"instagram|insta\b|ig feed"),
    ("competition", r"contest|competition|award|submission|jury|prize"),
    ("portfolio", r"portfolio|portfolio review"),
    ("print", r"\bprint\b|gallery|exhibition"),
]

# ── Recommended keywords (the brief editor's quick-add chips) ────────────────
# SINGLE SOURCE OF TRUTH for the UI's "tap to add intent" chips. Every entry
# is verified against the parser above — a chip must provably move the parsed
# Brief (pinned by tests/test_photo_brief.py), so the UI can never offer
# vocabulary the pipeline ignores (the "motion blur" chip was exactly that:
# invisible to every deterministic table).
RECOMMENDED_KEYWORDS: "dict[str, list[str]]" = {
    "Mood": [
        "rain", "wet", "fog", "mist", "overcast", "shadows",
        "dusk", "night", "blue hour", "neon",
        "golden hour", "sunset", "warm",
    ],
    "Subject": [
        "no people", "empty streets", "void",
        "vehicles only", "architecture only", "signs only",
        "reflections only", "food only", "crowds only",
    ],
    "Look": [
        "black and white", "monochrome", "vibrant", "saturated",
        "minimal", "negative space", "geometric", "symmetry", "patterns",
    ],
    "Audience": [
        "instagram feed", "competition", "portfolio", "gallery print",
    ],
}


def parse_subject_only(text: str):
    """Detect '<subject> only' in the brief. Returns (subject, siglip_query)
    or None. Known subjects map to a descriptive SigLIP phrase; anything else
    uses the user's own words — the text encoder is open-ended."""
    t = (text or "").lower()
    m = re.search(r"\b([a-z][a-z\s'-]{2,40}?)\s+only\b", t)
    if not m:
        return None
    subject = m.group(1).strip()
    # Strip conversational scaffolding around the subject noun:
    #   "i want the 5 photos to be vehicles only" → "vehicles"
    #   "5 photos of food stalls only"            → "food stalls"
    subject = re.sub(
        r"^(?:\d+\s+)?(?:the\s+|a\s+|an\s+)*"
        r"(?:want|wanted|need|needs|use|using|show|shows|showing|get|give|make|making|see|seeing|be\s+|of\s+)*\s*"
        r"(?:\d+\s+)?"
        r"(?:photos|pictures|shots|pics|frames|images)\s+"
        r"(?:to\s+be\s+|of\s+|that\s+are\s+|which\s+are\s+)*",
        "", subject).strip()
    subject = re.sub(r"\s+(?:photos|pictures|shots|pics|frames|images)$", "", subject).strip()
    subject = re.sub(r"^(?:the\s+|a\s+|an\s+|to\s+be\s+)+", "", subject).strip()
    subject = re.sub(r"^(?:use|using|want|wanted|need|show|showing|get|give|make|making|should\s+be|is\s+|are\s+)\s+", "", subject).strip()
    if len(subject) < 3 or subject in _SUBJECT_STOP:
        return None
    return subject, SUBJECT_SYNONYMS.get(subject, subject)


def detect_audience(text: str):
    t = (text or "").lower()
    for name, pat in _AUDIENCE_PATTERNS:
        if re.search(pat, t):
            return name
    return None


# ── The Brief ────────────────────────────────────────────────────────────────

@dataclass
class Brief:
    text: str
    mood: str = "neutral"
    geometry: str = "Normal"                      # High | Normal
    hard_filters: list = field(default_factory=list)   # e.g. ["no_people"]
    keywords: list = field(default_factory=list)
    subject: str | None = None                    # from "<subject> only"
    subject_query: str | None = None              # SigLIP phrase for it
    subject_mode: str | None = None               # "hard" (filter) when set
    audience: str | None = None                   # instagram | competition | ...
    expanded: bool = False                        # was LLM-expanded?

    def to_dict(self) -> dict:
        d = asdict(self)
        # Legacy keys the frontend already renders (keep until the UI migrates).
        d["LIGHTING_MOOD"] = self.mood
        d["GEOMETRIC_PRIORITY"] = self.geometry
        d["HARD_FILTER_PEOPLE"] = "no_people" in self.hard_filters
        d["BRIEF_KEYWORDS"] = self.keywords
        if self.subject:
            d["SUBJECT_ONLY"] = self.subject
        return d


def build_brief(text: str, expanded: bool = False) -> Brief:
    """Parse the user's brief into a Brief. Deterministic (keyword tables +
    subject regex, no models) — the same floor the live preview shows."""
    text = (text or "").strip()
    rs = _keyword_rule_set(text)
    hard_filters = []
    if rs.get("HARD_FILTER_PEOPLE"):
        hard_filters.append("no_people")
    subj = parse_subject_only(text)
    return Brief(
        text=text,
        mood=rs.get("LIGHTING_MOOD", "neutral"),
        geometry=rs.get("GEOMETRIC_PRIORITY", "Normal"),
        hard_filters=hard_filters,
        keywords=rs.get("BRIEF_KEYWORDS") or [],
        subject=subj[0] if subj else None,
        subject_query=subj[1] if subj else None,
        subject_mode="hard" if subj else None,
        audience=detect_audience(text),
        expanded=expanded,
    )


# ── SSE done-payload contract ────────────────────────────────────────────────
# The frontend renders from these keys; the run payload must carry all of
# them. Enforced at run time in creative_director's return and pinned here so
# backend/frontend cannot drift apart silently.
REQUIRED_DONE_KEYS = {
    "outputs", "output_dir", "alt_outputs", "rule_set", "subject",
    "director_fallback", "selection", "total", "success", "failed",
    "anchor_path", "brief", "timings",
}
# Documented done-payload keys (tests/test_photo_brief.py pins the contract
# against this set, and the pipeline enforces it at run time).

