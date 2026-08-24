"""
V2 grading pipeline — Frontier 2026 (Pure Vision Regression Stack).

Step 1  Discover images in the folder.
Step 2  SigLIP-2 ViT-g/14 NaFlex → 1536-d embeddings + brief prompt embedding.
Step 3  Detect duplicates via cosine similarity (0.88 threshold).
Step 4  Vision Regression Stack.
            4a  SpecVLMPipeline: per-aspect breakdown (Composition, Lighting, Narrative…).
            4b  UniQAHead: pyiqa 'uniqa' unified backbone with YOLO11s-seg routing.
                           Route 1 (empty scene): sqrt(comp × light) from SpecVLM.
                           Route 2 (layered frame): UniQA on subject crop.
            4c  Semantic anchor: SigLIP-2 dot-product vs. user-brief embedding.
            4d  Score fusion: q * 0.75 + fa * 0.25 (VLP) or q (standard).
Step 5  PersonalHead adjusts scores by learned user preference (if weights present),
            via a confidence-adaptive blend: taste weight scales from a 0.20 floor
            (neutral head → identical to the legacy 0.80/0.20) up to a ceiling as the
            head's opinion strengthens, so taste only sways grades where it has coverage.
Step 6  Absolute grade buckets: score ≥ 0.60 → Strong / 0.41–0.60 → Mid / < 0.41 → Weak.
            (Per-batch relative/quantile bucketing was removed 2026-06 — the grade
            reflects the photo itself, not its rank in the batch. Do NOT reintroduce.)
Step 7  Write to LanceDB (1536-d IVF-PQ schema).
Step 8  Build gallery response (V1-compatible keys).
Step 9  NSGA-III multi-objective sequence: Score × Semantic_Vibe
            × Portfolio_Diversity × Aspect_Ratio_Balance.

VRAM Protocol (4-6 GB cards):
    SigLIP-2 FP16 (~4.5 GB singleton) + IQA heads (~2 GB peak during scoring)
    → release_iqa_models() after Step 4b → ~4.5 GB for LanceDB + NSGA-III.
    IQA singletons released after each run; SigLIP-2 persists for fast repeat runs.
"""
from __future__ import annotations

import os
import gc
import json
import hashlib
import threading
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from raw_support import RAW_EXTS as _RAW_EXTS
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"} | _RAW_EXTS

STRONG_THRESH = 0.60
MID_THRESH    = 0.41

GRADE_STRONG = "Strong ✅"
GRADE_MID    = "Mid ⚠️"
GRADE_WEAK   = "Weak ❌"


# ── Per-tier cache namespacing (ORDER-CRITICAL for tier switching) ───────────
# The probe / archetype / people caches used to be keyed by PROMPT HASH ONLY.
# That is fine while there is one encoder, but each quality tier emits a
# different embedding dimension (Pro 1536, Balanced 1024, Fast 768). Switching
# tiers would therefore load, say, 1536-d archetype vectors and multiply them
# against 768-d image embeddings — a shape mismatch in the middle of Step 4d
# fusion (and in creative_director's people-similarity gate). The prompt hash
# cannot catch that, because the prompts did not change; the ENCODER did.
#
# Namespacing every cached embedding file by tier means each tier keeps its own
# vectors, nothing collides, and switching back and forth does not force a
# recompute each time. Loads are ALSO shape-checked, so a stale file from any
# other source is ignored rather than trusted.
class _TasteTierMismatch(Exception):
    """The taste head belongs to a different quality tier — skip it quietly.

    Distinct from a real failure so the handler does not print a traceback for
    what is an expected consequence of changing tiers.
    """


def _tier_tag() -> str:
    t = os.environ.get("SIGLIP_TIER", "high").strip().lower()
    return t if t in ("high", "mid", "low") else "high"


def _tier_cache_name(stem: str, ext: str) -> str:
    """'probe_embs','.npz' -> 'probe_embs.npz' (high) | 'probe_embs_mid.npz'.

    'high' keeps the historical unsuffixed name so existing caches stay valid.
    """
    tag = _tier_tag()
    return f"{stem}{ext}" if tag == "high" else f"{stem}_{tag}{ext}"


def _np2py(v):
    """Convert numpy scalar to Python primitive; pass-through for everything else."""
    return v.item() if hasattr(v, "item") else v


def _sanitize_bd(d: dict) -> dict:
    """Return a copy of breakdown dict with all values cast to Python primitives."""
    return {k: _np2py(v) for k, v in d.items()}


# ── Bounded crash diagnostics ────────────────────────────────────────────────
# The old handlers did `for k, v in locals().items(): print(f"{k}: {v}")`.
# At the points where they fire, locals() holds `gallery` / `lance_records` /
# `cached_rows` — lists of thousands of dicts, each carrying a 1536-element
# Python-float `embedding` list. repr() on a Python list does NOT truncate, so
# formatting one of those builds a multi-GB string, and it does so at the exact
# moment the run is already in trouble. That turned every *handled* exception
# into an unrecoverable MemoryError (and dumped the same GBs into crash.log).
#
# This keeps the diagnostic value — which variables existed, their type and
# size — with a hard cap on how much is ever materialised.
_DUMP_SKIP  = {"self", "encoder_model"}
_DUMP_MAXLEN = 200   # chars per value


def _describe(v) -> str:
    """One short, allocation-bounded line describing a value."""
    try:
        if isinstance(v, np.ndarray):
            return f"ndarray shape={v.shape} dtype={v.dtype}"
        if isinstance(v, (list, tuple, set, dict)):
            return f"{type(v).__name__} len={len(v)}"
        if isinstance(v, (str, bytes)):
            return f"{type(v).__name__} len={len(v)}: {v[:_DUMP_MAXLEN]!r}"
        if isinstance(v, (int, float, bool, type(None))):
            return repr(v)
        # Unknown object: repr() it, but never let a huge __repr__ through.
        r = repr(v)
        return r if len(r) <= _DUMP_MAXLEN else r[:_DUMP_MAXLEN] + "…"
    except Exception as _e:
        return f"<undescribable {type(v).__name__}: {_e}>"


def _dump_locals(loc: dict) -> None:
    """Print a bounded summary of a frame's locals for crash diagnosis."""
    try:
        print("\n--- CRASH LOCAL VARIABLES (bounded summary) ---")
        for key in sorted(loc):
            if key in _DUMP_SKIP:
                continue
            print(f"{key}: {_describe(loc[key])}")
        print("-----------------------------------------------\n")
    except Exception:
        pass   # diagnostics must never mask the original error


def _generate_brief_variants(brief: str) -> list[str]:
    """
    Generate 3–5 semantically equivalent phrasings of the CD brief.

    Encoding all variants and averaging their embeddings produces a more robust
    semantic anchor than a single-text encoding — reduces sensitivity to exact
    wording and covers both noun-phrase and descriptive-sentence formulations.
    """
    t = brief.strip()
    candidates = [
        t,
        f"street photography: {t}",
        f"photographic mood and visual atmosphere: {t}",
        f"a photograph that captures {t}",
        f"visual style and aesthetic: {t}",
    ]
    seen: set[str] = set()
    out:  list[str] = []
    for v in candidates:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:5]


# ── Low-contrast genre reference prompts for TOPIQ bias correction ────────────
# SigLIP-2 cosine similarity to these is computed per-photo; if max > 0.70,
# TOPIQ NR's flat-texture penalty is partially reversed (up to ×1.20 correction).
_GENRE_REF_PROMPTS: list[str] = [
    "minimalist photography with clean empty surfaces and geometric simplicity",
    "fine art architectural photography with stark geometric structure and symmetry",
    "liminal space photograph — empty corridor, atmospheric, quietly unsettling",
]

# Fine-art pictorialism anchor — 3-prompt ensemble averaged and L2-normalised.
# Used for Vintage Lens Protocol weight adjustment and Soft-Focus Protection Gate.
_FINE_ART_PROMPTS: list[str] = [
    "A fine-art street photograph with cinematic low-light chiaroscuro.",
    "Intentional vintage lens softness, pictorialism aesthetic.",
    "Atmospheric moody street scene, artistic analog film texture, deep shadows.",
]

# ── Street photography technique probes ───────────────────────────────────────
# Multi-probe SigLIP aesthetic scoring: positive probes capture street photography
# excellence across decisive moment, light, composition, and narrative dimensions.
# Negative probes capture common failures. Score = mean(pos_sims) - mean(neg_sims),
# batch-normalised to [0,1] and stored as street_aesthetic_scores (N,).
_STREET_POS_PROBES: list[str] = [
    # ── Intent-aware amnesty: deliberately soft / vintage / film aesthetics ────
    # These positives let genuinely artistic soft-focus, film-grain, and vintage
    # -glass shots out-score the "blurry/amateur" negatives, so they are NOT
    # dragged to Weak just for being non-clinical. Pairs with the forgiving blur
    # bouncer in early_exit_gate.py (only catastrophic blur is hard-rejected).
    "an artistic photograph with intentional vintage soft focus, film grain, and dreamy atmosphere",
    "expressive intentional motion blur conveying movement and energy, fine-art street photography",
    "nostalgic analog film look, soft vintage lens rendering, painterly grain, mood over sharpness",
    # ── Decisive moment (Cartier-Bresson tradition) ───────────────────────────
    "decisive moment peak action frozen gesture street photography",
    "split second timing human gesture caught mid-action candid street",
    "fraction of second frozen motion peak expression street documentary",
    "anticipation camera position waiting decisive moment street photography",
    "serendipitous accident chance encounter unexpected street photograph",
    "simultaneous actions multiple subjects peak moment street",
    "gesture hand movement body language caught decisive candid",
    "facial expression fleeting emotion caught candid unposed street",
    "child spontaneous play action energy street documentary",
    "elderly person gesture wisdom character street portrait candid",
    "direct eye contact subject aware camera confrontational street portrait",
    "multiple unrelated subjects accidentally composing together urban crowd",

    # ── Human condition ───────────────────────────────────────────────────────
    "raw human emotion grief joy laughter tears urban documentary",
    "social interaction two strangers encounter urban street moment",
    "isolation loneliness crowded city individual alone urban",
    "intimacy couple love connection public space street photography",
    "class contrast wealth poverty juxtaposition urban social documentary",
    "labor working hands street vendor market commerce documentary",
    "commuter exhaustion subway bus daily grind urban documentary",
    "protest march demonstration collective energy signs banners street",
    "celebration festival street party collective joy urban documentation",
    "conflict argument confrontation tension street documentary moment",
    "homeless figure urban poverty social documentary street photography",
    "children playing street game spontaneous joy urban neighborhood",
    "elderly couple slow walk tenderness urban street scene",
    "businessman rushing corporate speed urban street documentary",
    "immigrant cultural identity traditional dress urban western contrast",
    "public private boundary intimate space urban street documentary",
    "generational contrast old young together urban documentary",
    "cultural fusion diverse background urban melting pot street",
    "disability adaptation mobility urban environment documentary",
    "aging elderly grace dignity urban portrait candid documentary",
    "youth energy rebellion defiance urban street documentary",

    # ── Light mastery ─────────────────────────────────────────────────────────
    "chiaroscuro dramatic shaft of light dark shadow pool street",
    "golden hour magic light long shadow warm street photography",
    "blue hour twilight transition day night urban street atmosphere",
    "harsh noon overhead light graphic shadow pool street",
    "backlight silhouette rim light translucent halo street portrait",
    "window light soft directional interior portrait street scene",
    "neon artificial light night rain reflection cinematic street",
    "streetlight pool darkness night figure atmospheric street",
    "foggy diffused soft light atmosphere mystery street scene",
    "overcast flat even light portrait street subtle shadow",
    "dappled light through leaves tree shadow urban street",
    "reflected light bounce fill shadow urban canyon building",
    "shooting into sun lens flare diffraction star burst street",
    "high key bright minimalist urban street intentional overexposure",
    "low key dark moody night street atmospheric intentional",
    "dawn first light empty street solitude urban atmosphere",
    "last light dusk orange purple sky urban silhouette",
    "rain light refraction wet street prismatic color",
    "snow diffused reflection white ground uplighting face street",
    "fire emergency red blue police siren light night street",
    "available light only no flash cinematic natural shadow street",
    "fluorescent mixed light color cast indoor adjacent street",
    "candlelight warm flame artificial intimate light street documentary",

    # ── Shadow work ───────────────────────────────────────────────────────────
    "long afternoon shadow graphic silhouette ground street",
    "fence bars shadow prison graphic pattern subject street",
    "venetian blind shadow stripe graphic pattern portrait street",
    "tree branch shadow organic pattern ground urban scene",
    "window light shadow square pattern floor interior street",
    "shadow only abstract no visible source mysterious street",
    "multiple overlapping shadows complex graphic abstract urban",
    "photographer shadow self-portrait compositional device street",
    "building shadow cool dark subject warm light contrast street",
    "shadow leading edge pointing subject directional composition street",
    "umbrella shadow circular pattern ground rain street",
    "staircase shadow repeating diagonal graphic urban abstract",
    "bridge shadow underpass dark light transition street",
    "pillar column shadow stripe rhythm urban architecture street",
    "intersecting shadows abstract geometry urban graphic street",

    # ── Reflection ───────────────────────────────────────────────────────────
    "puddle mirror reflection perfect symmetry wet street inverted world",
    "shop window reflection layered interior exterior street scene",
    "car side mirror urban scene miniature contained world",
    "sunglasses lens reflection urban scene miniature portrait",
    "river canal water reflection distorted impressionist urban",
    "glass building mirror facade city reflection abstract",
    "wet pavement oil slick rainbow iridescent reflection urban",
    "elevator door polished metal reflection urban abstract",
    "window glass ghostly double exposure street overlay",
    "marble floor reflection building column symmetry urban",
    "rain drop world inside drop urban macro close-up",
    "shooting through rain condensation glass window soft layered",
    "coffee puddle spill reflection urban macro street abstract",
    "phone screen reflection face urban meta documentary",

    # ── Compositional mastery ─────────────────────────────────────────────────
    "dramatic leading lines converging perspective urban geometry shadow",
    "layered depth foreground midground background street scene",
    "frame within frame doorway arch window isolating subject",
    "strong silhouette backlit figure dramatic sky contrast",
    "negative space isolation minimalist subject stark urban background",
    "tilted diagonal horizon dynamic tension unstable Winogrand energy",
    "overhead aerial bird's eye view people pattern abstraction street",
    "shoot from hip extreme low angle unguarded candid perspective",
    "centered symmetrical subject mirror image balanced composition street",
    "diagonal line energy tension composition street photography",
    "converging perspective vanishing point urban geometry street",
    "S-curve winding path composition organic urban street",
    "layered planes depth foreground midground background street",
    "rule of odds three five seven subjects composition street",
    "visual weight balance asymmetric composition street photography",
    "split composition divided frame two contrasting worlds street",
    "circular composition round frame visual flow street photography",
    "triangular composition stable three points street photography",

    # ── Framing devices ───────────────────────────────────────────────────────
    "doorway threshold frame subject transitional space street",
    "arch architectural frame subject below street photography",
    "tunnel circular frame light end darkness street",
    "tree branches natural organic frame sky subject street",
    "crowd parting gap aperture frame distant subject street",
    "fence bars iron frame subject urban enclosure street",
    "railing foreground blur frame midground subject street",
    "scaffold urban construction frame worker street documentary",
    "car window frame street scene urban travel documentary",
    "overhanging roof eave frame street below urban",

    # ── Geometry and pattern ──────────────────────────────────────────────────
    "repeating pattern urban architecture rhythm visual music street",
    "alternating light dark rhythm stripe pattern urban abstract",
    "radial pattern starburst circular geometry urban abstract",
    "grid pattern window building facade urban abstract geometry",
    "texture rough brick concrete worn surface urban abstract street",
    "escalator stairs repeating geometric form urban abstract",
    "fire escape iron geometry vertical rhythm urban New York",
    "parking lot lines white yellow asphalt geometry abstract urban",
    "crosswalk zebra stripe graphic pattern urban abstract street",
    "manhole cover iron circle geometry urban macro street",
    "cobblestone texture pattern street surface urban history",
    "tile mosaic pattern floor surface urban abstract geometry",
    "chain link fence diamond pattern urban texture abstract",
    "corrugated metal surface texture pattern urban industrial",
    "curve organic versus straight industrial urban contrast geometry",

    # ── Motion and time ───────────────────────────────────────────────────────
    "motion blur long exposure light trail night street urban",
    "panning technique sharp subject blurred background street",
    "multiple exposure ghosting temporal layering street photography",
    "freeze action high shutter water splash street urban",
    "slow shutter ghost figure movement temporal street",
    "traffic light trail red white long exposure night urban",
    "rain drop frozen high shutter spray splash street",
    "bicycle wheel spokes blur motion urban commute street",
    "running crowd motion energy blur street documentary",
    "motion blur dynamic energy panning sharp subject street photography",

    # ── Weather and atmosphere ────────────────────────────────────────────────
    "heavy rain storm urban umbrella wet street documentary",
    "light drizzle atmospheric wet pavement reflective street",
    "dense fog mystery atmosphere subject emerging uncertain street",
    "light mist soft haze distance atmosphere urban landscape",
    "snow falling white dots urban scene silent street",
    "blizzard white out urban reduced visibility dramatic street",
    "heat haze shimmer summer asphalt urban distortion",
    "autumn leaves falling urban street seasonal change documentary",
    "wind blowing hair clothes movement energy urban street",
    "humidity steam manhole cover rising fog urban winter street",
    "smoke cigarette atmospheric figure urban moody street",
    "dust particle light shaft beam urban construction street",
    "thunder dark sky dramatic clouds urban street approaching storm",
    "rainbow after rain urban building colorful unexpected street",

    # ── Time of day ───────────────────────────────────────────────────────────
    "pre-dawn dark empty urban street solitary figure atmospheric",
    "sunrise first light empty street long shadow golden",
    "morning rush hour commuter crowd energy urban documentation",
    "midday harsh light graphic shadow empty street",
    "afternoon slant light warm golden shadow growing street",
    "late afternoon magic hour warm glow shadow long urban",
    "sunset silhouette orange sky urban building rooftop",
    "dusk blue hour twilight urban light transition street",
    "early evening artificial light window warm interior street",
    "night city neon reflection wet pavement cinematic street",
    "late night last person empty street solitude urban",
    "midnight blue black street atmospheric lonely urban",

    # ── Urban environment and structure ──────────────────────────────────────
    "subway underground platform artificial light urban transit documentary",
    "bus stop shelter waiting crowd urban transit street",
    "market bazaar vendor stall commerce human activity street",
    "restaurant cafe exterior seating urban social gathering street",
    "park urban green space nature city contrast human activity",
    "playground children urban neighborhood community street",
    "construction site worker labor urban development documentary",
    "demolition rubble decay urban change documentary street",
    "graffiti mural street art urban texture context",
    "advertisement billboard poster street commercial message irony",
    "fire escape iron vertical rhythm brick building urban structure",
    "laundry hanging clothesline urban neighborhood domestic life",
    "urban rooftop skyline view human scale architecture street",
    "escalator moving stairs repeating diagonal geometry urban abstract",
    "cafe window interior exterior boundary warmth cold observer observed",
    "phone booth urban retro communication solitary figure street",
    "stairs concrete urban structure human figure scale street",
    "underpass tunnel urban passage light dark transition street",
    "bridge walkway urban crossing human movement flow street",
    "alley narrow passage urban depth shadow street documentary",

    # ── Specific subjects ─────────────────────────────────────────────────────
    "dog animal urban pet companion street documentary humanity",
    "cat feral urban animal rooftop alley street documentary",
    "pigeon urban bird flock pattern street abstract",
    "street musician performance busker urban culture documentary",
    "vendor cart street food cook steam urban commerce",
    "newspaper reader analog information urban transit documentary",
    "phone user screen glow modern urban disconnected street",
    "sleeper urban bench exhaustion rest public space documentary",
    "jogger runner urban health park street morning documentary",
    "skateboarder youth urban sport trick movement street",
    "construction worker hard hat labor urban development street",
    "street preacher evangelist urban religion documentary",
    "mime statue performer human frozen urban art street",
    "market vendor commerce transaction exchange hands goods street life",
    "bicycle messenger urban speed delivery work documentary",
    "hands feet detail gesture labor texture street life",
    "protest demonstration collective human force signs emotion documentary",

    # ── Narrative and juxtaposition ───────────────────────────────────────────
    "visual irony juxtaposition contrasting elements urban storytelling",
    "environmental storytelling social documentary urban human condition",
    "scale contrast tiny human figure vast urban architecture street",
    "ironic signage text juxtaposition human subject urban contradiction",
    "old building new architecture temporal contrast urban transformation",
    "humor comedy absurdity human urban moment street photography",
    "beauty mundane ordinary elevated artistic urban street",
    "ugliness truth uncomfortable honest urban documentary street",
    "tenderness care affection public space urban documentary",
    "boredom waiting patience time passing urban documentary street",
    "excitement energy joy spontaneous urban celebration street",
    "contemplation introspection private public urban moment street",
    "connection disconnection paradox urban isolation crowd street",

    # ── Perspective and viewpoint ─────────────────────────────────────────────
    "worm's eye view extreme low angle upward perspective urban",
    "bird's eye overhead downward perspective urban pattern abstraction",
    "eye level human connection direct perspective street photography",
    "over shoulder voyeuristic perspective urban documentary street",
    "through gap crack narrow aperture selective reveal urban",
    "extreme foreground bokeh blur lead to sharp subject urban",
    "telephoto compression urban crowd density proximity street",
    "wide angle environmental context small figure urban street",
    "extreme close-up confrontational wide angle distortion face street",
    "shooting through crowd partial occlusion hidden subject mysterious depth",
    "lonely isolated figure vast empty urban landscape alienation modern",

    # ── Emotional resonance ───────────────────────────────────────────────────
    "hope aspiration upward look light urban optimism street",
    "despair defeat downcast figure urban shadow dark street",
    "love connection intimacy couple urban public space street",
    "serenity peace calm figure urban chaos contrast street",
    "pride dignity self-respect urban portrait documentary",
    "joy pure unguarded laughter urban spontaneous street",
    "grief mourning loss urban public emotion documentary",
    "nostalgia memory past urban retro analog feel street",
    "wonder awe small figure vast urban architecture street",
    "curiosity child urban exploration discovery street documentary",
    "exhaustion fatigue worker end of day urban documentary",
    "determination figure against odds urban street documentary",
    "fear uncertainty vulnerability urban figure dramatic street",
    "anger defiance resist raised fist urban documentary protest",

    # ── Abstract and experimental ─────────────────────────────────────────────
    "abstract urban shape color form line no literal subject",
    "intentional camera movement ICM painterly abstract street",
    "extreme grain noise texture abstract urban documentary",
    "high key white overexposed intentional graphic urban street",
    "low key dark underexposed intentional moody urban street",
    "long exposure water silk smooth contrast sharp architecture",
    "photo within photo urban meta self-reference street",
    "multiple exposure layered reality dreamlike urban street",

    # ── Provoke era / grain aesthetic (Moriyama tradition) ────────────────────
    "high grain texture intentional rough grainy bure boke black white street",
    "fragmented cropped reality partial face severed composition aggressive",
    "high contrast crushed blacks blown whites extreme tonal street",
    "snapshot aesthetic raw unpolished energy authentic street photography",

    # ── Color as language (Saul Leiter tradition) ─────────────────────────────
    "color dominates composition warm cool palette street photography",
    "saturated single color graphic impact urban street scene",
    "color harmony complementary palette urban environment street",
    "muted desaturated color palette melancholy urban street",
    "bold red yellow blue primary color urban graphic street",

    # ── Social documentary ────────────────────────────────────────────────────
    "social commentary injustice inequality urban documentary street",
    "urban decay abandonment empty building ghost town street",
    "gentrification old neighborhood new wealth change urban documentary",
    "community resilience survival dignity urban documentary street",
    "immigrant experience new life urban adaptation documentary",
    "tradition modernity clash urban cultural documentary street",
    "class distinction luxury poverty same frame urban contrast",
    "political rally crowd collective voice urban documentary",
    "sports crowd celebration urban energy collective joy",
    "fashion individual expression urban street style documentary",
    "subculture identity punk alternative urban youth street",

    # ── Technical mastery signals ─────────────────────────────────────────────
    "zone system exposure tonal range black white street photography",
    "film grain analog texture character black white street",
    "zone focus hyperfocal everything sharp documentary street",
    "push process high contrast grain deliberate aesthetic street",
    "35mm lens natural perspective candid street photography",
    "28mm wide angle environmental portrait street context",
    "85mm portrait compression urban background isolation street",
    # Master photographer visual languages
    "Vivian Maier self-portrait reflection mirror unexpected urban street",
    "Saul Leiter color painterly impressionist window rain soft street",
    "William Klein contact sheet raw energy urban aggressive New York",
    "Robert Frank road America outsider alienation melancholy documentary",
    "Josef Koudelka exile diaspora displacement theatrical street",
    "Martin Parr saturated British social documentary humor excess",
    "Joel Meyerowitz color light large format open street humanist",
    "Fan Ho Hong Kong atmosphere poetic shadow geometry street",
    "Shomei Tomatsu nuclear aftermath Japan social tension urban",
    "Helen Levitt street children chalk drawing New York spontaneous",
    # Specific urban subjects and scenes
    "shop mannequin window display uncanny human likeness urban street",
    "crosswalk pedestrian decisive crossing geometry urban street moment",
    "fire hydrant foreground red urban graphic texture street composition",
    "shoe cobbler repair craft labor hands worn leather urban documentary",
    "barber shop hair cut grooming urban community social documentary",
    "newspaper stand headline text irony juxtaposition urban street",
    "satellite dish rooftop informal urban connectivity documentary",
    "telephone wire silhouette bird perch urban nature line",
    "peeling paint wall urban decay layered texture history street",
    "narrow alley wash line laundry overhead crowded urban residential",
    # Additional light and atmosphere
    "lightning storm dramatic flash sky urban street documentary",
    "smoke stack industrial urban atmosphere documentary pollution street",
    "steam pipe rising winter urban cold breath figure street",
    "neon sign reflection bar club night urban cinematic street",
    "subway grate steam rising urban underground winter street",
    # Extended compositional mastery
    "Gestalt continuation visual flow eye movement urban composition",
    "figure ground reversal ambiguous positive negative space urban",
    "color temperature contrast warm cool split urban street",
    "bokeh orbs background lights night urban portrait street",
    "hyperfocal zone focus everything sharp gritty documentary street",
]

_STREET_NEG_PROBES: list[str] = [
    # Technical failures
    "blurry out of focus missed focus soft unsharp snapshot",
    "overexposed blown highlights washed out flat grey image",
    "underexposed muddy dark crushed shadows unreadable image",
    "harsh direct flash unflattering ugly artificial light snapshot",
    "extreme digital noise grain unintentional ugly sensor failure",
    "motion blur accidental missed timing not intentional soft",
    "chromatic aberration purple fringe lens quality distraction",
    "tilted horizon accidental not intentional sloppy framing",
    # Compositional failures
    "flat boring ordinary nothing happening mundane dull empty",
    "cluttered distracting chaotic background no clear subject",
    "completely empty scene no subject no moment no interest",
    "subject dead center static symmetric boring no visual tension",
    "too much empty space no subject lost frame wasted composition",
    "background subject same tone no separation flat merger",
    "trees telephone poles growing out of head amateur snapshot",
    "horizon line cutting head awkward framing snapshot error",
    # Authenticity failures
    "stiff posed artificial unnatural forced uncomfortable portrait",
    "tourist snapshot vacation ordinary travel record shot",
    "staged setup artificial scene directed not candid fake street",
    "selfie smartphone narcissistic no street context",
    "copy imitation derivative cliche obvious no original vision",
    # Processing failures
    "over processed HDR unnatural fake colors heavy vignette artificial",
    "heavily filtered Instagram aesthetic oversaturated artificial preset",
    "heavy vignette excessive darkening corners artificial processing",
    "oversaturated garish colors unnatural artificial street photography",
    # Context failures
    "generic portrait against plain wall studio no street context",
    "product photography clean sterile commercial no life no humanity",
    "food photography plate restaurant no street context",
    "fashion editorial artificial setup no authentic street life",
    "celebrity portrait controlled environment no street authenticity",
    "red eye flash reflection ugly snapshot amateur failure",
]

_EXIF_LOCK = threading.Lock()

# Module-level grader status — updated each run, read by /api/models/status.
_grader_status: dict = {
    "mode":        "idle",   # "idle" | "iqa_heads" | "clip_only"
    "verify_used": False,
    "photos_last": 0,
    "error":       None,
}

# ── SigLIP-2 singleton ────────────────────────────────────────────────────────
# Persists between grading runs — avoids 15-30 s weight-load overhead every run.
# VRAM budget: SigLIP-2 INT8 (~1.8 GB) + TOPIQ (~0.5 GB) = ~2.3 GB peak (safe).
# Released by release_grading_models() before Creative Mode loads LLMs.
_enc_singleton = None       # SigLIP2Encoder instance, or None
_text_emb_cache: dict = {}  # POS / NEG / ASPECT embeddings — static across runs

# Expected embedding dim for the active SIGLIP_TIER (Phase 2). Lets the pipeline
# accept 1024-/768-d encoders on weaker tiers instead of rejecting non-1536.
import os as _os_encdim
_ENC_DIM = {"high": 1536, "mid": 1024, "low": 768}.get(
    _os_encdim.environ.get("SIGLIP_TIER", "high").strip().lower(), 1536)

# ── Qwen2.5-VL singleton ──────────────────────────────────────────────────────
# Stays loaded between runs — first load takes 30-60 s (or downloads ~6 GB);
# subsequent runs reuse the resident weights instantly.
# SigLIP-2 is evicted before this loads, so VRAM budget is safe.
_qwen_singleton = None      # QwenVLMGrader instance, or None


def release_grading_models() -> None:
    """Evict all grading singletons (SigLIP-2 + IQA heads) before Creative Mode loads LLMs."""
    global _enc_singleton, _text_emb_cache, _qwen_singleton
    if _enc_singleton is not None:
        try:
            _enc_singleton.unload()
        except Exception:
            pass
        _enc_singleton = None
    if _qwen_singleton is not None:
        try:
            _qwen_singleton.unload()
        except Exception:
            pass
        _qwen_singleton = None
    _text_emb_cache.clear()
    try:
        from vision_grading_heads import release_iqa_models
        release_iqa_models()
    except Exception as _e_iqa:
        print(f"[v2] IQA singleton release skipped: {_e_iqa}")
    _vram_clear()
    print("[v2] All grading singletons released — VRAM freed for Creative Mode")


# EXIF timestamp reading now lives in pipeline_stages.read_exif_timestamps.


# ── VRAM helper ────────────────────────────────────────────────────────────────

def _vram_clear():
    """Release CUDA caches between pipeline phases."""
    try:
        from vram_manager import VRAMManager
        VRAMManager.purge_vram()
    except Exception:
        gc.collect()
        try:
            import torch
            # is_initialized() FIRST and alone. The guard used to read
            # `is_available() and is_initialized()`, which defeats itself: it is
            # is_available() that initialises CUDA, and it was evaluated first.
            # If CUDA was never initialised in this process there is nothing to
            # purge, so the check is also sufficient on its own. The grade worker
            # is intentionally CUDA-free (SigLIP and IQA are both isolated in
            # subprocesses); creating a context here makes the parent fault with
            # 0xC0000005 when a child exits.
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass


# ── Progress ticker for silent blocking phases ───────────────────────────────
# The SigLIP encode and the IQA scoring both run in an ISOLATED subprocess whose
# per-batch progress cannot cross back to this process's progress callback — so
# `_p` jumps 0.07 → 0.47 (encode) and stalls at 0.66 (IQA) with a multi-minute
# silent gap in between. On a large or RAM-tight folder the bar looks frozen (the
# "stuck at 7 %" report), and worse, the silence can outlast the client's no-data
# watchdog and make it abort a grade that is still running. This ticker fills the
# gap: a daemon thread emits an ever-advancing, asymptotic fraction between
# `start` and `end` while the blocking call runs. The curve approaches `end` but
# never reaches it, so it never overshoots the real milestone the caller emits on
# completion, and the periodic emits keep the SSE stream alive.
class _ProgressTicker:
    def __init__(self, progress, start: float, end: float, label: str,
                 tau: float = 30.0, interval: float = 1.5):
        self._p        = progress or (lambda f, d: None)
        self._start    = float(start)
        self._end      = float(end)
        self._label    = label
        self._tau      = max(1.0, float(tau))
        self._interval = max(0.2, float(interval))
        self._stop     = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_ProgressTicker":
        import time as _time, math as _math

        def _run() -> None:
            t0 = _time.monotonic()
            # Wait first so we never fight the caller's own start-of-phase emit.
            while not self._stop.wait(self._interval):
                el   = _time.monotonic() - t0
                frac = self._start + (self._end - self._start) * (1.0 - _math.exp(-el / self._tau))
                try:
                    self._p(round(frac, 3), self._label)
                except Exception:
                    pass  # progress reporting must never break the grade

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return False


def _dedup_chunk_size(n: int) -> int:
    """Row-block size for duplicate detection — FIXED for reproducibility.

    The similarity PAIRS this finds are chunk-independent (proven in the
    equivalence tests), but the ORDER union-find merges them is not: a different
    block boundary picks a different cluster ROOT, and the root is the frame
    marked eligible for composition analysis. So a RAM-derived block size
    silently changed which photo in a burst got analysed, and therefore its
    grade.

    Peak memory is O(chunk x n) floats; 512 rows against 10k photos is ~20 MB,
    which is nothing next to the encoder. FRAMEGRADE_DEDUP_CHUNK overrides.
    """
    _env = os.environ.get("FRAMEGRADE_DEDUP_CHUNK")
    if _env:
        try:
            return max(64, min(int(_env), max(n, 1)))
        except ValueError:
            pass
    return max(64, min(512, max(n, 1)))


# ── IQA subprocess bridge ────────────────────────────────────────────────────
# run_vision_heads (pyiqa topiq_nr + YOLO routing + composition heads) loads a GPU
# model. Running it directly inside the multiprocessing grade worker is what caused
# the recurring 0xC0000005 windowed crash. Like the SigLIP encode, we run it in an
# ISOLATED subprocess (src/iqa_worker.py) that loads the model, scores, and exits —
# so the grade worker itself never touches CUDA. Mirrors siglip2_encoder._run.
_IQA_WORKER = Path(__file__).resolve().parent / "iqa_worker.py"


def _iqa_via_subprocess(
    image_paths, image_embeddings, prompt_embedding, clip_scores,
    genre_ref_embs, lum_stats, comp_eligible_paths, vlm_breakdowns,
) -> dict:
    """Run vision_grading_heads.run_vision_heads in an isolated subprocess.
    Returns the same dict shape. Raises RuntimeError on subprocess failure so the
    caller can degrade cleanly (never a hard worker crash)."""
    import sys as _sys, json as _json, tempfile as _tf, subprocess as _sp, time as _time
    import win_job as _wj
    _root = Path(__file__).resolve().parent.parent
    _crash_log = _root / "crash.log"
    _fd, in_json = _tf.mkstemp(suffix=".iqa.json"); os.close(_fd)
    in_npz   = in_json + ".in.npz"
    out_npz  = in_json + ".out.npz"
    out_json = in_json + ".out.json"
    try:
        _arrs = {
            "image_embeddings": np.asarray(image_embeddings, dtype=np.float32),
            "clip_scores":      np.asarray(clip_scores,      dtype=np.float32),
        }
        if prompt_embedding is not None:
            _arrs["prompt_embedding"] = np.asarray(prompt_embedding, dtype=np.float32)
        if genre_ref_embs is not None:
            _arrs["genre_ref_embs"] = np.asarray(genre_ref_embs, dtype=np.float32)
        np.savez(in_npz, **_arrs)
        with open(in_json, "w", encoding="utf-8") as _f:
            _json.dump({
                "image_paths":         list(image_paths),
                "lum_stats":           [list(t) for t in (lum_stats or [])],
                "comp_eligible_paths": list(comp_eligible_paths or []),
                "vlm_breakdowns":      vlm_breakdowns or [],
            }, _f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        env = dict(os.environ); env.setdefault("PYTHONIOENCODING", "utf-8")
        # Retries: mirrors siglip2_encoder._run — the isolated GPU subprocesses
        # occasionally hit a transient "CUDA error: out of memory" / "device(s)
        # busy" at model load (WDDM contention with a sibling GPU subprocess) that
        # clears on the very next attempt. One retry turns that into a self-healing
        # blip instead of a failed grade.
        _last_rc = None
        for _attempt in range(1, 3):
            print(f"[v2] IQA subprocess: scoring {len(image_paths)} images (isolated GPU) attempt={_attempt}…", flush=True)
            with open(_crash_log, "a", encoding="utf-8", errors="replace") as _lf:
                r = _wj.run(
                    [_sys.executable, str(_IQA_WORKER), in_npz, in_json, out_npz, out_json],
                    env=env, cwd=str(_root), stdout=_lf, stderr=_lf, timeout=3600,
                )
            if r.returncode == 0 and os.path.exists(out_npz):
                break
            _last_rc = r.returncode
            if _attempt < 2:
                print(f"[v2] IQA subprocess attempt {_attempt} failed (rc={r.returncode}) — retrying", flush=True)
                _time.sleep(1.5)
        if r.returncode != 0 or not os.path.exists(out_npz):
            raise RuntimeError(f"IQA subprocess failed after 2 attempts (exit {_last_rc}) — see crash.log")
        _q = np.load(out_npz)["quality"]
        with open(out_json, encoding="utf-8") as _f:
            _pl = _json.load(_f)
        return {
            "quality": _q, "tech": _q, "aesthetic": _q,
            "breakdowns":            _pl.get("breakdowns", []),
            "composition_overrides": _pl.get("composition_overrides", {}),
            "chiaroscuro_flags":     _pl.get("chiaroscuro_flags", {}),
            "person_detected":       _pl.get("person_detected", {}),
            "subject_bboxes":        _pl.get("subject_bboxes", {}),
        }
    finally:
        for _tmp in (in_json, in_npz, out_npz, out_json):
            try: os.unlink(_tmp)
            except Exception: pass


def _iqa_resumable(
    image_paths, image_embeddings, prompt_embedding, clip_scores,
    genre_ref_embs, lum_stats, comp_eligible_paths, vlm_breakdowns,
    ckpt_key: str = "", progress=None,
) -> dict:
    """IQA in resumable slices, so a killed cull does not lose the whole pass.

    WHY THIS IS SAFE: the only cross-image step in IQA scoring is
    _batch_normalize, and that is a FIXED affine (centre 0.40, gain 2.5) — not
    batch statistics. A photo's score therefore does not depend on which other
    photos share its slice, so slicing is score-identical to one big call.
    Verified in tests/test_iqa_resume.py.

    Each completed slice is written to cache/iqa_ckpt/<key>.json. On a re-run
    those photos are skipped, so a 5 000-photo import that dies at 60 % resumes
    from 60 % instead of starting over. The checkpoint is deleted once the grade
    commits to LanceDB.

    Slice size trades resumability against re-loading the detector per slice;
    FRAMEGRADE_IQA_SLICE tunes it (0 disables slicing entirely).
    """
    _p = progress or (lambda f, d: None)
    n = len(image_paths)
    if n == 0:
        return _iqa_via_subprocess(image_paths, image_embeddings, prompt_embedding,
                                   clip_scores, genre_ref_embs, lum_stats,
                                   comp_eligible_paths, vlm_breakdowns)

    try:
        _slice = int(os.environ.get("FRAMEGRADE_IQA_SLICE", "400"))
    except ValueError:
        _slice = 400
    if _slice <= 0 or n <= _slice:
        # Small job: one call, but still checkpoint it so a later stage crash
        # (fusion, LanceDB) does not throw the IQA work away.
        out = _iqa_via_subprocess(image_paths, image_embeddings, prompt_embedding,
                                  clip_scores, genre_ref_embs, lum_stats,
                                  comp_eligible_paths, vlm_breakdowns)
        _iqa_ckpt_save(ckpt_key, image_paths, out)
        return out

    done = _iqa_ckpt_load(ckpt_key)
    todo = [i for i, p in enumerate(image_paths) if p not in done]
    if done:
        print(f"[v2] IQA resume: {n - len(todo)}/{n} already scored, "
              f"{len(todo)} remaining")

    for s0 in range(0, len(todo), _slice):
        idx = todo[s0:s0 + _slice]
        sl_paths = [image_paths[i] for i in idx]
        _p(0.66, f"Scoring image quality — {n - len(todo) + s0}/{n} photos…")
        part = _iqa_via_subprocess(
            image_paths         = sl_paths,
            image_embeddings    = np.asarray(image_embeddings)[np.asarray(idx, dtype=np.intp)],
            prompt_embedding    = prompt_embedding,
            clip_scores         = np.asarray(clip_scores)[np.asarray(idx, dtype=np.intp)],
            genre_ref_embs      = genre_ref_embs,
            lum_stats           = [lum_stats[i] for i in idx] if lum_stats else None,
            comp_eligible_paths = comp_eligible_paths,
            vlm_breakdowns      = [vlm_breakdowns[i] for i in idx] if vlm_breakdowns else None,
        )
        _iqa_ckpt_save(ckpt_key, sl_paths, part)
        done = _iqa_ckpt_load(ckpt_key)

    # Reassemble in the caller's original order.
    quality = np.array([float(done.get(p, {}).get("q", 0.5)) for p in image_paths],
                       dtype=np.float32)
    breakdowns = [dict(done.get(p, {}).get("bd", {})) for p in image_paths]
    comp, chi, per, bbox = {}, {}, {}, {}
    for p in image_paths:
        e = done.get(p, {})
        if e.get("co") is not None: comp[p] = e["co"]
        if e.get("ch") is not None: chi[p] = e["ch"]
        if e.get("pd") is not None: per[p] = e["pd"]
        if e.get("bb"):             bbox[p] = e["bb"]
    return {"quality": quality, "tech": quality, "aesthetic": quality,
            "breakdowns": breakdowns, "composition_overrides": comp,
            "chiaroscuro_flags": chi, "person_detected": per,
            "subject_bboxes": bbox}


def _iqa_ckpt_path(key: str) -> Path:
    d = Path(__file__).resolve().parent.parent / "cache" / "iqa_ckpt"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key or 'default'}.json"


def _iqa_ckpt_load(key: str) -> dict:
    try:
        p = _iqa_ckpt_path(key)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except Exception as exc:
        print(f"[v2] IQA checkpoint unreadable ({exc}) — starting fresh")
    return {}


def _iqa_ckpt_save(key: str, paths, out: dict) -> None:
    """Fold one slice's results into the checkpoint (atomic)."""
    try:
        cur = _iqa_ckpt_load(key)
        q  = np.asarray(out.get("quality", []), dtype=np.float32)
        bd = out.get("breakdowns", []) or []
        co = out.get("composition_overrides", {}) or {}
        ch = out.get("chiaroscuro_flags", {}) or {}
        pd = out.get("person_detected", {}) or {}
        bb = out.get("subject_bboxes", {}) or {}
        for i, p in enumerate(paths):
            cur[p] = {
                "q":  float(q[i]) if i < len(q) else 0.5,
                "bd": _sanitize_bd(bd[i]) if i < len(bd) and isinstance(bd[i], dict) else {},
                "co": co.get(p), "ch": ch.get(p), "pd": pd.get(p), "bb": bb.get(p),
            }
        f = _iqa_ckpt_path(key)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cur, default=_np2py), encoding="utf-8")
        tmp.replace(f)
    except Exception as exc:
        print(f"[v2] IQA checkpoint save skipped ({exc})")


def _iqa_ckpt_clear(key: str) -> None:
    try:
        _iqa_ckpt_path(key).unlink(missing_ok=True)
    except Exception:
        pass


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_v2(
    folder_path: str,
    preset: str = "Classic Street",
    force_rescan: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
    mogco_target: int = 5,
    scan_mode: bool = False,
    sample_limit: int = 0,
    deep_grade: bool = False,
) -> dict:
    """
    Run the full V2 Vision Regression pipeline on `folder_path`.

    Returns:
        gallery         list[dict]   per-photo result (V1-compatible keys + reasoning_log)
        mogco_sequence  list[dict]   NSGA-III output
        strong / mid / weak int      counts
        total           int
        pipeline        "v2"
    """
    # Per-stage peak-RSS trace. wrap_progress keeps _p's exact (frac, desc)
    # signature and forwards unchanged, so no stage call site or SSE text moves.
    try:
        import ram_probe as _ram_probe
        _ram_probe.reset()
        _p = _ram_probe.wrap_progress(progress)
    except Exception:
        _ram_probe = None
        _p = progress or (lambda f, d: None)

    # ── Whole-cull memory check ──────────────────────────────────────────────
    # The encoder's own floor only guards the encode subprocess (~1.2 GB). A full
    # cull peaks around 2.5 GB across the parent plus whichever subprocess is
    # live, so a machine with less than that will PAGE — measured: a run starting
    # with 2.07 GB free drove free RAM to 0.01 GB, grew the pagefile 1.22 GB and
    # took 394s, against 192s for the same run with a little more headroom.
    # Paging is what the old "it freezes" reports actually were, so say it
    # plainly instead of silently going disk-bound.
    _CULL_PEAK_GB = 2.5
    try:
        import psutil as _ps_cull
        _free_gb = _ps_cull.virtual_memory().available / 1e9
        if _free_gb < _CULL_PEAK_GB:
            _msg = (f"Low memory: {_free_gb:.1f} GB free, a cull needs about "
                    f"{_CULL_PEAK_GB:.1f} GB. This will still finish, but it will "
                    f"be slower while Windows swaps to disk — closing a browser "
                    f"or editor window roughly halves the time.")
            print(f"[v2] {_msg}", flush=True)
            _p(0.005, f"Low memory ({_free_gb:.1f} GB free) — this cull will be slower")
    except Exception:
        pass

    # ── Step 1: Discover images ───────────────────────────────────────────────
    _p(0.01, "Scanning folder…")
    folder = Path(folder_path)
    all_paths  = sorted(
        str(f) for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not all_paths:
        return {"error": "No images found in folder.", "gallery": [], "total": 0}

    # Pre-grade niche detection: cap to an evenly-spaced sample so a fast scan
    # over a representative subset completes in a few seconds regardless of how
    # many photos the folder holds. Even spacing (not the first N) avoids bias
    # toward whatever happens to sort first.
    if sample_limit and sample_limit > 0 and len(all_paths) > sample_limit:
        _step = len(all_paths) / sample_limit
        all_paths = [all_paths[int(i * _step)] for i in range(sample_limit)]
        print(f"[v2] DETECT sample: capped to {len(all_paths)} evenly-spaced images")

    # ── Incremental: skip already-graded images when force_rescan=False ───────
    import lance_store as _ls_diag
    # ORDER-CRITICAL: pin pyarrow/lancedb's native DLLs into this process BEFORE
    # any GPU subprocess is spawned. Their first import after a CUDA child has
    # exited faults with an access violation and kills the grade with no
    # traceback (see lance_store.warm_native). Importing lance_store already
    # does this; the explicit call keeps the requirement visible so a future
    # reorder of these imports cannot silently reintroduce the crash.
    _ls_diag.warm_native()
    print(f"[v2] RUN START  folder={folder_path}  force_rescan={force_rescan}")
    print(f"[v2] LanceDB    path={_ls_diag._DB_DIR}  table={_ls_diag._TBL_NAME}")

    cached_rows: dict[str, dict] = {}
    if not force_rescan:
        try:
            import lance_store as _ls
            fp_str = str(Path(folder_path).resolve())
            for row in _ls.query_all(min_score=0.0):
                rp = row.get("path", "")
                # Normalise separators so Windows backslash paths and frontend
                # forward-slash paths both map to the same key.
                rp_norm = str(Path(rp)) if rp else ""
                if rp_norm.startswith(fp_str) and float(row.get("score", 0)) >= 0.10:
                    cached_rows[rp_norm] = row
        except Exception as _ce:
            print(f"[v2] LanceDB cache check failed: {_ce}")

    paths = [p for p in all_paths if p not in cached_rows]
    n     = len(paths)
    print(f"[v2] Images     total={len(all_paths)}  cached(skipped)={len(cached_rows)}  to_grade={n}")

    def _cached_to_gallery(row: dict) -> dict:
        bd = row.get("breakdown", {})
        if isinstance(bd, str):
            try: bd = json.loads(bd)
            except Exception: bd = {}
        _cs = round(float(row.get("score", 0.5)), 3)
        return {
            "id": row["path"], "path": row["path"],
            "filename": Path(row["path"]).name,
            "grade": row.get("grade", GRADE_MID),
            "score":         _cs,
            "overall_score": _cs,
            "rating":        _cs,
            "human_perception": round(float(row.get("personal_score", 0.5)), 3),
            "personal_score":   round(float(row.get("personal_score", 0.5)), 3),
            "embedding": row.get("embedding", []),
            "breakdown": bd,
            "critique": "",
            "reasoning_log": "",
            "is_verified": False,
            "exif_ts": float(row.get("exif_ts", 0.0)),
            "stars": 0, "reject": False, "sim_flag": "", "cluster_id": -1,
        }

    if cached_rows and not paths:
        # All images already graded — return cached data immediately
        _p(1.0, f"All {len(cached_rows)} photos already graded — use Re-grade to redo them")
        gallery = [_cached_to_gallery(cached_rows[p]) for p in all_paths if p in cached_rows]
        grades  = [g["grade"] for g in gallery]
        return {
            "gallery": gallery,
            "mogco_sequence": [],
            "strong": sum(1 for g in grades if g == GRADE_STRONG),
            "mid":    sum(1 for g in grades if g == GRADE_MID),
            "weak":   sum(1 for g in grades if g == GRADE_WEAK),
            "total":  len(gallery),
            "pipeline": "v2_cached",
        }

    if cached_rows:
        _p(0.02, f"Found {n} new images to grade ({len(cached_rows)} already graded)")
    else:
        _p(0.02, f"Found {n} images")

    # ── Step 1b: Cascaded Early-Exit Gate ────────────────────────────────────
    # Fastest checks first — CPU Laplacian blur, then brief-conditional YOLO gate.
    # Disqualified images get score 0.00 written to LanceDB; all downstream GPU
    # models see only the survivors, eliminating wasted compute.
    from pipeline_stages import run_gate_stage as _run_gate_stage
    _gate = _run_gate_stage(paths, _p)
    _blur_disqualified   = _gate.blur_disqualified
    _yolo_disqualified   = _gate.yolo_disqualified
    _yolo_soft_penalized = _gate.yolo_soft_penalized
    _technical_disq      = _gate.technical_disq

    # ── Pre-flush: commit fail records before GPU stages begin ────────────────
    # Persists disqualified images to LanceDB immediately so that if the GPU
    # pipeline aborts mid-run, score=0.00 records are already in the store and
    # won't re-enter the processing queue on the next run.
    from pipeline_stages import flush_gate_failures as _flush_gate_failures
    _flush_gate_failures(_gate, _ENC_DIM, GRADE_WEAK, _p)

    # ── Step 2: Bulk encoding ─────────────────────────────────────────────────
    # Singleton path (repeat runs): reuse encoder already in VRAM — no reload.
    # Cold path (first run): load, encode, cache static text embeddings, keep
    # encoder in VRAM as _enc_singleton for subsequent runs.
    global _enc_singleton, _text_emb_cache, _qwen_singleton
    embs            = None
    embed_dim       = 1152
    siglip_ok       = False
    _pos_text_embs  = None
    _neg_text_embs  = None
    _aspect_pos     = None
    _aspect_neg     = None
    _aspect_names   = None
    _prompt_emb      = None   # (1536,) L2-normalised brief ensemble embedding for SemanticHead
    _genre_ref_embs         = None   # (3, 1536) low-contrast genre refs for TOPIQ bias correction
    _fine_art_anchor        = None   # (1536,) averaged fine-art pictorialism anchor
    street_aesthetic_scores = None   # (N,) multi-probe street photography aesthetic scores
    _sp_embs                = None   # (P, 1536) street positive probe embeddings
    _sn_embs                = None   # (Q, 1536) street negative probe embeddings
    _enc_reused             = False

    # Evict any warm Qwen singleton left in VRAM by a PRIOR cull BEFORE loading
    # SigLIP-2. SigLIP-2 FP16 (~3.5 GB) + a resident Qwen INT4 (~2.2 GB) overflows
    # the 6 GB card → the worker dies NATIVELY (no traceback) during encode. The
    # old INT8 SigLIP (1.8 GB) coexisted with warm Qwen; FP16 does not. Qwen
    # reloads fast from its INT4 cache when grading starts.
    if _qwen_singleton is not None:
        try:
            _qwen_singleton.unload()
            print("[v2] Evicted warm Qwen singleton before SigLIP-2 load (VRAM headroom)")
        except Exception:
            pass
        _qwen_singleton = None
        _vram_clear()

    if _enc_singleton is not None:
        _p(0.03, "Analyzing images…")
        try:
            with _ProgressTicker(_p, 0.07, 0.46, f"Analyzing {len(paths)} photos…"):
                embs = _enc_singleton.encode_images(paths, progress=_p)
            if _text_emb_cache:
                _pos_text_embs  = _text_emb_cache["pos"]
                _neg_text_embs  = _text_emb_cache["neg"]
                _aspect_names   = _text_emb_cache["aspect_names"]
                _aspect_pos     = _text_emb_cache["aspect_pos"]
                _aspect_neg     = _text_emb_cache["aspect_neg"]
                _genre_ref_embs  = _text_emb_cache.get("genre_ref_embs")
                _fine_art_anchor = _text_emb_cache.get("fine_art_anchor")
                _sp_embs         = _text_emb_cache.get("sp")
                _sn_embs         = _text_emb_cache.get("sn")
            try:
                from specvlm_pipeline import _CD_BRIEF as _brief_text
                if _brief_text and _brief_text.strip():
                    _p(0.49, "Reading your creative brief…")
                    _brief_variants = _generate_brief_variants(_brief_text)
                    _brief_raw  = _enc_singleton.encode_text(_brief_variants)  # (V, 1536)
                    _prompt_emb = _brief_raw.mean(axis=0).astype(np.float64)
                    _prompt_emb /= (np.linalg.norm(_prompt_emb) + 1e-9)
                    _prompt_emb  = _prompt_emb.astype(np.float32)
                    print(f"[v2] Brief ensemble ({len(_brief_variants)} variants): '{_brief_text[:60]}'")
            except Exception as _e_brief:
                print(f"[v2] Brief embedding skipped: {_e_brief}")
            embed_dim   = embs.shape[1] if embs is not None else _ENC_DIM
            siglip_ok   = True
            _enc_reused = True
            print("[v2] Encoder: SigLIP-2 singleton reused — no VRAM reload")
        except Exception as _e_reuse:
            print(f"[v2] Singleton reuse failed ({_e_reuse}) — reloading encoder")
            try:
                _enc_singleton.unload()
            except Exception:
                pass
            _enc_singleton = None
            _text_emb_cache.clear()

    if not _enc_reused:
        import traceback as _tb
        _siglip_last_err: str = ""
        for _attempt, _kwargs in enumerate([
            # GPU FP16 (~3.5 GB, fits the 6 GB card alone). INT8 quantize=True is
            # DISABLED: torchao INT8 weight-only on SigLIP-2 ViT-g emits all-NaN
            # embeddings on this stack (verified 2026-06-14) → flat-0.5 grades.
            # FP16 produces correct unit-norm embeddings.
            {"device": "auto", "quantize": False},  # 1st: GPU FP16 (correct)
            {"device": "cpu",  "quantize": False},  # 2nd: CPU FP16 (slow but correct)
        ]):
            try:
                from siglip2_encoder import SigLIP2Encoder
                from specvlm_pipeline import _POS_PROMPTS, _NEG_PROMPTS, _ASPECT_PROMPTS

                # Pull any embeddings already stored in LanceDB — skip re-encoding
                # images the model has seen before.  Cache invalidation is implicit:
                # if the user re-grades after editing a photo they would clear
                # LanceDB (or the stale embedding just re-grades identically, which
                # is correct since SigLIP-2 is deterministic for the same pixels).
                # Encoder-source migration guard: when the embedding source
                # changes (legacy open_clip ↔ HF transformers), every cached
                # embedding/probe is from the OLD model. Force a full re-encode
                # in the new space — mixing spaces corrupts dedup/archetypes/the
                # personal head. Also clears the probe cache so probes recompute.
                _src_changed = False
                try:
                    from siglip2_encoder import ENCODER_SOURCE as _enc_src
                    # Per-tier, like the probe cache: embeddings live in per-tier
                    # LanceDB tables, so a single global marker made every switch
                    # between Pro and Balanced look like an encoder migration and
                    # re-encode the whole folder. A machine hovering near the tier
                    # threshold would re-encode on every run, in both directions.
                    # _tier_cache_name keeps 'high' unsuffixed, so existing
                    # markers (and caches) stay valid.
                    _src_marker = (Path(__file__).resolve().parent.parent / "cache"
                                   / _tier_cache_name("encoder_source", ".txt"))
                    _prev_src = _src_marker.read_text(encoding="utf-8").strip() if _src_marker.exists() else ""
                    if _prev_src != _enc_src:
                        _src_changed = True
                        print(f"[v2] Encoder source changed ({_prev_src or 'none'} -> {_enc_src}) "
                              f"— re-encoding all images + clearing probe cache")
                        _pc = Path(__file__).resolve().parent.parent / "cache"
                        for _pf in (_tier_cache_name("probe_embs", ".npz"),
                                    _tier_cache_name("probe_embs", ".hash")):
                            try:
                                (_pc / _pf).unlink(missing_ok=True)
                            except Exception:
                                pass
                        # NOTE: the marker is NOT written here. It is written only
                        # after the re-encode actually succeeds (see below) —
                        # writing it on detection meant a failed attempt left the
                        # marker claiming the migration had happened, so the retry
                        # saw "no change", reused the OLD-space cache, and logged
                        # "re-encoding all images" while re-encoding almost none.
                        # Observed live: attempt 1 hit the RAM floor, attempt 2
                        # then re-encoded 3 of 514.
                except Exception as _e_src:
                    print(f"[v2] encoder-source check skipped: {_e_src}")

                _cached_embs: dict = {}
                try:
                    from lance_store import query_embeddings_by_paths as _qe
                    _cached_embs = {} if _src_changed else _qe(paths)
                    # Drop NaN/degenerate cached embeddings (e.g. the all-NaN ones
                    # the old INT8 path wrote) so they are RE-ENCODED rather than
                    # poisoning the grade with flat-0.5 scores. Self-heals the cache.
                    _bad = [p for p, e in _cached_embs.items()
                            if not np.all(np.isfinite(np.asarray(e, dtype=np.float32)))]
                    for _p_bad in _bad:
                        _cached_embs.pop(_p_bad, None)
                    if _bad:
                        print(f"[v2] Dropped {len(_bad)} NaN/Inf cached embeddings — will re-encode")
                    if _cached_embs:
                        print(f"[v2] LanceDB emb cache: {len(_cached_embs)}/{n} hits — skipping re-encode")
                except Exception as _e_lc:
                    print(f"[v2] LanceDB emb lookup skipped: {_e_lc}")

                _paths_to_encode = [p for p in paths if p not in _cached_embs]

                if _paths_to_encode:
                    # Stage timing: the "Analyzing" progress label covers the
                    # encode AND the durable-embedding persist, and a 514-photo
                    # run spent 429s under it while the encode itself measures
                    # ~220 ms/img (=113s). Splitting the label's contents shows
                    # where the rest actually goes instead of assuming.
                    import time as _t_enc
                    _te0 = _t_enc.monotonic()
                    enc  = SigLIP2Encoder(**_kwargs, progress=_p)
                    _te_init = _t_enc.monotonic() - _te0
                    _te0 = _t_enc.monotonic()
                    with _ProgressTicker(_p, 0.07, 0.46, f"Analyzing {len(_paths_to_encode)} photos…"):
                        _new_embs = enc.encode_images(_paths_to_encode, progress=_p)
                    _te_encode = _t_enc.monotonic() - _te0
                    _te0 = _t_enc.monotonic()
                    _new_emb_map = dict(zip(_paths_to_encode, _new_embs))
                    _te_map = _t_enc.monotonic() - _te0
                    # DURABLE PROGRESS: persist freshly-encoded embeddings to
                    # LanceDB immediately (placeholder score/grade, overwritten by
                    # the final upsert). If the long scoring/IQA tail then crashes,
                    # the expensive SigLIP encode is NOT redone — query_embeddings_
                    # by_paths finds these next run. Skip zero-rows (unreadable RAW).
                    try:
                        import lance_store as _ls_emb
                        _emb_records = [
                            {"path": _pp, "embedding": _ee.tolist(),
                             "score": 0.0, "grade": "Pending"}
                            for _pp, _ee in _new_emb_map.items()
                            if float(np.linalg.norm(_ee)) >= 1e-6
                        ]
                        _te0 = _t_enc.monotonic()
                        if _emb_records:
                            _ls_emb.upsert_batch(_emb_records)
                            print(f"[v2] Persisted {len(_emb_records)} fresh embeddings "
                                  f"to LanceDB (resumable)")
                        _te_persist = _t_enc.monotonic() - _te0
                        print(f"[v2] STAGE TIME  encoder-init {_te_init:5.1f}s  "
                              f"encode {_te_encode:6.1f}s  map {_te_map:5.1f}s  "
                              f"lancedb-persist {_te_persist:6.1f}s", flush=True)
                    except Exception as _e_emb_persist:
                        print(f"[v2] Early embedding persist skipped: {_e_emb_persist}")
                else:
                    # All embeddings from cache — no GPU needed for encoding
                    enc = None
                    _new_emb_map = {}

                # Reconstruct full embs array in original path order
                embs = np.stack(
                    [_cached_embs[p] if p in _cached_embs else _new_emb_map[p]
                     for p in paths],
                    axis=0,
                ).astype(np.float32)   # (N, 1536)

                # Encode aesthetic text references and cache for subsequent runs
                # Augment positive prompts with any PDF reference phrases already ingested
                _pos_prompts_augmented = list(_POS_PROMPTS)
                try:
                    # Was a DIRECT file read, which bypassed the opt-in gate in
                    # pdf_rag.load_concepts() entirely -- book-derived phrases
                    # would still have reached the positive rubric.
                    from pdf_rag import load_concepts as _load_rag_pos
                    _rag_phrases = _load_rag_pos()
                    if _rag_phrases:
                        _pos_prompts_augmented = _pos_prompts_augmented + _rag_phrases
                        print(f"[v2] RAG: added {len(_rag_phrases)} PDF concept phrases to positive rubric")
                except Exception as _e_rag:
                    print(f"[v2] RAG load skipped: {_e_rag}")

                # ── One shared rubric: every niche's vocabulary ────────────────
                # niche_registry has tailored probes for 20 niches, and its own
                # docstring claimed grade_pipeline_v2 consumed them. It never
                # did — the function had no callers — so every folder was judged
                # by street-photography language whatever it held. On this
                # machine that is 0.6% of the library: 93 classic_street photos
                # against 5,707 travel and 4,366 fine-art.
                #
                # Deliberately PRESET-INDEPENDENT. Using only the detected
                # genre's probes would derive a scale per genre and normalise
                # each to the same spread — every folder ~30% Strong, whether or
                # not the work is comparable. That is the batch-relative curve
                # one level up. The union asks "excellent by ANY recognised
                # standard" on one shared scale, so a weak photograph of any
                # genre still has nowhere to win.
                #
                # It also fixes a thin negative side: five generic negatives
                # made max(negative) near-constant, leaving the discriminant
                # effectively positive-only. The union brings ~81.
                _neg_prompts_augmented = list(_NEG_PROMPTS)
                try:
                    from niche_registry import union_probes as _union_probes
                    _p_before, _n_before = len(_pos_prompts_augmented), len(_neg_prompts_augmented)
                    _pos_prompts_augmented, _neg_prompts_augmented = _union_probes(
                        _pos_prompts_augmented, _neg_prompts_augmented)
                    print(f"[v2] Shared rubric: {_p_before}->{len(_pos_prompts_augmented)} positive, "
                          f"{_n_before}->{len(_neg_prompts_augmented)} negative probes "
                          f"(all 20 niches; preset '{preset}' labels but does not score)")
                except Exception as _e_niche:
                    print(f"[v2] Niche probe union skipped: {_e_niche}")

                # ── Probe embedding disk cache ────────────────────────────────
                # All static text probes (pos/neg/aspect/genre/fine-art/street)
                # are keyed by an MD5 of the prompt lists + RAG phrases.
                # On cache hit every encode_text call is skipped — saves ~400 ms
                # of GPU time and allows SigLIP-2 to be freed sooner.
                # The CD brief is session-specific and is never cached here.
                import hashlib as _hl
                _probe_cache_dir  = Path(__file__).resolve().parent.parent / "cache"
                _probe_cache_path = _probe_cache_dir / _tier_cache_name("probe_embs", ".npz")
                _probe_hash_path  = _probe_cache_dir / _tier_cache_name("probe_embs", ".hash")
                _probe_key_src = repr((
                    # BOTH augmented lists must be in the cache key: the niche
                    # probes change with `preset`, and keying on the base
                    # negatives would serve a cached encode from a different
                    # niche.
                    _pos_prompts_augmented, _neg_prompts_augmented,
                    list(_ASPECT_PROMPTS.keys()),
                    [v for pair in _ASPECT_PROMPTS.values() for v in pair],
                    _GENRE_REF_PROMPTS, _FINE_ART_PROMPTS,
                    _STREET_POS_PROBES, _STREET_NEG_PROBES,
                ))
                _probe_key_hash = _hl.md5(_probe_key_src.encode()).hexdigest()
                _probe_hit = False
                _sp_embs = _sn_embs = None   # initialise so refs below never raise UnboundLocalError

                if _probe_cache_path.exists() and _probe_hash_path.exists():
                    try:
                        if _probe_hash_path.read_text().strip() == _probe_key_hash:
                            _pcd = dict(np.load(str(_probe_cache_path)))
                            _pos_text_embs   = _pcd["pos"]
                            _neg_text_embs   = _pcd["neg"]
                            _aspect_pos      = _pcd["aspect_pos"]
                            _aspect_neg      = _pcd["aspect_neg"]
                            _genre_ref_embs  = _pcd["genre_ref"]
                            _fine_art_anchor = _pcd["fine_art"]
                            _sp_embs         = _pcd["sp"]
                            _sn_embs         = _pcd["sn"]
                            _ppl_mean_cached = _pcd.get("ppl")
                            _aspect_names    = list(_ASPECT_PROMPTS.keys())
                            _probe_hit       = True
                            if _ppl_mean_cached is not None:
                                try:
                                    np.save(str(_probe_cache_dir / _tier_cache_name("people_emb", ".npy")), _ppl_mean_cached)
                                except Exception:
                                    pass
                            _text_emb_cache.update({
                                "pos":             _pos_text_embs,
                                "neg":             _neg_text_embs,
                                "aspect_names":    _aspect_names,
                                "aspect_pos":      _aspect_pos,
                                "aspect_neg":      _aspect_neg,
                                "genre_ref_embs":  _genre_ref_embs,
                                "fine_art_anchor": _fine_art_anchor,
                                "sp":              _sp_embs,
                                "sn":              _sn_embs,
                            })
                            print(f"[v2] Probe embeddings: loaded from disk cache "
                                  f"({len(_pos_text_embs)}pos / {len(_neg_text_embs)}neg / "
                                  f"{len(_STREET_POS_PROBES)}+{len(_STREET_NEG_PROBES)} street)")
                    except Exception as _e_pcd:
                        print(f"[v2] Probe cache load failed: {_e_pcd}")

                # When all images came from LanceDB cache, enc=None but text
                # embeddings are still required for SpecVLM / score fusion.
                # Load a minimal encoder just for text encoding.
                if not _probe_hit and enc is None:
                    print("[v2] Probe cache miss + all images cached — loading encoder for text embeddings only")
                    for _te_kw in [{"device": "auto", "quantize": True}, {"device": "cpu", "quantize": False}]:
                        try:
                            enc = SigLIP2Encoder(**_te_kw, progress=_p)
                            print(f"[v2] Text-only encoder loaded ({_te_kw['device']})")
                            break
                        except Exception as _e_te:
                            print(f"[v2] Text-only encoder load failed ({_te_kw['device']}): {_e_te}")

                if not _probe_hit and enc is not None:
                    _p(0.48, "Preparing the style reference…")
                    _aspect_names  = list(_ASPECT_PROMPTS.keys())
                    _PEOPLE_PROMPTS = [
                        "people", "crowds", "pedestrians", "human figure", "faces",
                    ]
                    # ONE subprocess call / ONE model load for all 9 probe groups.
                    # Previously each enc.encode_text() call below spawned its own
                    # encode_worker.py subprocess, which reloads the whole SigLIP-2
                    # model from scratch — 9 sequential calls meant 9 full reloads
                    # back to back (each an 8 GB RAM spike with the open_clip
                    # fallback loader), which is the actual cause of the "stalls at
                    # 48%, RAM 94-99%" symptom. encode_text_groups() flattens every
                    # group into one list, encodes once, and splits the result.
                    _probe_groups = enc.encode_text_groups({
                        "pos":        _pos_prompts_augmented,                  # (P+R, 1536)
                        "neg":        _neg_prompts_augmented,                  # (Q+niche, 1536)
                        "aspect_pos": [v[0] for v in _ASPECT_PROMPTS.values()],  # (A, 1536)
                        "aspect_neg": [v[1] for v in _ASPECT_PROMPTS.values()],  # (A, 1536)
                        "people":     _PEOPLE_PROMPTS,
                        "genre":      _GENRE_REF_PROMPTS,
                        "fine_art":   _FINE_ART_PROMPTS,
                        "street_pos": _STREET_POS_PROBES,
                        "street_neg": _STREET_NEG_PROBES,
                    })
                    _pos_text_embs = _probe_groups["pos"]
                    _neg_text_embs = _probe_groups["neg"]
                    _aspect_pos    = _probe_groups["aspect_pos"]
                    _aspect_neg    = _probe_groups["aspect_neg"]

                    _text_emb_cache.update({
                        "pos":          _pos_text_embs,
                        "neg":          _neg_text_embs,
                        "aspect_names": _aspect_names,
                        "aspect_pos":   _aspect_pos,
                        "aspect_neg":   _aspect_neg,
                    })

                    _ppl_raw  = _probe_groups["people"]
                    _ppl_mean = _ppl_raw.mean(axis=0)
                    _ppl_mean /= (np.linalg.norm(_ppl_mean) + 1e-9)
                    _ppl_mean = _ppl_mean.astype(np.float32)
                    try:
                        _probe_cache_dir.mkdir(parents=True, exist_ok=True)
                        np.save(str(_probe_cache_dir / _tier_cache_name("people_emb", ".npy")), _ppl_mean)
                        print("[v2] people_emb.npy saved for empty-brief CD gate")
                    except Exception as _e_ppl:
                        print(f"[v2] people_emb save skipped: {_e_ppl}")

                    try:
                        _genre_raw      = _probe_groups["genre"]
                        _gnorms         = np.linalg.norm(_genre_raw, axis=1, keepdims=True)
                        _genre_ref_embs = (_genre_raw / (_gnorms + 1e-9)).astype(np.float32)
                        _text_emb_cache["genre_ref_embs"] = _genre_ref_embs
                        print("[v2] Genre reference embeddings cached for TOPIQ bias correction")
                    except Exception as _e_genre_enc:
                        print(f"[v2] Genre ref encoding skipped: {_e_genre_enc}")

                    try:
                        _fa_raw          = _probe_groups["fine_art"]
                        _fa_mean         = _fa_raw.mean(axis=0).astype(np.float64)
                        _fa_mean        /= (np.linalg.norm(_fa_mean) + 1e-9)
                        _fine_art_anchor = _fa_mean.astype(np.float32)
                        _text_emb_cache["fine_art_anchor"] = _fine_art_anchor
                        print("[v2] Fine-art anchor encoded and cached")
                    except Exception as _e_fa:
                        print(f"[v2] Fine-art anchor encoding skipped: {_e_fa}")

                    try:
                        _sp_embs = _probe_groups["street_pos"]
                        _sn_embs = _probe_groups["street_neg"]
                        _text_emb_cache["sp"] = _sp_embs
                        _text_emb_cache["sn"] = _sn_embs
                    except Exception as _e_sp_enc:
                        print(f"[v2] Street probe encoding skipped: {_e_sp_enc}")
                        _sp_embs = _sn_embs = None

                    # Persist all static probes to disk for future runs
                    try:
                        _probe_cache_dir.mkdir(parents=True, exist_ok=True)
                        _save_kw: dict = dict(
                            pos=_pos_text_embs, neg=_neg_text_embs,
                            aspect_pos=_aspect_pos, aspect_neg=_aspect_neg,
                            ppl=_ppl_mean,
                        )
                        if _genre_ref_embs  is not None: _save_kw["genre_ref"]  = _genre_ref_embs
                        if _fine_art_anchor is not None: _save_kw["fine_art"]   = _fine_art_anchor
                        if _sp_embs         is not None: _save_kw["sp"]         = _sp_embs
                        if _sn_embs         is not None: _save_kw["sn"]         = _sn_embs
                        np.savez(str(_probe_cache_path), **_save_kw)
                        _probe_hash_path.write_text(_probe_key_hash)
                        print(f"[v2] Probe embeddings saved to disk cache")
                    except Exception as _e_pcs:
                        print(f"[v2] Probe cache save skipped: {_e_pcs}")

                # CD brief — session-specific, never disk-cached
                if enc is not None:
                    try:
                        from specvlm_pipeline import _CD_BRIEF as _brief_text
                        if _brief_text and _brief_text.strip():
                            _p(0.49, "Reading your creative brief…")
                            _brief_variants = _generate_brief_variants(_brief_text)
                            _brief_raw  = enc.encode_text(_brief_variants)
                            _prompt_emb = _brief_raw.mean(axis=0).astype(np.float64)
                            _prompt_emb /= (np.linalg.norm(_prompt_emb) + 1e-9)
                            _prompt_emb  = _prompt_emb.astype(np.float32)
                            print(f"[v2] Brief ensemble ({len(_brief_variants)} variants): '{_brief_text[:60]}'")
                    except Exception as _e_brief:
                        print(f"[v2] Brief embedding skipped: {_e_brief}")

                if enc is not None:
                    _enc_singleton = enc   # keep in VRAM — evicted by release_grading_models()

                # Commit the encoder-source marker ONLY now that the encode has
                # actually succeeded. Any earlier and a failed attempt would
                # convince the retry (and every future run) that the migration
                # was already done.
                if _src_changed:
                    try:
                        _src_marker.parent.mkdir(parents=True, exist_ok=True)
                        _src_marker.write_text(_enc_src, encoding="utf-8")
                        print(f"[v2] Encoder source marker committed -> {_enc_src}")
                    except Exception as _e_mark:
                        print(f"[v2] Encoder source marker write failed: {_e_mark}")

                embed_dim = embs.shape[1] if embs is not None else _ENC_DIM
                siglip_ok = True
                _tag = "GPU" if _kwargs["device"] == "auto" else "CPU fallback"
                _p(0.50, "Image analysis complete…")
                print(f"[v2] Encoder: SigLIP-2 NaFlex ({_tag})  dim={embed_dim}")
                break
            except Exception as e_siglip2:
                _siglip_last_err = str(e_siglip2)
                print(f"[v2] SigLIP-2 attempt {_attempt+1} failed: {e_siglip2}")
                if _attempt == 0:
                    print("[v2] Retrying SigLIP-2 on CPU…")
                else:
                    print("[v2] SigLIP-2 unavailable after all attempts.")
                    print(_tb.format_exc())

        # SigLIP-2 of the active tier is required (dim = _ENC_DIM: high 1536 /
        # mid 1024 / low 768). A mismatch means the encoder failed to load.
        if embed_dim != _ENC_DIM:
            raise RuntimeError(
                f"SigLIP-2 failed to load on both GPU and CPU.\n"
                f"Reason: {_siglip_last_err}"
            )

    # ── Street probe scoring ──────────────────────────────────────────────────
    # Runs on both fresh-load and singleton-reuse paths.  sp/sn are now cached in
    # _text_emb_cache so they survive across calls within the same server session.
    if street_aesthetic_scores is None:
        try:
            if _sp_embs is not None and _sn_embs is not None:
                _street_raw = (embs @ _sp_embs.T).mean(axis=1) - (embs @ _sn_embs.T).mean(axis=1)
                _s_min, _s_max = float(_street_raw.min()), float(_street_raw.max())
                if _s_max > _s_min:
                    street_aesthetic_scores = ((_street_raw - _s_min) / (_s_max - _s_min)).astype(np.float32)
                else:
                    street_aesthetic_scores = np.full(n, 0.5, dtype=np.float32)
                print(f"[v2] Street aesthetic scores: min={street_aesthetic_scores.min():.3f}  "
                      f"max={street_aesthetic_scores.max():.3f}  mean={street_aesthetic_scores.mean():.3f}  "
                      f"({len(_STREET_POS_PROBES)}pos/{len(_STREET_NEG_PROBES)}neg probes)")
            else:
                street_aesthetic_scores = np.full(n, 0.5, dtype=np.float32)
        except Exception as _e_sp:
            print(f"[v2] Street probe scoring skipped: {_e_sp}")
            street_aesthetic_scores = np.full(n, 0.5, dtype=np.float32)

    # ── Archetype text projections ────────────────────────────────────────────
    # Five frozen concept reference vectors for soft routing in Step 4d.
    # Cached after first run — never requires the encoder to reload for this.
    # Index map: 0=geo  1=night  2=layered  3=messy  4=maximalist_documentary
    _ARCHETYPE_PROMPTS = [
        "minimalist architectural and interior geometry, graphic lines, vanishing points, empty liminal commercial space, Edward Hopper quiet interior light, stark geometric shadow, empty diner or laundromat",
        "cinematic low-key street photography, dark atmospheric shadows, intense chiaroscuro light pools",
        "layered environmental street portrait, crisp subject focus framed by intentional out-of-focus foreground elements",
        "unintentional messy amateur snapshot, accidental random camera angles, domestic clutter, junk, trash, throwaway frame, zero artistic value",
        "highly detailed maximalist environmental documentary photography, authentic traditional shop interior or street life scene, dense cultural artifacts, rich storytelling composition, intentional maximalism",
    ]
    _arch_cache_path      = Path("cache") / _tier_cache_name("archetype_embs", ".npy")
    _arch_hash_path       = Path("cache") / _tier_cache_name("archetype_embs", ".hash")
    _arch_prompt_hash     = hashlib.md5(json.dumps(_ARCHETYPE_PROMPTS).encode()).hexdigest()
    archetype_embs: Optional[np.ndarray] = None

    # Hash-based invalidation: delete cached embeddings if prompts changed.
    if _arch_cache_path.exists():
        _cached_hash = _arch_hash_path.read_text().strip() if _arch_hash_path.exists() else ""
        if _cached_hash != _arch_prompt_hash:
            _arch_cache_path.unlink()
            print("[v2] Archetype prompts changed — invalidating embedding cache")

    if _arch_cache_path.exists():
        archetype_embs = np.load(str(_arch_cache_path)).astype(np.float32)
        # Shape guard: a cached file whose width does not match the CURRENT
        # encoder would blow up the Step 4d projection (rated_unit @ arch_unit.T)
        # with a dimension mismatch. The prompt hash cannot catch this — the
        # prompts are unchanged, the encoder is not. Belt-and-braces on top of
        # the per-tier filename, so a file from any other source is ignored.
        _want_dim = int(embs.shape[1]) if embs is not None else _ENC_DIM
        if archetype_embs.ndim != 2 or archetype_embs.shape[1] != _want_dim:
            print(f"[v2] Archetype cache is {archetype_embs.shape} but this encoder "
                  f"emits {_want_dim}-d — discarding and recomputing")
            archetype_embs = None
            try:
                _arch_cache_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            print(f"[v2] Archetype embeddings: loaded from cache {archetype_embs.shape}")

    if archetype_embs is not None:
        pass                                   # usable cache hit, verified above
    elif _enc_singleton is not None:
        # Cache miss — compute from the live encoder (guaranteed loaded at this stage).
        _arch_raw      = _enc_singleton.encode_text(_ARCHETYPE_PROMPTS)
        _arch_nrm      = np.linalg.norm(_arch_raw, axis=1, keepdims=True)
        archetype_embs = (_arch_raw / (_arch_nrm + 1e-9)).astype(np.float32)
        _arch_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(_arch_cache_path), archetype_embs)
        _arch_hash_path.write_text(_arch_prompt_hash)
        print(f"[v2] Archetype embeddings: computed and cached {archetype_embs.shape}")
    else:
        raise RuntimeError(
            f"[v2] Archetype embeddings cache missing and _enc_singleton is None. "
            f"The encoder must be loaded before this stage. "
            f"Delete cache/{_tier_cache_name('archetype_embs', '.npy')} and re-grade to regenerate."
        )

    # Flush caching allocator — singleton weights remain resident in VRAM
    _vram_clear()

    # ── Step 3: Duplicate detection ───────────────────────────────────────────
    # ── Drop unreadable RAW files ENTIRELY ───────────────────────────────────
    # encode_worker emits a zero-vector for any RAW whose embedded preview could
    # not be read. Remove those rows from paths/embs here (before dedup, scoring,
    # and the gallery) so they vanish from the pipeline instead of being judged as
    # a 0.00 "Weak" alongside genuinely bad photos.
    from pipeline_stages import drop_unreadable_rows as _drop_unreadable_rows
    paths, embs, n = _drop_unreadable_rows(paths, embs, np)

    if n == 0:
        # Every file that survived the cache check turned out to be unreadable, so the
        # drop above emptied paths/embs. Their 0.00 fail records were ALREADY flushed to
        # LanceDB by the early-exit gate, so this folder's work is genuinely finished.
        #
        # Falling through instead is what killed a 785-folder cull three times: every
        # stage below assumes at least one row, and `scores` becomes zero-length, so
        # `scores_arr.min()` — inside a diagnostic print — raises "zero-size array to
        # reduction operation minimum which has no identity". Guarding the prints one
        # by one just moves the crash; the degenerate walk itself is the bug. A folder
        # of unreadable files is a normal outcome (AppleDouble `._*` sidecars alone
        # produce it), not an error.
        print("[v2] All remaining files were unreadable — nothing to grade in this folder")
        _p(1.0, "No usable images in this folder")
        gallery = [_cached_to_gallery(cached_rows[p]) for p in all_paths if p in cached_rows]
        grades  = [g["grade"] for g in gallery]
        return {
            "gallery": gallery,
            "mogco_sequence": [],
            "strong": sum(1 for g in grades if g == GRADE_STRONG),
            "mid":    sum(1 for g in grades if g == GRADE_MID),
            "weak":   sum(1 for g in grades if g == GRADE_WEAK),
            "total":  len(gallery),
            "pipeline": "v2_cached",
        }

    _p(0.50, "Finding near-duplicate shots…")
    cluster_ids:     list[int] = [-1] * n
    sim_flags:       list[str] = [""] * n
    to_rate_indices: list[int] = list(range(n))
    _comp_eligible:  set[str]  = set(paths)   # default: all paths eligible for composition

    if siglip_ok and n >= 2:
        try:
            from collections import defaultdict as _dd
            norms  = np.linalg.norm(embs, axis=1, keepdims=True)
            normed = (embs / (norms + 1e-9)).astype(np.float32)

            SIM_THRESH = 0.96   # true burst duplicates only (same frame ±ms)

            parent = list(range(n))
            def _find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            # Row-blocked instead of one full n×n allocation — keeps peak
            # memory O(chunk×n) so very large imports can't spike RAM here.
            _chunk = _dedup_chunk_size(n)
            for r0 in range(0, n, _chunk):
                r1 = min(r0 + _chunk, n)
                block_sims = normed[r0:r1] @ normed.T                 # (r1-r0, n)
                # Keep only the strict upper triangle (j > i). Doing it by
                # writing -1.0 over the j<=i entries IN PLACE is equivalent to
                # the old `(sims > THRESH) & (col_idx > row_idx)` — -1.0 can
                # never exceed a positive threshold — but avoids materialising
                # three extra (chunk, n) arrays (the broadcast index compare, the
                # threshold compare, and their AND) on top of block_sims itself.
                for _li in range(r1 - r0):
                    block_sims[_li, : r0 + _li + 1] = -1.0
                dup_i, dup_j = np.where(block_sims > SIM_THRESH)
                del block_sims
                for li, j in zip(dup_i.tolist(), dup_j.tolist()):
                    i = r0 + li
                    ri, rj = _find(i), _find(j)
                    if ri != rj:
                        parent[ri] = rj

            groups_d: dict = _dd(list)
            for i in range(n):
                groups_d[_find(i)].append(i)

            # Populate cluster_ids for all photos in duplicate groups (size >= 2)
            _comp_eligible: set[str] = set()
            for root, members in groups_d.items():
                if len(members) >= 2:
                    for i in members:
                        cluster_ids[i] = root
                    _comp_eligible.add(paths[root])   # only representative gets composition
                else:
                    _comp_eligible.add(paths[members[0]])   # singleton: always eligible

            n_clustered = sum(1 for c in cluster_ids if c >= 0)
            n_reps      = sum(1 for root, members in groups_d.items() if len(members) >= 2)
            if n_clustered:
                print(f"[v2] Duplicate detection: {n_clustered} images in clusters"
                      f" ({n_reps} representatives) — burst dedup saves depth/seg/chiaroscuro"
                      f" for {n_clustered - n_reps} cluster members")

        except Exception as e:
            import traceback as _tb_dedup
            print(f"[v2] Duplicate detection failed: {e}")
            _tb_dedup.print_exc()
            to_rate_indices = list(range(n))

    # Step 3b: YOLO gate handled by early-exit gate (Step 1b) before SigLIP-2.
    # _yolo_disqualified and _yolo_soft_penalized are already populated above.

    # ── Step 4: Vision Regression Stack ──────────────────────────────────────
    scores                = np.full(n, 0.5, dtype=np.float32)
    per_photo_breakdowns: list[dict] = [{} for _ in range(n)]

    # Stamp all early-exit disqualified photos (score 0.00, skip IQA).
    # Technical failures (flat / unreadable) are included so SigLIP scoring, TOPIQ,
    # and Qwen never see a void frame and hallucinate a score for it.
    _all_disqualified = _blur_disqualified | _yolo_disqualified | set(_technical_disq)
    for i, p in enumerate(paths):
        if p in _all_disqualified:
            scores[i] = 0.00

    # Exclude disqualified images from IQA / VLM scoring
    to_rate_indices = [i for i in to_rate_indices if paths[i] not in _all_disqualified]
    paths_to_rate = [paths[i] for i in to_rate_indices]

    # Index array for every `embs[...]` / `_sa[...]` gather below. Built ONCE with an
    # explicit integer dtype because `np.array([])` defaults to float64 and NumPy
    # rejects float arrays as indices ("arrays used as indices must be of integer
    # type"). That is not hypothetical: a folder whose images are all cached or all
    # disqualified empties this list, and the bare `np.array(to_rate_indices)` form
    # then killed the whole multi-folder cull. An empty intp array gathers to an
    # empty slice, and the vectorized scoring below is shape-safe at M=0.
    _rate_idx = np.asarray(to_rate_indices, dtype=np.intp)

    # VRAM telemetry via nvidia-smi — NEVER torch.cuda here. get_device_properties /
    # memory_reserved would INITIALIZE a CUDA context in THIS grade-worker process,
    # and holding a CUDA context while the isolated GPU subprocess (encode / iqa)
    # runs faults the worker with 0xC0000005 — the original crash, hidden in a
    # diagnostic print. The grade worker must stay completely CUDA-free.
    try:
        import subprocess as _sp, shutil as _sh
        _smi = _sh.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
        _o = _sp.run([_smi, "--query-gpu=memory.total,memory.free",
                      "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=4,
                     creationflags=0x08000000 if os.name == "nt" else 0)
        if _o.returncode == 0 and _o.stdout.strip():
            _tot, _free = [float(x) for x in _o.stdout.strip().splitlines()[0].split(",")]
            print(f"[v2] VRAM before IQA heads: {_free/1024:.2f} GB free / {_tot/1024:.2f} GB total")
    except Exception:
        pass

    # ── Step 4a: Vision grading (SigLIP zero-shot DEFAULT · Qwen opt-in "Deep Grade") ──
    # DEFAULT (deep_grade=False): SpecVLMPipeline scores each image by SigLIP-2
    #           zero-shot cosine similarity against positive/negative/aspect text
    #           prompts (instant, pure numpy on the embeddings already in RAM — NO
    #           extra GPU model). This is the fast, GPU-light path and — crucially —
    #           it never loads Qwen, so it has none of Qwen's VRAM footprint or the
    #           WebView2 GPU-contention crash surface (0xC0000005 at Qwen load).
    # Deep Grade (deep_grade=True): Qwen2.5-VL looks at each image and outputs
    #           absolute aspect scores (0-100) directly from vision. More nuanced,
    #           slower, GPU-heavy — protected by the WebView2 software-render +
    #           decoupled-server layers. RAG concept phrases are injected here.
    # scan_mode: always SpecVLM CLIP and also skips IQA (ultra-fast niche pass).
    # TOPIQ IQA runs for BOTH default and Deep Grade (only scan_mode skips it).
    _p(0.51, "Judging each photo…")
    print(f"[v2] Grading engine: "
          f"{'Qwen VLM (Deep Grade)' if (deep_grade and not scan_mode) else 'SigLIP zero-shot (default)'}"
          f"{' + scan (no IQA)' if scan_mode else ''}")
    vlm_scores_rated  = np.full(len(paths_to_rate), 0.5, dtype=np.float32)
    comp_scores_rated = np.full(len(paths_to_rate), 0.5, dtype=np.float32)
    _raw_comp_by_path: dict[str, float] = {}
    _vlm_ran = False   # True when Qwen scored — skips SpecVLM CLIP fallback

    if deep_grade and not scan_mode:
        try:
            from qwen_vlm_grader import QwenVLMGrader
            if True:
                try:
                    from pdf_rag import load_concepts as _load_rag
                    _rag_phrases = _load_rag()
                except Exception:
                    _rag_phrases = []

                # Build per-image archetype hints so Qwen evaluates each shot on its own
                # visual register (e.g. don't penalise intentional grain in night shots).
                # Uses embeddings + archetype_embs already in RAM — no extra model needed.
                _ARCH_LABEL_MAP = [
                    "geometric_minimal", "night_chiaroscuro", "layered_portrait",
                    "raw_snapshot", "maximalist_documentary",
                ]
                _arch_hints = None
                if archetype_embs is not None and len(paths_to_rate) > 0:
                    _prate_embs  = embs[_rate_idx].astype(np.float32)
                    _prate_norms = np.linalg.norm(_prate_embs, axis=1, keepdims=True)
                    _prate_unit  = _prate_embs / np.where(_prate_norms > 1e-9, _prate_norms, 1.0)
                    _arch_norms  = np.linalg.norm(archetype_embs, axis=1, keepdims=True)
                    _arch_u      = archetype_embs / np.where(_arch_norms > 1e-9, _arch_norms, 1.0)
                    _arch_dom    = np.argmax(_prate_unit @ _arch_u.T, axis=1)
                    _arch_hints  = {p: _ARCH_LABEL_MAP[int(_arch_dom[j])] for j, p in enumerate(paths_to_rate)}
                    print(f"[v2] Arch hints: {np.bincount(_arch_dom, minlength=5).tolist()} (geo/night/layered/raw/max)")

                # ── CLIP pre-cull funnel ──────────────────────────────────────
                # SpecVLM CLIP scoring is near-free (cosine math on embeddings
                # already in RAM). Photos it scores decisively low skip the
                # expensive VLM entirely and keep their CLIP grade — the VLM
                # spends its seconds ranking contenders, not confirming trash.
                # Threshold is conservative: anything ≥ 0.30 still gets the
                # full VLM look. Disable with FRAMEGRADE_FUNNEL=0.
                _funnel_results: dict = {}
                _vlm_paths = list(paths_to_rate)
                import os as _os_fn
                if _os_fn.environ.get("FRAMEGRADE_FUNNEL", "1") != "0" and len(paths_to_rate) >= 8:
                    try:
                        from specvlm_pipeline import SpecVLMPipeline as _SpecPre
                        _pre = _SpecPre()
                        _pre_res = _pre.grade_images(
                            paths_to_rate,
                            progress        = (lambda f, d: None),
                            scan_mode       = True,
                            preset          = preset,
                            embeddings      = embs[to_rate_indices],
                            pos_text_embs   = _pos_text_embs,
                            neg_text_embs   = _neg_text_embs,
                            aspect_pos_embs = _aspect_pos,
                            aspect_neg_embs = _aspect_neg,
                            aspect_names    = _aspect_names,
                        )
                        _pre.unload()
                        del _pre
                        # Aggressive percentile funnel: skip the VLM for the
                        # bottom FRAC of this batch by the cheap CLIP score
                        # (percentile, not an absolute threshold — robust to the
                        # batch-normalised street scores). A ceiling guard means
                        # a photo the CLIP pass itself rates >= CEIL is never
                        # pre-culled, so an all-strong batch keeps its best.
                        # Culled photos keep their CLIP grade (_grader=
                        # "clip-funnel") — nothing is deleted; toggle off with
                        # FRAMEGRADE_FUNNEL=0 or dial FRAMEGRADE_FUNNEL_FRAC=0.
                        _FUNNEL_FRAC = float(_os_fn.environ.get("FRAMEGRADE_FUNNEL_FRAC", "0.35"))
                        _FUNNEL_CEIL = float(_os_fn.environ.get("FRAMEGRADE_FUNNEL_CEIL", "0.50"))
                        _scored_pre  = sorted(_pre_res, key=lambda r: float(r.score))
                        _n_cull      = int(len(_scored_pre) * _FUNNEL_FRAC)
                        for _fr in _scored_pre[:_n_cull]:
                            if float(_fr.score) < _FUNNEL_CEIL:
                                _funnel_results[_fr.path] = _fr
                        if _funnel_results:
                            _vlm_paths = [p for p in paths_to_rate if p not in _funnel_results]
                            print(f"[v2] CLIP pre-cull funnel: {len(_funnel_results)}/"
                                  f"{len(paths_to_rate)} skip the VLM "
                                  f"(bottom {_FUNNEL_FRAC:.0%} by CLIP, ceil {_FUNNEL_CEIL}) "
                                  f"-> {len(_vlm_paths)} go to Qwen")
                    except Exception as _e_funnel:
                        print(f"[v2] CLIP funnel skipped: {_e_funnel}")

                # Evict SigLIP-2 before grading regardless of Qwen warmth —
                # embeddings are already in NumPy, and SigLIP + Qwen resident
                # together would halve the grading batch size.
                if _enc_singleton is not None:
                    try:
                        _enc_singleton.unload()
                    except Exception:
                        pass
                    _enc_singleton = None
                _vram_clear()

                if _qwen_singleton is None:
                    _p(0.51, "Preparing deep analysis (first run takes longer)…")
                    _qwen_singleton = QwenVLMGrader(progress=_p)
                    print("[v2] Qwen singleton created — will reuse across runs")
                else:
                    _p(0.51, "Deep analysis ready…")
                    print("[v2] Reusing Qwen singleton (warm — no reload)")

                # ── Resume checkpoint ─────────────────────────────────────
                # Cache Qwen grades per (folder, preset) so a cull interrupted
                # mid-grade (window closed) resumes the expensive VLM step
                # instead of re-grading from zero. Only the Qwen step is
                # checkpointed — IQA/sequencing still run for every photo each
                # run, so resume never leaves a photo missing downstream data.
                import json as _json_ck, hashlib as _hl_ck
                from qwen_vlm_grader import VLMScoredResult as _VSR_ck
                _ck_dir  = Path(__file__).resolve().parent.parent / "cache" / "grade_ckpt"
                _ck_dir.mkdir(parents=True, exist_ok=True)
                _ck_key  = _hl_ck.sha1(f"{folder_path}|{preset}".encode()).hexdigest()[:16]
                _ck_path = _ck_dir / f"{_ck_key}.json"
                _ck_cache: dict = {}
                if _ck_path.exists():
                    try:
                        _ck_cache = _json_ck.loads(_ck_path.read_text(encoding="utf-8"))
                    except Exception:
                        _ck_cache = {}
                _ck_hits  = [p for p in _vlm_paths if p in _ck_cache]
                _vlm_todo = [p for p in _vlm_paths if p not in _ck_cache]
                if _ck_hits:
                    print(f"[v2] Resume: {len(_ck_hits)} Qwen grades from checkpoint, "
                          f"{len(_vlm_todo)} still to grade")

                # Checkpoint flush: throttled + atomic.
                #   - Throttled: the old code re-serialised the ENTIRE growing
                #     cache on every batch callback (batch size 3), so a
                #     500-photo deep grade rewrote it ~166 times and built a
                #     progressively larger JSON string each time — O(n²) work
                #     and a repeated large transient allocation, both peaking
                #     exactly when the VLM is already the heaviest thing running.
                #   - Atomic: it wrote straight over the live file, so a crash
                #     mid-write left truncated JSON, which the loader above
                #     silently reduces to {} — losing every checkpointed grade,
                #     i.e. the exact scenario the checkpoint exists for.
                import time as _t_ck
                _CK_FLUSH_S  = 10.0
                _ck_last     = [0.0]

                def _ck_flush(force: bool = False) -> None:
                    _now = _t_ck.monotonic()
                    if not force and (_now - _ck_last[0]) < _CK_FLUSH_S:
                        return
                    _ck_last[0] = _now
                    try:
                        _tmp = _ck_path.with_suffix(".json.tmp")
                        _tmp.write_text(_json_ck.dumps(_ck_cache), encoding="utf-8")
                        _tmp.replace(_ck_path)
                    except Exception:
                        pass

                def _ck_on_batch(_batch):
                    for _r in _batch:
                        _ck_cache[_r.path] = {"score": float(_r.score),
                                              "breakdown": _r.breakdown,
                                              "critique": _r.critique}
                    _ck_flush()

                _fresh = _qwen_singleton.grade_images_scored(
                    _vlm_todo,
                    mode        = preset,
                    rag_phrases = _rag_phrases,
                    arch_hints  = _arch_hints,
                    progress    = _p,
                    on_batch    = _ck_on_batch,
                ) if _vlm_todo else []
                _ck_flush(force=True)   # throttle must not lose the last batches
                _qwen_results = list(_fresh)
                for _p_hit in _ck_hits:
                    _c = _ck_cache[_p_hit]
                    _qwen_results.append(_VSR_ck(
                        path=_p_hit, score=float(_c["score"]),
                        breakdown=_c.get("breakdown") or {},
                        critique=_c.get("critique", ""),
                    ))
                # Keep the Qwen singleton warm: grade_worker_loop is a
                # persistent process, so the model survives across culls —
                # repeat grades skip the ~25 s INT4 load entirely. The VRAM
                # squeeze that previously forced an unload here (UniQA tight,
                # Step 4e's Ollama silently CPU-bound at ~23 min/image) is
                # resolved: 4e is opt-in now, and Qwen 2.2 GB + UniQA/YOLO
                # ~0.7 GB fit together. When 4e IS enabled, the 4e block
                # evicts Qwen before its Ollama warmup.
                _vram_clear()

                _qmap = {r.path: r for r in _qwen_results}
                for local_i, idx in enumerate(to_rate_indices):
                    r = _qmap.get(paths[idx])
                    if r:
                        _raw_comp = float(r.breakdown.get("Composition", r.score))
                        vlm_scores_rated[local_i]     = float(r.score)
                        comp_scores_rated[local_i]    = _raw_comp
                        _raw_comp_by_path[paths[idx]] = _raw_comp
                        per_photo_breakdowns[idx]           = dict(r.breakdown)
                        per_photo_breakdowns[idx]["_grader"] = "qwen"
                        scores[idx]                         = float(r.score)
                    elif paths[idx] in _funnel_results:
                        _fr = _funnel_results[paths[idx]]
                        _raw_comp = float((_fr.breakdown or {}).get("Composition", _fr.score))
                        vlm_scores_rated[local_i]     = float(_fr.score)
                        comp_scores_rated[local_i]    = _raw_comp
                        _raw_comp_by_path[paths[idx]] = _raw_comp
                        per_photo_breakdowns[idx]           = dict(_fr.breakdown or {})
                        per_photo_breakdowns[idx]["_grader"] = "clip-funnel"
                        scores[idx]                         = float(_fr.score)

                _vlm_ran = True
                _p(0.65, "Scoring image quality…")
                if vlm_scores_rated.size:
                    print(
                        f"[v2] Qwen2.5-VL scores: min={vlm_scores_rated.min():.3f}  "
                        f"max={vlm_scores_rated.max():.3f}  mean={vlm_scores_rated.mean():.3f}"
                        + (f"  rag={len(_rag_phrases)} phrases" if _rag_phrases else "")
                    )
                else:
                    print("[v2] Qwen2.5-VL scores: nothing rateable in this folder")
            else:
                print("[v2] Qwen2.5-VL weights not cached — using SpecVLM CLIP scoring")
        except Exception as _e_qwen:
            print(f"[v2] Qwen2.5-VL grading failed ({_e_qwen}) — using SpecVLM CLIP")

    # `paths_to_rate` can be empty even when the folder holds images: everything
    # may already be cached, and whatever is new can be removed by the
    # disqualification filter above (blur / YOLO / technical gate). Running a
    # scoring pipeline over zero images then reducing the empty score array
    # raised "zero-size array to reduction operation minimum", and because the
    # handler re-raises it killed an entire multi-folder cull mid-run. A folder
    # with nothing rateable is a normal outcome, not an error.
    if not _vlm_ran and not paths_to_rate:
        print("[v2] Nothing left to rate in this folder "
              f"({len(_all_disqualified)} disqualified) — skipping VLM scoring")
        _vlm_ran = True

    if not _vlm_ran:
        # ── SpecVLM CLIP scoring (scan mode or Qwen2.5-VL unavailable) ───────
        try:
            from specvlm_pipeline import SpecVLMPipeline
            pipeline        = SpecVLMPipeline()
            specvlm_results = pipeline.grade_images(
                paths_to_rate,
                progress        = _p,
                scan_mode       = scan_mode,
                preset          = preset,
                embeddings      = embs[to_rate_indices],
                pos_text_embs   = _pos_text_embs,
                neg_text_embs   = _neg_text_embs,
                aspect_pos_embs = _aspect_pos,
                aspect_neg_embs = _aspect_neg,
                aspect_names    = _aspect_names,
            )
            pipeline.unload()
            del pipeline

            clip_map = {r.path: r for r in specvlm_results}
            for local_i, idx in enumerate(to_rate_indices):
                r = clip_map.get(paths[idx])
                if r:
                    _raw_comp = float(r.breakdown.get("Composition", r.score))
                    vlm_scores_rated[local_i]     = float(r.score)
                    comp_scores_rated[local_i]    = _raw_comp
                    _raw_comp_by_path[paths[idx]] = _raw_comp
                    per_photo_breakdowns[idx]           = r.breakdown or {}
                    per_photo_breakdowns[idx]["_grader"] = "clip"
                    scores[idx]                         = float(r.score)

            _p(0.55, "Scoring image quality…")
            if vlm_scores_rated.size:
                print(
                    f"[v2] SpecVLM scores: min={vlm_scores_rated.min():.3f}  "
                    f"max={vlm_scores_rated.max():.3f}  mean={vlm_scores_rated.mean():.3f}"
                )

        except Exception as e_clip:
            import traceback as _tb_clip
            print(f"[v2] SpecVLM scoring FATAL — full traceback:")
            _tb_clip.print_exc()
            _dump_locals(locals())
            raise

    _vram_clear()  # Free grader VRAM before IQA heads load

    # Pre-compute luminance stats — shared by composition analysis, ChiaroscuroHead,
    # and Vintage Lens Protocol in Step 4d.
    # 0.655, not 0.555 — vision grading ends at 0.65 and the bar must not move
    # backwards (it used to sit at "55%" through the whole IQA stage).
    _p(0.655, "Measuring light and contrast…")

    # FRAMEGRADE_LUM_DRAFT=1 lets the JPEG decoder downscale in the DCT domain
    # before the full frame is ever materialised (the idiom at encode_worker.py's
    # image read). Much cheaper, and the result is thumbnailed to 128×128 anyway
    # — but it is NOT bit-identical (a different resample chain shifts Y mean/std
    # slightly, and those feed the Chiaroscuro / Vintage-Lens thresholds), so it
    # stays opt-in until verified against a full before/after score diff.
    _LUM_DRAFT = os.environ.get("FRAMEGRADE_LUM_DRAFT", "0").strip() == "1"

    def _lum_stats(path: str):
        try:
            from PIL import Image as _PILI
            with _PILI.open(path) as _raw:
                if _LUM_DRAFT:
                    try: _raw.draft("RGB", (256, 256))
                    except Exception: pass
                img = _raw.convert("RGB")
            img.thumbnail((128, 128), _PILI.LANCZOS)   # 10-100× faster; lum stats are invariant to scale
            _arr = np.array(img, dtype=np.float32)
            _Y   = 0.299 * _arr[:, :, 0] + 0.587 * _arr[:, :, 1] + 0.114 * _arr[:, :, 2]
            return float(_Y.mean()), float(_Y.std())
        except Exception:
            return 128.0, 60.0   # neutral defaults — no VLP trigger

    from concurrent.futures import ThreadPoolExecutor as _TPELUM
    # Was os.cpu_count() → 16-20 concurrent FULL-RESOLUTION decodes (~72 MB each
    # for a 24 MP frame) before the thumbnail step shrank them. Same free-RAM
    # derived cap as the early-exit gate; output is bit-identical either way.
    try:
        from early_exit_gate import decode_workers as _dec_workers
        # 200 MB/worker, not the 120 MB default. `_PILI.open()` materialises the
        # full frame and `.convert("RGB")` then makes a SECOND full-resolution
        # copy (PIL's convert copies even when the mode already matches), so a
        # 24 MP frame costs ~145 MB per worker before the 128×128 thumbnail
        # shrinks it — the 120 MB figure sized the pool for one buffer, not two,
        # and let the fan-out over-commit on exactly the biggest files.
        _lum_workers = _dec_workers(len(paths_to_rate), mb_per_worker=200.0)
    except Exception:
        _lum_workers = min(8, max(len(paths_to_rate), 1))
    with _TPELUM(max_workers=max(1, min(_lum_workers, max(len(paths_to_rate), 1)))) as _lpool:
        lum_stats_rated = list(_lpool.map(_lum_stats, paths_to_rate))

    # ── Step 4b: Vision IQA Head (UniQA unified backbone) ───────────────────────
    # scan_mode bypasses IQA — Scan uses composition scores (already set in Step 4a).
    # Full grading runs:
    #   1. run_composition_analysis (Depth → Seg → Chiaroscuro) — within run_vision_heads
    #   2. UniQAHead with YOLO11s-seg routing (empty-scene / layered-frame / standard)
    composition_overrides: dict[str, float] = {}
    chiaroscuro_flags:     dict[str, bool]  = {}
    person_detected_dict:  dict[str, bool]  = {}
    _subject_bboxes_dict:  dict             = {}   # set in the IQA block below; scan_mode skips it

    if scan_mode:
        _p(0.84, "Quick scan complete…")
        print(f"[v2] Scan mode: IQA skipped, {len(paths_to_rate)} photos at CLIP speed")
        tech_scores_rated      = vlm_scores_rated.copy()
        aesthetic_scores_rated = vlm_scores_rated.copy()
    elif not to_rate_indices:
        # Nothing left to rate (all photos cached or disqualified). The IQA block
        # below indexes embs[np.array(to_rate_indices)] — an EMPTY list becomes a
        # float64 array and raises "arrays used as indices must be of integer
        # type", crashing the worker. Skip IQA; cached/disqualified scores already
        # in `scores` carry through to the gallery.
        print("[v2] No photos to rate (all cached/disqualified) — skipping IQA heads")
        tech_scores_rated      = np.array([], dtype=np.float32)
        aesthetic_scores_rated = np.array([], dtype=np.float32)
    else:
        _p(0.66, f"Scoring image quality — {len(paths_to_rate)} photos…")
        # Per (folder, preset) key so two folders never share a checkpoint.
        import hashlib as _hl_iqa
        _iqa_key = _hl_iqa.sha1(f"{folder_path}|{preset}".encode()).hexdigest()[:16]
        try:
            iqa_embs  = embs[np.asarray(to_rate_indices, dtype=np.intp)]   # (M, 1536); intp = safe even if empty
            _vlm_bds  = [per_photo_breakdowns[idx] for idx in to_rate_indices]

            # ISOLATED subprocess — the grade worker never loads the IQA GPU model
            # itself (see _iqa_via_subprocess / iqa_worker.py). This removes the last
            # in-worker GPU load and with it the 0xC0000005 windowed-crash class.
            with _ProgressTicker(_p, 0.66, 0.83, f"Scoring image quality — {len(paths_to_rate)} photos…"):
                iqa_out = _iqa_resumable(
                    ckpt_key            = _iqa_key,
                    progress            = _p,
                    image_paths         = paths_to_rate,
                    image_embeddings    = iqa_embs,
                    prompt_embedding    = _prompt_emb,
                    clip_scores         = vlm_scores_rated,
                    genre_ref_embs      = _genre_ref_embs,
                    lum_stats           = lum_stats_rated,
                    comp_eligible_paths = _comp_eligible,
                    vlm_breakdowns      = _vlm_bds,
                )

            tech_scores_rated        = iqa_out["quality"]                      # (M,) UniQA technical
            aesthetic_scores_rated   = vlm_scores_rated                        # (M,) VLM aesthetic (Step 4a)
            iqa_breakdowns           = iqa_out["breakdowns"]                   # list[dict]
            composition_overrides    = iqa_out.get("composition_overrides",  {})
            chiaroscuro_flags        = iqa_out.get("chiaroscuro_flags",      {})
            person_detected_dict     = iqa_out.get("person_detected",        {})
            _subject_bboxes_dict     = iqa_out.get("subject_bboxes",       {})

            for local_i, idx in enumerate(to_rate_indices):
                per_photo_breakdowns[idx].update(iqa_breakdowns[local_i])
                # Apply over-the-shoulder portrait composition override
                _opath = paths[idx]
                if _opath in composition_overrides:
                    per_photo_breakdowns[idx]["Composition"] = composition_overrides[_opath]

            _p(0.84, "Quality scoring complete…")
            try:
                from vision_grading_heads import release_iqa_models as _rel_iqa
                _rel_iqa()
            except Exception as _e_rel:
                print(f"[v2] IQA singleton release skipped: {_e_rel}")
            _vram_clear()
            print(f"[v2] IQA heads: {len(paths_to_rate)} photos scored")
            if composition_overrides:
                print(f"[v2] Composition overrides: {len(composition_overrides)} images "
                      f"(over-the-shoulder portrait → 0.85)")
            _n_ch = sum(1 for v in chiaroscuro_flags.values() if v)
            if _n_ch:
                print(f"[v2] Chiaroscuro: {_n_ch}/{len(chiaroscuro_flags)} images flagged "
                      f"(VLP forced, YOLO soft penalty waived)")

        except Exception as e_iqa:
            # DEGRADE, never crash the worker: if the isolated IQA subprocess fails
            # (or is killed), fall back to the SpecVLM/CLIP technical scores so the
            # cull still completes with sensible grades instead of dying.
            import traceback as _tb_iqa
            print(f"[v2] IQA subprocess failed ({e_iqa}) — degrading to CLIP tech scores")
            _tb_iqa.print_exc()
            tech_scores_rated      = vlm_scores_rated.copy()
            aesthetic_scores_rated = vlm_scores_rated
            # composition_overrides / chiaroscuro_flags / person_detected_dict /
            # _subject_bboxes_dict keep their pre-initialised {} defaults; no
            # per-photo IQA breakdown to merge.

    _grader_status.update({"mode": "iqa_heads" if not scan_mode else "clip_only",
                           "verify_used": False, "photos_last": len(paths_to_rate), "error": None})

    # ── Step 4c: Fine-art anchor similarity + Min-Max stretch ────────────────
    # Raw cosine sims cluster in a narrow band (e.g., 0.28–0.42) because all street
    # photos share some similarity to the anchor. A naive (sim+1)/2 map compresses
    # everything into 0.64–0.71 — useless for differentiation.
    # Min-Max normalization stretches the batch distribution to full [0,1] range,
    # giving fine-art semantic alignment equal mathematical weight to technical metrics.
    _fine_art_sims_all = np.zeros(n, dtype=np.float32)
    if _fine_art_anchor is not None:
        _fine_art_sims_all = (embs @ _fine_art_anchor).astype(np.float32)  # (n,) raw cosine

    fine_art_scores_rated = np.full(len(paths_to_rate), 0.5, dtype=np.float32)
    if _fine_art_anchor is not None and len(to_rate_indices) > 0:
        _fa_sims_rated = _fine_art_sims_all[_rate_idx]
        _fa_lo   = float(_fa_sims_rated.min())
        _fa_hi   = float(_fa_sims_rated.max())
        _fa_span = max(_fa_hi - _fa_lo, 1e-4)
        # Stretch rated-batch sims to [0,1]
        fine_art_scores_rated = np.clip(
            (_fa_sims_rated - _fa_lo) / _fa_span, 0.0, 1.0
        ).astype(np.float32)
        # Normalise all-image sims using same batch statistics so the Soft-Focus Gate
        # threshold is consistent with the per-image fine-art scores used in Step 4d.
        _fine_art_sims_all = np.clip(
            (_fine_art_sims_all - _fa_lo) / _fa_span, 0.0, 1.0
        ).astype(np.float32)
        print(
            f"[v2] Fine-art sims (raw): min={_fa_lo:.3f}  max={_fa_hi:.3f}  "
            f"-> stretched to [0,1]  rated mean={fine_art_scores_rated.mean():.3f}"
        )

    # ── Step 4d: Score fusion with Vintage Lens Protocol + Anchor Floor ────────
    # Base formula: q * 0.35 + q * 0.65 = q  (t == a == UniQA quality score)
    #
    # Vintage Lens Protocol fires when BOTH conditions hold:
    #   (a) Image is low-light or low-contrast: mean_lum < 40 (0-255) OR std < 30
    #   (b) UniQA quality ≥ 0.556 — confirms intentional quality above neutral.
    # On trigger: quality weight drops to 0.75; freed 0.25 reallocates to fine-art sem.
    #   Triggered formula: q * 0.10 + q * 0.65 + fine_art_sem * 0.25 = q * 0.75 + fa * 0.25
    #
    # YOLO Soft Penalty: silhouette-in-dark-scene photos stay in IQA but receive
    # -0.15 to their fused score before Anchor Floor evaluation.
    #
    # Creative Director Anchor Floor: if UniQA quality ≥ 0.611 OR fine-art normalised
    # similarity ≥ 0.75 → enforce overall_score = max(score, _ANCHOR_FLOOR).
    # This guarantees compositionally elite or fine-art-aligned photos can never drop
    # below the Strong bucket threshold due to IQA penalties.
    _AES_VLP_THRESHOLD    = 0.556   # TOPIQ NR threshold for VLP trigger
    _AES_ANCHOR_THRESHOLD = 0.72    # TOPIQ NR threshold for Anchor Floor (raised from 0.611 — old value was calibrated for broken UniQA that always returned 0.5)
    _ANCHOR_FLOOR         = 0.58   # BELOW STRONG_THRESH 0.60 — see note at the
                              # Anchor Floor step: a protective floor must
                              # never mint a Strong by itself (same rationale
                              # as the archetype floors in step 7).
    _FA_ANCHOR_THRESHOLD  = 0.75                 # normalised fine-art sim threshold

    _p(0.86, "Combining scores…")

    _ARCHETYPE_TEMP = 0.15

    M = len(to_rate_indices)

    # ── Pre-extract per-image arrays ──────────────────────────────────────────
    # Role-based extraction: Qwen emits niche-specific axis names ("Light
    # Atmosphere", "Sense Of Place", …), so a hardcoded canonical-key lookup
    # left every non-street niche fusing on 0.5 placeholder defaults — the
    # niche axes were computed and then ignored. aspect_role() maps each axis
    # onto its fusion slot; multiple axes with the same role are averaged.
    from niche_registry import aspect_role as _aspect_role

    def _role_arr(role: str, fallback: "Optional[np.ndarray]" = None,
                  default: float = 0.5) -> np.ndarray:
        out = np.full(len(to_rate_indices), default, dtype=np.float32)
        if fallback is not None:
            out = fallback.astype(np.float32).copy()
        for li, idx in enumerate(to_rate_indices):
            vals = [
                float(v) for k, v in per_photo_breakdowns[idx].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and not str(k).startswith("_") and _aspect_role(str(k)) == role
            ]
            if vals:
                out[li] = float(np.mean(vals))
        return out

    arr_t     = tech_scores_rated                           # (M,) raw IQA
    arr_a     = aesthetic_scores_rated                      # (M,)
    arr_fa    = fine_art_scores_rated                       # (M,)
    arr_lum   = np.array([ls[0] for ls in lum_stats_rated], dtype=np.float32)
    arr_std   = np.array([ls[1] for ls in lum_stats_rated], dtype=np.float32)
    arr_comp  = _role_arr("comp")
    arr_light = _role_arr("light")
    arr_hc    = _role_arr("human")
    arr_narr  = _role_arr("auth")
    arr_tech  = _role_arr("tech", fallback=arr_t)
    arr_raw_comp = np.array(
        [float(_raw_comp_by_path.get(paths[idx], float(arr_comp[li])))
         for li, idx in enumerate(to_rate_indices)],
        dtype=np.float32,
    )
    arr_has_person     = np.array([person_detected_dict.get(paths[idx], True)   for idx in to_rate_indices], dtype=bool)
    arr_is_ots         = np.array([paths[idx] in composition_overrides           for idx in to_rate_indices], dtype=bool)
    arr_is_chiaroscuro = np.array([bool(chiaroscuro_flags.get(paths[idx], False)) for idx in to_rate_indices], dtype=bool)
    arr_yolo_soft      = np.array([paths[idx] in _yolo_soft_penalized            for idx in to_rate_indices], dtype=bool)
    # Street aesthetic genre-fit signal: 304-probe multi-vector cosine scoring (N,) → (M,)
    _sa = street_aesthetic_scores if street_aesthetic_scores is not None else np.full(n, 0.5, dtype=np.float32)
    arr_street = _sa[_rate_idx]                                                  # (M,)

    # ── Batch archetype projection: one matrix multiply for all M images ──────
    rated_embs  = embs[_rate_idx].astype(np.float32)                          # (M, 1536)
    rated_norms = np.linalg.norm(rated_embs, axis=1, keepdims=True)
    rated_unit  = rated_embs / np.where(rated_norms > 1e-9, rated_norms, 1.0)

    arch_norms = np.linalg.norm(archetype_embs, axis=1, keepdims=True)
    arch_unit  = archetype_embs / np.where(arch_norms > 1e-9, arch_norms, 1.0)

    all_sims = (rated_unit @ arch_unit.T).clip(-1.0, 1.0)                     # (M, 5)
    all_st   = all_sims / _ARCHETYPE_TEMP
    all_st  -= all_st.max(axis=1, keepdims=True)
    all_exp  = np.exp(all_st)
    all_wts  = all_exp / (all_exp.sum(axis=1, keepdims=True) + 1e-9)          # (M, 5)

    # Safety: zero w_messy where max-doc beats messy in raw cosine sim
    _safe = all_sims[:, 4] > all_sims[:, 3]
    all_wts[_safe, 3] = 0.0
    all_wts[_safe]   /= (all_wts[_safe].sum(axis=1, keepdims=True) + 1e-9)

    w_geo     = all_wts[:, 0]
    w_night   = all_wts[:, 1]
    w_layered = all_wts[:, 2]
    w_messy   = all_wts[:, 3]
    w_max_doc = all_wts[:, 4]
    dom_arch  = np.argmax(all_wts, axis=1).astype(np.int32)                   # (M,)

    # ── Gate masks ────────────────────────────────────────────────────────────
    _rival_max = np.column_stack([w_geo, w_layered, w_messy, w_max_doc]).max(axis=1)
    night_gate = (
        ((dom_arch == 1) & (w_night > _rival_max + 0.05))
        | ((w_night > 0.35) & ((arr_lum / 2.55) < 30.0))
    )
    max_gate = (dom_arch == 4)

    geo_sterile  = ((dom_arch == 0) | (w_geo > 0.45)) & (arr_hc < 0.30)
    narr_clutter = (
        ~geo_sterile
        & (arr_t < 0.50)
        & ((arr_narr - arr_t) > 0.10)
        & (arr_hc < 0.55)
    )
    low_tech_clutter = narr_clutter & (arr_t < 0.42) & (dom_arch != 0) & ~night_gate

    arr_narr_eff = arr_narr.copy()
    arr_narr_eff[geo_sterile]  *= 0.50
    arr_narr_eff[narr_clutter] *= 0.50   # mutually exclusive with geo_sterile — no double-apply

    # Peopleless frames score low on the Human/Culture probe by design. The
    # human-weighted archetype blends below (night/layered/max-doc weight HC at
    # 0.30–0.35, and max-doc is usually the dominant archetype) then drag empty
    # architectural/liminal shots as if a subject were missing in error. Use a
    # neutral HC for strongly subject-less frames in the POSITIVE blends only (not
    # in the reject gates), so they're scored on composition/light/atmosphere
    # rather than punished for lacking people.
    arr_hc_eff = np.where(arr_hc < 0.45, np.maximum(arr_hc, 0.58), arr_hc)

    # ── Per-archetype formula scores (all vectorized) ─────────────────────────
    fused_geo     = arr_comp * 0.40 + arr_light * 0.30 + arr_a * 0.30
    # Night: removed artificial comp-floor (was max(comp, 0.70)) and over-generous +0.075
    # constant. Added arr_a (VLM holistic score) so mediocre night shots can't ride the
    # comp floor to Strong. Constant reduced to +0.05 (ambient light bonus only).
    fused_night   = arr_hc_eff * 0.30 + arr_comp * 0.30 + arr_a * 0.20 + arr_t * 0.15 + 0.05
    # Layered + Max_doc: added arr_a (Qwen holistic verdict) so direct VLM quality
    # judgment tempers over-reliance on individual aspects that Qwen may rate loosely.
    fused_layered = arr_comp * 0.35 + arr_hc_eff * 0.35 + arr_narr_eff * 0.15 + arr_a * 0.15
    fused_messy   = np.maximum(0.0, arr_t * 0.35 + arr_a * 0.65 - 0.08)
    fused_max_doc = arr_hc_eff * 0.35 + arr_narr_eff * 0.30 + arr_a * 0.20 + arr_t * 0.15

    fused = (
        w_geo     * fused_geo     +
        w_night   * fused_night   +
        w_layered * fused_layered +
        w_messy   * fused_messy   +
        w_max_doc * fused_max_doc
    )                                                                           # (M,) blend

    # Street aesthetic genre-fit modifier: ±0.06 max based on 304-probe cosine scoring.
    # Centered at 0.50 so neutral images receive no adjustment.
    # The probes are human / decisive-moment centric, so this signal is only valid
    # for street-family niches. For niches where a peopleless frame is legitimate
    # (architecture, travel/cultural, landscape, minimalist, liminal, abstract, …)
    # it would punish strong images purely for lacking a human subject — so gate it.
    try:
        from niche_registry import is_human_centric as _is_human_centric
        _apply_street_fit = _is_human_centric(preset)
    except Exception:
        _apply_street_fit = True
    if _apply_street_fit:
        fused = np.clip(fused + 0.12 * (arr_street - 0.50), 0.0, 1.0)
    else:
        print(f"[v2] Street genre-fit modifier skipped for non-human niche '{preset}'")

    # ── Post-fusion gates (vectorized) ────────────────────────────────────────

    # 1. Vintage Lens Protocol — blend fine-art similarity for dark/pictorialist work
    vlp_mask = arr_is_chiaroscuro | (((arr_lum < 80.0) | (arr_std < 35.0)) & (arr_a >= _AES_VLP_THRESHOLD))
    fused    = np.where(vlp_mask, fused * 0.75 + arr_fa * 0.25, fused)

    # 2. YOLO soft penalty for dark-scene silhouettes (waived for chiaroscuro)
    yolo_pen_mask = arr_yolo_soft & ~arr_is_chiaroscuro
    fused = np.where(yolo_pen_mask, np.maximum(0.0, fused - 0.15), fused)

    # 3. Anchor Floor — genre-matched frames with elite deep-model scores cannot
    # be dragged below 0.58 by the penalty gates. The floor sits BELOW
    # STRONG_THRESH deliberately (mirrors step 7): the old 0.65 floor sat above
    # the Strong threshold, so a single high VLM aesthetic reading minted a
    # guaranteed Strong even when technical quality / composition had failed —
    # a recurring Strong-bucket false positive. Elite frames now survive the
    # gates as high-Mid and reach Strong only when their fused quality agrees.
    doc_strong  = (arr_hc >= 0.70) & (arr_narr >= 0.60)
    anchor_mask = (arr_a >= _AES_ANCHOR_THRESHOLD) | (arr_fa >= _FA_ANCHOR_THRESHOLD) | doc_strong
    fused       = np.where(anchor_mask & (fused < _ANCHOR_FLOOR), _ANCHOR_FLOOR, fused)

    # 4. Smooth messy attenuation (replaces hard ceiling at 0.49)
    # Severity combines direct weight and runner-up signal (comp failure when runner-up).
    # Continuous: 0 penalty at w_messy=0 → ~50% reduction at w_messy=1.
    _n_beats_messy    = (all_wts > w_messy[:, np.newaxis]).sum(axis=1)
    _messy_runner_up  = (_n_beats_messy == 1).astype(np.float32)
    _runner_severity  = _messy_runner_up * np.clip(1.0 - arr_comp / 0.35, 0.0, 1.0)
    _clutter_density  = np.maximum(w_messy, 0.60 * _runner_severity)
    messy_atten       = np.clip(1.0 - 0.40 * _clutter_density, 0.0, 1.0)
    fused            *= messy_atten

    # 5. Technical failure ceilings (archetype-matched images that failed technically)
    night_tech_reject = night_gate & (arr_tech < 0.50) & ~arr_is_chiaroscuro & (arr_lum >= 80.0)
    max_hard_reject   = max_gate  & (arr_tech < 0.35) & (arr_hc < 0.55)
    fused = np.minimum(fused, np.where(night_tech_reject, 0.42, np.inf))
    fused = np.minimum(fused, np.where(max_hard_reject,   0.38, np.inf))

    # 6. Smooth composition penalty (replaces hard penalty + hard cap)
    # The blocked-frame and dead-foreground penalties below are STREET-centric:
    # they read low conventional composition / low Human-Culture as failure. For
    # strongly subject-less frames (architecture, liminal, minimalist, empty travel
    # scenes) that is a genre signal, not a fault — exempt them (Human/Culture <
    # 0.45) so empty shots are not penalised for lacking a conventional subject.
    comp_exempt = np.isin(dom_arch, [0, 1, 4]) | (arr_hc < 0.45)

    # Case A: blocked frame (comp < 0.40) — linear penalty: −0.25 at comp=0, 0 at comp=0.40
    comp_block_sev = np.clip(1.0 - arr_comp / 0.40, 0.0, 1.0)
    fused = np.where(~comp_exempt & (arr_comp < 0.40), fused - 0.25 * comp_block_sev, fused)

    # Case B: dead foreground (comp 0.40–0.54, HC < 0.50) — continuous attenuation
    dead_fg_zone = (arr_comp >= 0.40) & (arr_comp <= 0.54) & (arr_hc < 0.50)
    dead_fg_sev  = (
        np.clip((0.54 - arr_comp) / 0.14, 0.0, 1.0) *
        np.clip((0.50 - arr_hc)   / 0.50, 0.0, 1.0)
    )
    dead_fg_atten = 1.0 - 0.20 * dead_fg_sev
    fused = np.where(~comp_exempt & dead_fg_zone, fused * dead_fg_atten, fused)

    # 7. Archetype floors — protect genre-matched frames from the penalty gates,
    # but never mint a grade: all floors sit in the Mid band (< STRONG_THRESH 0.60).
    # The old 0.62–0.68 floors guaranteed Strong from mid-level signals (geo_assist
    # needed only 18% geo weight + tech ≥ 0.38), inflating the Strong bucket.
    # A photo now reaches Strong only when its actual fused quality is ≥ 0.60.
    fused = np.where(night_gate & (arr_tech >= 0.50),
                     np.maximum(fused, 0.55), fused)
    fused = np.where((dom_arch == 0) & (arr_tech >= 0.50) & (arr_comp >= 0.55) & (arr_a >= 0.55),
                     np.maximum(fused, 0.55), fused)
    fused = np.where(max_gate & (arr_tech >= 0.55) & (arr_light >= 0.35),
                     np.maximum(fused, 0.55), fused)
    geo_assist = (w_geo >= 0.18) & (dom_arch != 3) & ~night_gate & ~max_gate & ~narr_clutter & (arr_t >= 0.38)
    fused = np.where(geo_assist, np.maximum(fused, 0.50), fused)

    # 8. Low-tech clutter override — beats all floors, belongs Weak
    fused = np.where(low_tech_clutter & (fused > 0.39), 0.39, fused)

    fused = np.clip(fused, 0.0, 1.0)

    # ── Route 1: Empty Scene override ────────────────────────────────────────
    route1_dark = ~arr_has_person & (arr_is_chiaroscuro | (arr_lum < 80.0))
    route1_norm = ~arr_has_person & ~arr_is_chiaroscuro & (arr_lum >= 80.0)
    fused_r1_dark = np.clip(arr_comp * 0.55 + arr_a * 0.45, 0.0, 1.0)
    fused_r1_norm = np.clip(arr_comp * 0.40 + arr_light * 0.30 + arr_a * 0.30, 0.0, 1.0)
    fused = np.where(route1_dark, fused_r1_dark, fused)
    fused = np.where(route1_norm, fused_r1_norm, fused)

    # ── Route 2B: Layered Frame Portrait override ─────────────────────────────
    route2b_mask = (arr_hc >= 0.65) & (arr_raw_comp < 0.35) & arr_is_ots
    fused_r2b    = np.clip(arr_a * 0.40 + arr_hc * 0.30 + arr_narr * 0.30, 0.0, 1.0)
    r2b_anchor   = (arr_hc >= 0.70) & (arr_narr >= 0.60) & (fused_r2b < _ANCHOR_FLOOR)
    fused_r2b    = np.where(r2b_anchor, _ANCHOR_FLOOR, fused_r2b)
    fused        = np.where(route2b_mask, fused_r2b, fused)

    # ── Fusion input dump (opt-in, for offline ablation) ─────────────────────
    # FRAMEGRADE_FUSION_DUMP=<path.npz> writes every per-photo signal that feeds
    # Step 4d plus the resulting score. That lets the scoring formula's many
    # hand-tuned constants (archetype weights, VLP 0.556, anchor floor 0.58,
    # soft-focus +0.15, street ±0.06, the penalty gates) be ablated against real
    # user ratings OFFLINE — one grade run, many experiments — instead of
    # re-grading for each variant. Off by default; pure instrumentation.
    _dump_to = os.environ.get("FRAMEGRADE_FUSION_DUMP", "").strip()
    if _dump_to:
        try:
            np.savez(
                _dump_to,
                paths=np.array([paths[i] for i in to_rate_indices], dtype=object),
                fused=fused.astype(np.float32),
                arr_t=arr_t, arr_a=arr_a, arr_fa=arr_fa, arr_lum=arr_lum,
                arr_std=arr_std, arr_comp=arr_comp, arr_light=arr_light,
                arr_hc=arr_hc, arr_narr=arr_narr, arr_tech=arr_tech,
                arr_raw_comp=arr_raw_comp, arr_street=arr_street,
                arr_has_person=arr_has_person, arr_is_ots=arr_is_ots,
                arr_is_chiaroscuro=arr_is_chiaroscuro, arr_yolo_soft=arr_yolo_soft,
                all_wts=all_wts, dom_arch=dom_arch,
                night_gate=night_gate, max_gate=max_gate,
                geo_sterile=geo_sterile, narr_clutter=narr_clutter,
                low_tech_clutter=low_tech_clutter,
                apply_street_fit=np.array([bool(_apply_street_fit)]),
            )
            print(f"[v2] Fusion dump written -> {_dump_to} ({len(to_rate_indices)} photos)")
        except Exception as _e_dump:
            print(f"[v2] Fusion dump skipped: {_e_dump}")

    # ── Write scores + breakdown updates ─────────────────────────────────────
    for _li, _idx in enumerate(to_rate_indices):
        scores[_idx] = float(fused[_li])
        if route2b_mask[_li]:
            per_photo_breakdowns[_idx]["Composition"] = 0.82
        # Log archetype weights into breakdown for UI inspection
        per_photo_breakdowns[_idx]["_arch_w"] = {
            "geo": round(float(w_geo[_li]),     3),
            "night": round(float(w_night[_li]), 3),
            "layer": round(float(w_layered[_li]), 3),
            "messy": round(float(w_messy[_li]), 3),
            "maxdoc": round(float(w_max_doc[_li]), 3),
        }

    # ── Diagnostic counters + summary ─────────────────────────────────────────
    _route1_count        = int((~arr_has_person).sum())
    _route2_count        = int(route2b_mask.sum())
    _blend_count         = int((arr_has_person & ~route2b_mask).sum())
    _vlp_count           = int(vlp_mask.sum())
    _chiaroscuro_vlp     = int((vlp_mask & arr_is_chiaroscuro).sum())
    _penalty_count       = int(yolo_pen_mask.sum())
    _anchor_count        = int(anchor_mask.sum())
    _max_gate_count      = int((max_gate & (arr_tech >= 0.55) & (arr_light >= 0.35)).sum())
    _geo_floor_count     = int(((dom_arch == 0) & (arr_tech >= 0.50) & (arr_comp >= 0.55)).sum())
    _night_floor_count   = int((night_gate & (arr_tech >= 0.50)).sum())
    _max_reject_count    = int(max_hard_reject.sum())
    _night_reject_count  = int(night_tech_reject.sum())
    _comp_block_count    = int((~comp_exempt & (arr_comp < 0.40)).sum())
    _comp_deadfg_count   = int((~comp_exempt & dead_fg_zone).sum())

    if _route1_count:
        print(f"[v2] Route 1 (Empty Scene): {_route1_count} images")
    if _route2_count:
        print(f"[v2] Route 2B (Layered Frame): {_route2_count} images")
    if _blend_count:
        print(f"[v2] Archetype Blend (vectorised): {_blend_count} images")
    if _vlp_count:
        print(f"[v2] VLP: {_vlp_count} triggered"
              + (f" ({_chiaroscuro_vlp} chiaroscuro)" if _chiaroscuro_vlp else ""))
    if _penalty_count:
        print(f"[v2] YOLO soft penalty: {_penalty_count} images −0.15")
    if _anchor_count:
        print(f"[v2] Anchor Floor: {_anchor_count} images → {_ANCHOR_FLOOR}")
    if _max_gate_count:
        print(f"[v2] Max-doc floor: {_max_gate_count} → 0.55 (Mid)")
    if _geo_floor_count:
        print(f"[v2] Geo floor: {_geo_floor_count} → 0.55 (Mid)")
    if _night_floor_count:
        print(f"[v2] Night floor: {_night_floor_count} → 0.55 (Mid)")
    if _max_reject_count:
        print(f"[v2] Max-doc reject: {_max_reject_count} → ≤0.38")
    if _night_reject_count:
        print(f"[v2] Night-tech reject: {_night_reject_count} → ≤0.42")
    if _comp_block_count:
        print(f"[v2] Comp-block smooth penalty: {_comp_block_count} images (comp < 0.40)")
    if _comp_deadfg_count:
        print(f"[v2] Dead-FG smooth attenuation: {_comp_deadfg_count} images")

    # Compatibility stubs for code downstream that referenced these loop-local vars
    _messy_ceiling_count = 0   # replaced by smooth attenuation — no hard cap

    # ── DIAG: target image breakdown (replaces per-image loop prints) ─────────
    scores_arr = np.array(scores, dtype=np.float32)
    # The n == 0 bail-out above should make this unreachable, but this exact print is
    # what crashed the cull, so it stays defended: a diagnostic must never be able to
    # take down a multi-hour run.
    if scores_arr.size:
        print(
            f"[v2] grader scores — min={scores_arr.min():.3f}  "
            f"max={scores_arr.max():.3f}  mean={scores_arr.mean():.3f}  "
            f"median={float(np.median(scores_arr)):.3f}"
        )

    # ── Step 4e: Qwen2.5-VL Ollama fast-scan pass — final score authority ────────
    # Runs AFTER IQA heads so TOPIQ + YOLO data are available for injection.
    # Uses FAST_SCAN_PROMPT_TEMPLATE — outputs global_score + spatial_localization_map
    # ONLY (no text generation).  ~128 predicted tokens vs 512 → 3-5× faster.
    #
    # Deep narrative/geometry text is NOT generated here.  It is produced
    # on-demand when the user selects a photo and calls POST /api/critique/details.
    # Only global_score (overrides fusion) and vlm_bboxes are stored.
    #
    # OPT-IN (FRAMEGRADE_STEP4E=1) as of 2026-06-11: on this 6 GB machine the
    # Ollama VLM is always CPU-bound — the latency probe skipped the pass on
    # every observed run, but each cull still paid ~25-40 s of warmup+eviction
    # to find that out. The refinement only contributed a ±0.08 spatial nudge
    # + 25% blend; bbox overlays fall back to YOLO synthesis regardless.
    import os as _os_4e
    if not scan_mode and _os_4e.environ.get("FRAMEGRADE_STEP4E", "0") == "1":
        try:
            from critique_engine import _check_ollama_available
            from qwen_vlm_grader import (
                execute_vlm_culling_sync,
                warmup_vlm_models,
                resolve_fast_scan_model,
                _FAST_SCAN_MODEL,
                _is_monochrome as _mono_check,
            )
            _ollama_up         = _check_ollama_available()
            _active_scan_model = None   # guard: defined here so finally block never hits NameError
            print(f"[v2] Step 4e: Ollama available={_ollama_up}")
            if _ollama_up:
                try:
                    from pdf_rag import load_concepts as _load_rag4e
                    _rag4e = _load_rag4e()
                except Exception:
                    _rag4e = []
                # Pre-compute RAG phrase embeddings once for per-image top-3 selection.
                # SigLIP image embeddings (embs) are L2-normalised — dot product = cosine sim.
                _rag4e_embs = None
                if _rag4e and _enc_singleton is not None:
                    try:
                        _re_raw  = _enc_singleton.encode_text(_rag4e)            # (R, 1536)
                        _re_nrm  = np.linalg.norm(_re_raw, axis=1, keepdims=True)
                        _rag4e_embs = (_re_raw / (_re_nrm + 1e-9)).astype(np.float32)
                        print(f"[v2] RAG phrase embeddings: {len(_rag4e)} phrases encoded for per-image selection")
                    except Exception as _e_re:
                        print(f"[v2] RAG phrase encoding skipped: {_e_re}")
                _rag_ctx_fallback = (
                    "\n".join(f"- {p}" for p in _rag4e[:3])
                    if _rag4e else ""
                )
                _mode4e        = "story" if "story" in preset.lower() else "competition"
                _vlm4e_ok      = 0
                _active_scan_model = resolve_fast_scan_model()
                _p(0.865, f"Refining composition — {len(paths_to_rate)} photos…")

                print(f"[v2] Step 4e: pre-loading VLM models…")
                # Evict the warm Qwen singleton first — Ollama needs the VRAM
                # or it silently schedules its VLM on CPU (~23 min/image).
                if _qwen_singleton is not None:
                    try:
                        _qwen_singleton.unload()
                    except Exception:
                        pass
                    _qwen_singleton = None
                    _vram_clear()
                warmup_vlm_models()

                # CPU-placement guard: when VRAM is occupied at load time,
                # Ollama silently schedules the VLM on CPU — observed at
                # ~23 min/image (A/B test 2026-06-11) vs ~1-2 s on GPU. A
                # 1-token probe on the now-resident model separates the two:
                # GPU answers in well under a second, CPU takes tens of
                # seconds. Skip the refinement pass rather than stall the cull
                # (it only contributes the ±0.08 spatial nudge + 25% blend).
                import time as _t4e_probe, requests as _rq4e_probe
                _probe_t0 = _t4e_probe.time()
                _rq4e_probe.post(
                    "http://localhost:11434/api/generate",
                    json={"model": _active_scan_model, "prompt": "OK",
                          "stream": False, "options": {"num_predict": 1}},
                    timeout=45,
                )
                _probe_dt = _t4e_probe.time() - _probe_t0
                # 2.5 s: verified run 2026-06-11 measured 5.26 s — which passed
                # a 6 s bar and then EVERY image hit the 8 s generate timeout.
                # GPU-resident answers a 1-token probe in well under a second.
                if _probe_dt > 2.5:
                    raise RuntimeError(
                        f"Ollama 1-token probe took {_probe_dt:.1f}s — model is "
                        f"CPU-bound, skipping Step 4e refinement"
                    )
                print(f"[v2] Step 4e: models resident (probe {_probe_dt:.2f}s) — "
                      f"starting {_active_scan_model} batch")

                # Pre-filter: skip images that are clear Weak rejects after fusion.
                # Gemma confirmation adds nothing for scores < 0.28 — they stay Weak.
                # Ensure street_aesthetic_scores exists even if encoder failed
                if street_aesthetic_scores is None:
                    street_aesthetic_scores = np.full(n, 0.5, dtype=np.float32)

                _VLM4E_REJECT_THRESH = 0.38
                _vlm4e_pairs = [
                    (local_i, idx)
                    for local_i, idx in enumerate(to_rate_indices)
                    if paths[idx] not in _all_disqualified
                    and scores[idx] >= _VLM4E_REJECT_THRESH
                ]
                _vlm4e_skipped = len(to_rate_indices) - len(_vlm4e_pairs) - sum(
                    1 for _, idx in enumerate(to_rate_indices)
                    if paths[idx] in _all_disqualified
                )
                if _vlm4e_skipped > 0:
                    print(f"[v2] Step 4e: skipping {_vlm4e_skipped} clear rejects "
                          f"(fused score < {_VLM4E_REJECT_THRESH}) — "
                          f"{len(_vlm4e_pairs)} images queued for Gemma")

                # Parallel execution: 2 threads keep Ollama's queue full so Python
                # HTTP overhead doesn't sit idle between GPU calls.
                import threading as _t4e
                from concurrent.futures import ThreadPoolExecutor as _TPE4e, as_completed as _ase4e

                _abort  = _t4e.Event()
                _consec = [0]
                _clock  = _t4e.Lock()

                def _comp_fit_from_center(cx: float, cy: float) -> float:
                    # Max dist from any corner to its nearest thirds node: sqrt(0.34²+0.34²) ≈ 0.481
                    thirds = [(0.33, 0.33), (0.33, 0.66), (0.66, 0.33), (0.66, 0.66)]
                    min_dist = min(((cx - nx)**2 + (cy - ny)**2)**0.5 for nx, ny in thirds)
                    return float(max(0.0, 1.0 - min_dist / 0.481))

                def _run_one_vlm4e(pair):
                    local_i, idx = pair
                    if _abort.is_set():
                        return None, idx, local_i
                    _path4e  = paths[idx]
                    _topiq4e = int(round(float(tech_scores_rated[local_i]) * 100))
                    _has_p   = person_detected_dict.get(_path4e, True)
                    # Per-image top-3 RAG: dot product of this image's SigLIP embedding
                    # against all phrase embeddings; inject only the most relevant phrases.
                    if _rag4e_embs is not None and embs is not None:
                        try:
                            _sims   = _rag4e_embs @ embs[idx]        # cosine sim (R,)
                            _top3   = np.argsort(_sims)[-3:][::-1]
                            _rag_ctx = "\n".join(f"- {_rag4e[i]}" for i in _top3)
                        except Exception:
                            _rag_ctx = _rag_ctx_fallback
                    else:
                        _rag_ctx = _rag_ctx_fallback
                    result   = execute_vlm_culling_sync(
                        image_path  = _path4e,
                        mode        = _mode4e,
                        cpu_metrics = {
                            "is_monochrome":   _mono_check(_path4e),
                            "topiq_score":     _topiq4e,
                            "yolo_detections": {"persons": 1 if _has_p else 0},
                        },
                        rag_context = _rag_ctx,
                        model       = _active_scan_model,
                        fast_scan   = True,
                        timeout     = 8,
                    )
                    with _clock:
                        if result is None:
                            _consec[0] += 1
                            if _consec[0] >= 3:
                                _abort.set()
                        else:
                            _consec[0] = 0
                    return result, idx, local_i

                with _TPE4e(max_workers=3) as _pool4e:
                    _futs4e = {_pool4e.submit(_run_one_vlm4e, pair): pair for pair in _vlm4e_pairs}
                    for _fut4e in _ase4e(_futs4e):
                        result, idx, local_i = _fut4e.result()
                        _path_4e = paths[idx]
                        # Gemma → YOLO cascade: if Gemma returned no bbox, fall back to the
                        # highest-confidence YOLO detection for that image.
                        _bbox_center = None
                        if result is not None:
                            _bbox_center = result.get("bbox_center_norm")
                        if _bbox_center is None:
                            _yolo_boxes = _subject_bboxes_dict.get(_path_4e, [])
                            if _yolo_boxes:
                                _yb = _yolo_boxes[0]   # [x1n, y1n, x2n, y2n] normalised [0,1]
                                _bbox_center = ((_yb[0] + _yb[2]) / 2, (_yb[1] + _yb[3]) / 2)
                                if result is None:
                                    print(f"[v2] Gemma miss → YOLO cascade: {Path(_path_4e).name}")
                        _vlm_status = None
                        _had_subject_bbox = _bbox_center is not None
                        if _bbox_center is None:
                            _bbox_center = (0.5, 0.5)  # ghost town: no subjects anywhere
                            print(f"[v2] GHOST_TOWN_DEFAULT — center anchor: {Path(_path_4e).name}")
                            _vlm_status = "GHOST_TOWN_DEFAULT"
                        _comp_fit   = _comp_fit_from_center(*_bbox_center)
                        _topiq_norm   = float(tech_scores_rated[local_i])
                        # Zero-sensor guard: TOPIQ returning 0.0 is a model failure, not blur.
                        # Substitute neutral 0.50 so the blur gate doesn't misfire on a dead sensor.
                        if _topiq_norm < 0.001:
                            print(f"[v2] TOPIQ_SENSOR_FAILURE on {Path(_path_4e).name} — substituting 0.50")
                            _topiq_norm = 0.50
                            _vlm_status = "TOPIQ_SENSOR_FAILURE"
                        elif _topiq_norm < 0.40 and _vlm_status is None:
                            _vlm_status = "CRITICAL_BLUR"
                        # Step 4e is a spatial refinement, NOT a replacement of Step 4d.
                        # Step 4d already encodes TOPIQ, VLM scores, street aesthetic,
                        # narrative, Human/Culture, lighting, and archetype gates.
                        # The only new information Gemma contributes is subject placement
                        # relative to the rule of thirds (comp_fit).  Apply a small nudge
                        # (max ±0.08) so the Step 4d archetype analysis stays the primary
                        # driver while spatial placement provides a minor refinement.
                        _4d_score      = scores[idx]
                        _spatial_nudge = 0.16 * (_comp_fit - 0.50)       # ±0.08 max
                        _post_nudge    = float(np.clip(_4d_score + _spatial_nudge, 0.0, 1.0))

                        # Blend Gemma's global_score (holistic visual verdict) at 25%.
                        # Previously this number was computed and then discarded.
                        # 75% archetype fusion keeps genre logic in charge; 25% Gemma
                        # corrects cases where the formula diverges from what the eye sees.
                        _gemma_gs = None
                        if result is not None:
                            _raw_gs = result.get("global_score")
                            if _raw_gs is not None:
                                try:
                                    _gemma_gs = max(0.0, min(1.0, int(_raw_gs) / 100.0))
                                except (ValueError, TypeError):
                                    pass

                        if _gemma_gs is not None:
                            scores[idx] = float(np.clip(0.75 * _post_nudge + 0.25 * _gemma_gs, 0.0, 1.0))
                            per_photo_breakdowns[idx]["gemma_score"] = round(_gemma_gs, 3)
                            print(f"[v2 4e] {Path(paths[idx]).name}: "
                                  f"4d+nudge={_post_nudge:.3f}  gemma={_gemma_gs:.2f}  "
                                  f"final={scores[idx]:.3f}")
                        else:
                            scores[idx] = _post_nudge

                        # Only overwrite Composition when a real subject bbox
                        # existed (Gemma or YOLO). The ghost-town center default
                        # produces a constant ~0.50 placement value that was
                        # stomping Qwen's actual composition score on every
                        # subject-less frame — the "53% Composition everywhere"
                        # bug in the Judge's Eye. Placement is always stored
                        # separately under a private key for inspection.
                        if _had_subject_bbox:
                            per_photo_breakdowns[idx]["Composition"] = round(_comp_fit, 3)
                        per_photo_breakdowns[idx]["_subject_placement"] = round(_comp_fit, 3)
                        per_photo_breakdowns[idx]["Technical"]    = round(_topiq_norm, 3)
                        if _vlm_status:
                            per_photo_breakdowns[idx]["vlm_status"] = _vlm_status
                        if result is not None:
                            per_photo_breakdowns[idx]["vlm_bboxes"] = result.get("spatial_localization_map", [])
                        _vlm4e_ok += 1

                if _abort.is_set():
                    print(f"[v2] Step 4e: 3 consecutive Ollama failures — aborted early")

                if _vlm4e_ok:
                    scores_arr = np.array(scores, dtype=np.float32)
                    # `_vlm4e_ok` implies images were scored, which implies n > 0 —
                    # but that reasoning depends on the bail-out ~900 lines up. This
                    # exact reduction pattern has taken down a cull three times, so
                    # the invariant is asserted locally rather than inherited.
                    if scores_arr.size:
                        print(
                            f"[v2] VLM fast-scan: {_vlm4e_ok}/{len(paths_to_rate)} scored — "
                            f"min={scores_arr.min():.3f}  max={scores_arr.max():.3f}  "
                            f"mean={scores_arr.mean():.3f}"
                        )
                    # Score stretch REMOVED (2026-06): it rescaled any tightly
                    # clustered batch (range < 0.20) onto [0.18, 0.88], which
                    # manufactured Strong and Weak grades out of trivial relative
                    # differences — a uniformly good shoot got its weakest frames
                    # branded Weak and vice versa. Grades are absolute now: a
                    # clustered batch that is genuinely all-Mid stays all-Mid.
        except Exception as _e4e:
            print(f"[v2] Step 4e Qwen VLM pass skipped: {_e4e}")
        finally:
            # Evict fast-scan model so VRAM is free for PersonalHead / IQA stages.
            if _active_scan_model:
                try:
                    import requests as _rq
                    _rq.post("http://localhost:11434/api/generate",
                             json={"model": _active_scan_model, "keep_alive": 0},
                             timeout=5)
                    print(f"[v2] Step 4e: evicted {_active_scan_model}")
                except Exception:
                    pass

    # ── Anchor-subject bbox fallback ────────────────────────────────────────
    # If Step 4e (Ollama) didn't store vlm_bboxes for a rated image, synthesize
    # an anchor_subject bbox from the highest-confidence YOLO detection so the
    # Vision Critique overlay always has something to draw on the photo.
    _bbox_synth_count = 0
    for _si, _sidx in enumerate(to_rate_indices):
        if per_photo_breakdowns[_sidx].get("vlm_bboxes"):
            continue  # already have Gemma bboxes
        try:
            from PIL import Image as _PILI_DIM
            with _PILI_DIM.open(paths[_sidx]) as _dim_img:
                _iw, _ih = _dim_img.size
        except Exception:
            continue
        _yolo_fb = _subject_bboxes_dict.get(paths[_sidx], [])
        if _yolo_fb:
            _yb = _yolo_fb[0]  # [x1n, y1n, x2n, y2n] normalised [0,1]
            per_photo_breakdowns[_sidx]["vlm_bboxes"] = [{
                "label": "anchor_subject",
                "bbox_2d": [
                    int(_yb[0] * _iw), int(_yb[1] * _ih),
                    int(_yb[2] * _iw), int(_yb[3] * _ih),
                ]
            }]
        else:
            # No detectable subject (architecture, landscape, minimalist, abstract,
            # peopleless travel …). Fall back to a central rule-of-thirds region so
            # the Vision Critique overlay still has a compositional anchor to draw —
            # otherwise these frames show no overlay at all.
            per_photo_breakdowns[_sidx]["vlm_bboxes"] = [{
                "label": "compositional_center",
                "bbox_2d": [
                    int(_iw * 0.17), int(_ih * 0.17),
                    int(_iw * 0.83), int(_ih * 0.83),
                ]
            }]
        _bbox_synth_count += 1
    if _bbox_synth_count:
        print(f"[v2] Anchor-subject bbox synthesised from YOLO for {_bbox_synth_count} images")

    # ── Step 5: PersonalHead adjustment ──────────────────────────────────────
    _p(0.87, "Applying your taste profile…")
    pers         = np.full(n, 0.5, dtype=np.float32)
    final_scores = scores_arr.copy()  # copy so Soft-Focus gate doesn't mutate scores_arr
    _ph_weights  = Path("cache/personal_head.pt")
    if _ph_weights.exists():
        print("[v2] PersonalHead weights found — confidence-adaptive taste blend")
        try:
            # Numpy scorer FIRST: `import personal_head` pulls torch (~349 MB
            # measured) into this CUDA-free worker just to run three matmuls.
            # personal_head_np computes the identical forward pass from a numpy
            # mirror of the same weights (auto-refreshed when the .pt changes).
            # Only fall back to the torch module if the mirror is unusable.
            import personal_head_np as ph_np
            _head_dim = ph_np.head_dim()
            _emb_dim  = int(embs.shape[1]) if embs is not None else 0
            if _head_dim is not None and _emb_dim and _head_dim != _emb_dim:
                # Quality-tier mismatch — an EXPECTED condition, not an error.
                # The taste head was trained against a different encoder, so its
                # weights are meaningless here. The torch path would respond by
                # discarding them and initialising a RANDOM head, whose output
                # still gets blended into every score at the 0.20 floor — noise,
                # not a neutral fallback. Skip the blend; grades stand on the
                # grader alone until the head is retrained on this tier.
                print(f"[v2] Taste head was trained on {_head_dim}-d embeddings but "
                      f"this quality tier emits {_emb_dim}-d — skipping the taste "
                      f"blend (rate photos on this tier to train it)")
                raise _TasteTierMismatch()
            pers = ph_np.score(embs)
            if pers is None:
                print("[v2] numpy taste head unavailable — falling back to torch head")
                import personal_head as ph
                pers = ph.score(embs)
            # ── Confidence-adaptive taste blend (guarded, non-regressive) ──────
            # The old blend was a flat 0.80*grader + 0.20*taste. Problem: when the
            # taste head has no opinion on an image (output ~0.5 — e.g. a genre it
            # was never trained on) that flat 20% just drags the score toward the
            # mean and mushes the distribution. So instead we scale the taste
            # weight by how CONFIDENT the head is (how far its output sits from the
            # neutral 0.5):
            #   • neutral head (~0.5)  → weight collapses to the 0.20 FLOOR → the
            #     result is IDENTICAL to the old 80/20 blend → cannot regress.
            #   • confident head (→0 or →1, i.e. it has learned taste for shots
            #     like this) → weight rises toward the CEIL → taste becomes a
            #     first-class vote, but only where it actually has coverage.
            # Net: adding a few ratings in a new genre raises the head's confidence
            # there, so it speaks up and pulls grades toward your taste automatically
            # — with zero effect anywhere it hasn't learned yet.
            _pers_arr = np.asarray(pers, dtype=np.float32)
            _conf     = np.clip(np.abs(_pers_arr - 0.5) / 0.5, 0.0, 1.0)   # 0=neutral … 1=certain
            _w_floor  = 0.20                                                # == today's weight
            # ── Baseline-size-scaled taste authority ─────────────────────────
            # The ceiling used to be a flat 0.35 no matter how many ratings the
            # user had banked — a 100+ rating baseline could only ever NUDGE
            # grades, never lead them. Now evidence earns authority: the ceiling
            # grows with the durable star-rating count (ratings_store):
            #     <25 ratings → 0.35   (head still young; unchanged behaviour)
            #     ≥25         → 0.45
            #     ≥50         → 0.55
            #     ≥100        → 0.70   taste LEADS, the machine grader tiebreaks
            # Confidence scaling below is untouched: even at a 0.70 ceiling an
            # image only gets that weight if the head is actually confident
            # about it; neutral (~0.5) outputs still collapse to the floor.
            try:
                import ratings_store as _rs
                _n_ratings = len(_rs.load())
            except Exception:
                _n_ratings = 0
            if   _n_ratings >= 100: _w_ceil = 0.70
            elif _n_ratings >= 50:  _w_ceil = 0.55
            elif _n_ratings >= 25:  _w_ceil = 0.45
            else:                   _w_ceil = 0.35
            _env_cap = os.environ.get("FRAMEGRADE_PH_WEIGHT_MAX", "").strip()
            if _env_cap:
                # Explicit override wins as a HARD CAP (escape hatch); sanity-
                # clamped so it can neither go negative-float nor past 0.80.
                _w_ceil = min(max(float(_env_cap), _w_floor), 0.80)
            _w_pers   = _w_floor + (_w_ceil - _w_floor) * _conf            # per-image taste weight
            final_scores = (1.0 - _w_pers) * scores_arr + _w_pers * _pers_arr
            _n_op = int((_conf > 0.20).sum())
            print(f"[v2] PersonalHead blend: {_n_ratings} banked ratings → "
                  f"taste weight {_w_floor:.2f}–{_w_ceil:.2f} "
                  f"(mean {float(_w_pers.mean()):.3f}); {_n_op}/{n} images got a confident taste vote")
        except _TasteTierMismatch:
            pass          # already reported above; grades use the grader alone
        except Exception as _e:
            import traceback
            print(f"[v2] PersonalHead blend failed: {_e}")
            traceback.print_exc()
            _dump_locals(locals())
    else:
        print("[v2] PersonalHead weights absent — using raw grader scores")

    # ── Step 5c: Soft-Focus Protection Gate ──────────────────────────────────
    # Images with high cosine similarity to the fine-art anchor (> 0.68) receive
    # a flat +0.15 score boost applied before quantile bucketing. This prevents
    # atmospheric, low-contrast, or soft-focus fine-art frames from being dropped
    # to Weak purely because pixel-sharpness metrics ranked them lower.
    # The gate is additive, not multiplicative — it shifts the score up the
    # distribution without changing relative ordering within the fine-art cohort.
    # Soft-Focus Gate: only fires for images with strong fine-art similarity (> 0.75)
    # AND a base score already in solid Mid territory (≥ 0.50). This prevents
    # Mid/borderline shots with merely incidental atmospheric resemblance from
    # being inflated into Strong. The +0.15 boost is reserved for shots that
    # already earned Mid on their own merit and have genuine fine-art character.
    _sfpg_count = 0
    if _fine_art_anchor is not None:
        for i in range(n):
            _base_score = float(final_scores[i])
            _fa_sim     = float(_fine_art_sims_all[i])
            if _fa_sim > 0.75 and _base_score >= 0.50:
                final_scores[i] = float(np.clip(_base_score + 0.15, 0.0, 1.0))
                _sfpg_count += 1
        if _sfpg_count:
            print(
                f"[v2] Soft-Focus Gate: {_sfpg_count} images boosted +0.15 "
                f"(fine_art_sim > 0.75, base_score ≥ 0.50)"
            )

    # ── Step 5b: Duplicate sim-flag assignment based on final_scores ──────────
    _p(0.88, "Marking duplicates…")
    from pipeline_stages import mark_duplicate_groups as _mark_duplicate_groups
    sim_flags = _mark_duplicate_groups(cluster_ids, final_scores, paths)

    # ── Step 6: Absolute grade thresholds ────────────────────────────────────
    # Strong ≥ 0.60  |  Mid 0.41–0.59  |  Weak ≤ 0.40
    _p(0.89, "Assigning grades…")
    from pipeline_stages import assign_grades as _assign_grades
    final_scores, grades = _assign_grades(
        final_scores, paths, np,
        strong_thresh=STRONG_THRESH, mid_thresh=MID_THRESH,
        strong_label=GRADE_STRONG, mid_label=GRADE_MID, weak_label=GRADE_WEAK)

    # ── Step 7: EXIF + LanceDB ────────────────────────────────────────────────
    _p(0.90, "Reading photo details…")
    from pipeline_stages import read_exif_timestamps as _read_exif_timestamps
    timestamps = _read_exif_timestamps(paths)

    _p(0.92, "Saving results…")
    lance_ok = False
    try:
        import lance_store as ls
        import traceback as _tb_lance
        print(f"[v2] LanceDB WRITE START — {n} records → {ls._DB_DIR}")
        # Built and upserted in RAM-bounded slices, NOT all at once.
        #
        # `embedding: embs[i].tolist()` turns a 6 KB float32 row into a 1536-entry
        # Python list (~49 KB of boxed floats). Materialising every record first
        # meant that at a 5 000-photo import this list alone held ~250 MB — and it
        # stayed alive through the gallery build (which makes a SECOND set of the
        # same lists) and through upsert_batch's own _pad() copy (a THIRD). Three
        # simultaneous copies of data that is 30 MB as float32.
        # Slicing caps the boxed-float working set at _LANCE_CHUNK records, and the
        # slice is dropped before the next one is built. The write is identical:
        # upsert_batch is a merge_insert keyed on `path`, so N calls of C records
        # commit exactly what one call of N records did.
        _LANCE_CHUNK = int(os.environ.get("FRAMEGRADE_LANCE_CHUNK", "500"))
        _written = 0
        for _c0 in range(0, n, _LANCE_CHUNK):
            _c1 = min(_c0 + _LANCE_CHUNK, n)
            lance_records: list[dict] = []
            for i in range(_c0, _c1):
                bd = {"aesthetic": round(float(scores_arr[i]), 3),
                      "personal":  round(float(pers[i]),       3)}
                if per_photo_breakdowns[i]:
                    bd.update(_sanitize_bd(per_photo_breakdowns[i]))
                lance_records.append({
                    "path":           paths[i],
                    "embedding":      embs[i].tolist(),
                    "score":          float(final_scores[i]),
                    "personal_score": float(pers[i]),
                    "grade":          grades[i],
                    "reasoning_log":  "",      # LLM layer removed; field kept for schema compat
                    "breakdown":      bd,
                    "exif_ts":        timestamps[i],
                })
            ls.upsert_batch(lance_records)
            _written += len(lance_records)
            del lance_records          # free this slice's boxed floats immediately
        ls.compact_after_write()
        ls.close_table()
        lance_ok = True
        gc.collect()
        print(f"[v2] LanceDB WRITE OK — {_written} records committed "
              f"(chunk={_LANCE_CHUNK})")
        # Cull finished and grades are durably in LanceDB — clear the resume
        # checkpoint so the next run of this folder starts clean.
        try:
            import hashlib as _hl_done
            _ck_done = (Path(__file__).resolve().parent.parent / "cache" / "grade_ckpt" /
                        f"{_hl_done.sha1(f'{folder_path}|{preset}'.encode()).hexdigest()[:16]}.json")
            _ck_done.unlink(missing_ok=True)
            # Same for the IQA checkpoint — the work it protects is now in the DB.
            _iqa_ckpt_clear(_hl_done.sha1(f'{folder_path}|{preset}'.encode()).hexdigest()[:16])
        except Exception:
            pass
    except Exception as _e_lance:
        import traceback as _tb_lance
        print(f"[v2] !!! LanceDB WRITE FAILED: {_e_lance}")
        _tb_lance.print_exc()
        lance_ok = False

    # ── Step 8: Gallery response ──────────────────────────────────────────────
    _p(0.94, "Building your gallery…")
    # Restore the user's durable star ratings onto fresh gallery items so a
    # re-cull never wipes the taste baseline (the catalog defaults stars=0).
    try:
        import ratings_store as _ratings_store
        _user_ratings = _ratings_store.load()
        if _user_ratings:
            print(f"[v2] Restoring {len(_user_ratings)} durable star ratings onto gallery")
    except Exception as _e_rs:
        print(f"[v2] ratings_store load skipped: {_e_rs}")
        _user_ratings = {}
    gallery = []
    for i, path in enumerate(paths):
        fn        = Path(path).name
        breakdown = {
            "Aesthetic": round(float(scores_arr[i]), 3),
            "Personal":  round(float(pers[i]),       3),
        }
        if per_photo_breakdowns[i]:
            breakdown.update(_sanitize_bd(per_photo_breakdowns[i]))
        _fscore = round(float(final_scores[i]), 3)
        gallery.append({
            "id":              path,
            "path":            path,
            "filename":        fn,
            "grade":           grades[i],
            "score":           _fscore,
            "overall_score":   _fscore,
            "rating":          _fscore,
            "human_perception":round(float(pers[i]),         3),
            "personal_score":  round(float(pers[i]),         3),
            "embedding":       embs[i].tolist(),
            "breakdown":       breakdown,
            "critique":        "",
            "reasoning_log":   "",
            "is_verified":     False,
            "exif_ts":         float(timestamps[i]),
            "stars":           int(_user_ratings.get(path, 0)),
            "reject":          cluster_ids[i] >= 0 and not sim_flags[i].startswith("★"),
            "sim_flag":        sim_flags[i],
            "cluster_id":      int(cluster_ids[i]),
        })

    # ── Face / subject-focus signals ──────────────────────────────────────────
    # REPORTED ONLY: nothing here feeds the score. Aesthetic grading has no
    # notion of whether the subject is actually sharp, which is a fact about the
    # photo rather than a matter of taste — a portrait focused on the ear is a
    # reject no aesthetic model catches.
    #
    # Gated on the person detector, which has already run: measured at ~430
    # ms/photo, running it on a whole folder would add ~3.7 min to a 514-photo
    # cull to answer a question that only applies where there is a person. On
    # street work most frames have no face at all (8 of 60 in a real sample),
    # so this stays cheap; on portrait/event work it is where the value is.
    from pipeline_stages import attach_face_signals as _attach_face_signals
    _attach_face_signals(gallery, person_detected_dict)

    # embs used for last time above (embedding tolist); release before catalog write
    del embs
    gc.collect()

    # Merge cached images back into the gallery (preserving folder sort order)
    if cached_rows:
        gallery_by_path = {g["path"]: g for g in gallery}
        gallery = [
            gallery_by_path[p] if p in gallery_by_path else _cached_to_gallery(cached_rows[p])
            for p in all_paths
            if p in gallery_by_path or p in cached_rows
        ]

    # ── Step 8b: Atomic server-side catalog.json write ───────────────────────
    # Keeps catalog.json in sync immediately after LanceDB upsert, regardless of
    # whether the frontend later calls POST /api/catalog/save.  Atomic rename
    # prevents a partially-written file from corrupting the next app load.
    from pipeline_stages import write_catalog as _write_catalog
    _write_catalog(gallery)

    # ── Step 9: NSGA-III multi-objective sequencing ───────────────────────────
    # mogco_target=0 means "skip here — grade_worker.py runs the sequencer once
    # across all combined folders after run_v2() returns."
    _p(0.96, "Sequencing your story…")
    mogco_seq:   list[dict] = []
    mogco_error: str        = ""
    if mogco_target > 0 and siglip_ok and lance_ok:
        try:
            # run_nsga3_sequence_with_vlm / SequencerConstraintError no longer
            # exist in nsga3_sequencer — the import crashed every mogco run
            # (and the stale except clause turned it into an UnboundLocalError).
            from nsga3_sequencer import run_creative_story_sequencer

            # Pass brief so the sequencer can apply literal pre-filter
            try:
                from specvlm_pipeline import _CD_BRIEF as _seq_brief
            except Exception:
                _seq_brief = ""

            # Pass Strong + Mid candidates with embeddings, reasoning logs, and breakdown
            seq_candidates = [
                {
                    "path":          g["path"],
                    "score":         g["score"],
                    "embedding":     np.array(g["embedding"], dtype=np.float32),
                    "reasoning_log": g["reasoning_log"],
                    "breakdown":     g.get("breakdown", {}),
                }
                for g in gallery
                if g["grade"] in (GRADE_STRONG, GRADE_MID)
            ]

            selected = run_creative_story_sequencer(
                seq_candidates,
                target = mogco_target,
                brief  = _seq_brief,
            )

            info_by_path = {g["path"]: g for g in gallery}
            for rank, frame in enumerate(selected):
                base = {
                    k: v for k, v in
                    info_by_path.get(frame["path"], {"path": frame["path"]}).items()
                    if k != "embedding"
                }
                base.update({
                    "slot":             frame.get("slot", _SEQUENCE_SLOTS[rank % len(_SEQUENCE_SLOTS)]),
                    "slot_role":        frame.get("slot_role", ""),
                    "slot_score":       frame.get("slot_score", 0.0),
                    "mogco_objectives": frame.get("nsga3_objectives", {}),
                    "engine":           "nsga3",
                })
                mogco_seq.append(base)

        except Exception as e:
            import traceback
            mogco_error = str(e)
            print(f"[v2] NSGA-III sequencing failed: {e}")
            traceback.print_exc()
            _dump_locals(locals())

    _p(1.0, "Done")

    all_grades = [g["grade"] for g in gallery]
    strong = sum(1 for g in all_grades if g == GRADE_STRONG)
    mid    = sum(1 for g in all_grades if g == GRADE_MID)
    weak   = sum(1 for g in all_grades if g == GRADE_WEAK)
    print(f"[v2] SUMMARY: {len(gallery)} photos → Strong={strong}  Mid={mid}  Weak={weak}  (new={n}  cached={len(cached_rows)})")

    _ram_peaks = _ram_probe.summary() if _ram_probe is not None else []

    return {
        "gallery":        gallery,
        "mogco_sequence": mogco_seq,
        "mogco_error":    mogco_error,
        "strong":         strong,
        "mid":            mid,
        "weak":           weak,
        "total":          len(gallery),
        "pipeline":       "v2",
        "scan_mode":      scan_mode,
        "ram_peaks":      _ram_peaks,
    }


_SEQUENCE_SLOTS = [
    "Opening",
    "Act 1",
    "Act 2",
    "Climax",
    "Resolution",
    "Coda",
    "Epilogue",
]


# ── File management (no-op stubs) ─────────────────────────────────────────────

def sort_files(folder_path: str, gallery: list[dict], copy: bool = False) -> dict:
    """Photos stay in their original folder. Returns a summary only."""
    strong = sum(1 for g in gallery if g.get("grade") == GRADE_STRONG)
    mid    = sum(1 for g in gallery if g.get("grade") == GRADE_MID)
    weak   = sum(1 for g in gallery if g.get("grade") == GRADE_WEAK)
    return {
        "moved":   0,
        "errors":  [],
        "dirs":    {},
        "summary": {"strong": strong, "mid": mid, "weak": weak},
        "message": "Photos remain in original folder (file moving disabled)",
    }


# ── CLI entry point ────────────────────────────────────────────────────────────
# Usage:  venv\Scripts\python.exe src/grade_pipeline_v2.py \
#             --input_dir "path/to/photos" --mode story
#
# Logs flow to stdout in real time.  Watch for:
#   [v2] Qwen2.5-VL scores  → Step 4a transformers path active
#   [vlm_cull] …score=…     → Step 4e Ollama culling pass active
#   [v2] Qwen VLM culling   → Step 4e summary line

if __name__ == "__main__":
    import argparse, sys, time as _cli_time

    _parser = argparse.ArgumentParser(
        description="FrameGrade — vision grading pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _parser.add_argument(
        "--input_dir", required=True,
        help="Folder of JPEG/PNG/TIFF images to grade",
    )
    _parser.add_argument(
        "--mode", default="story",
        choices=["story", "competition", "classic street", "street",
                 "architectural", "fine art", "liminal"],
        help="Grading mode — affects VLM prompt framing (default: story)",
    )
    _parser.add_argument(
        "--force-rescan", dest="force_rescan",
        action=argparse.BooleanOptionalAction, default=True,
        help="Re-grade even if images are already in LanceDB (default: True). "
             "Pass --no-force-rescan to reuse cached grades.",
    )
    _args = _parser.parse_args()

    # Ensure src/ is on the path when called from project root
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 62)
    print(f"  FrameGrade — pipeline test run")
    print(f"  input_dir : {_args.input_dir}")
    print(f"  mode      : {_args.mode}")
    print("=" * 62)

    _t0 = _cli_time.time()

    def _cli_progress(frac: float, msg: str) -> None:
        bar  = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        pct  = int(frac * 100)
        print(f"\r  [{bar}] {pct:3d}%  {msg[:60]:<60}", end="", flush=True)
        if frac >= 1.0:
            print()

    result = run_v2(
        folder_path   = _args.input_dir,
        preset        = _args.mode,
        force_rescan  = _args.force_rescan,
        progress      = _cli_progress,
    )
    print()

    elapsed = _cli_time.time() - _t0

    if "error" in result:
        print(f"\n  ERROR: {result['error']}")
        sys.exit(1)

    gallery = result.get("gallery", [])
    strong  = result.get("strong", 0)
    mid     = result.get("mid",    0)
    weak    = result.get("weak",   0)
    total   = result.get("total",  0)

    print("=" * 62)
    print(f"  Results — {total} photos graded in {elapsed:.1f}s "
          f"({elapsed/max(total,1):.1f}s/photo)")
    print(f"  Strong ✅  {strong:3d}  ({strong/max(total,1)*100:.0f}%)")
    print(f"  Mid    ⚠️   {mid:3d}  ({mid/max(total,1)*100:.0f}%)")
    print(f"  Weak   ❌  {weak:3d}  ({weak/max(total,1)*100:.0f}%)")
    print("=" * 62)
    print()
    print("  Top 5 by score:")
    _top = sorted(gallery, key=lambda g: g["score"], reverse=True)[:5]
    for _g in _top:
        _bd  = _g.get("breakdown", {})
        _vlm = _bd.get("vlm_verdict", "")[:60]
        print(
            f"    {_g['score']:.2f}  {_g['grade']:<12}  {_g['filename']}"
            + (f"\n          └─ {_vlm}" if _vlm else "")
        )
    print()
