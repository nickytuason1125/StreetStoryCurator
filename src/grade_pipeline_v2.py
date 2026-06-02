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
Step 5  PersonalHead adjusts scores by learned user preference (if weights present).
Step 6  Relative quantile buckets: top 25% → Strong / bottom 20% → Weak / rest → Mid.
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

_HIST_ANCHOR_CACHE = Path(__file__).parent.parent / "cache" / "hist_anchor.json"
_HIST_ANCHOR_TTL   = 86_400   # 24 hours in seconds


def _write_anchor_cache(p75: "Optional[float]", p20: "Optional[float]", n: int) -> None:
    import json as _json, time as _time
    try:
        _HIST_ANCHOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _HIST_ANCHOR_CACHE.write_text(_json.dumps({
            "timestamp":  _time.time(),
            "hist_p75":   p75,
            "hist_p20":   p20,
            "n_sessions": n,
        }))
    except Exception:
        pass


def _historical_anchor() -> "tuple[Optional[float], Optional[float], int]":
    """
    Recency-weighted p75/p20 from past sessions in LanceDB, with a 24-hour file cache.

    Cache hit  → instant, zero DB I/O (safe at 10k+ photos).
    Cache miss → full query, results written to cache/hist_anchor.json for next run.

    Groups rows by directory (folder = one shoot session). Each session passes a
    contamination filter before contributing to the anchor:
      - ≥ 5 photos with score > 0.01
      - score range ≥ 0.04 (rejects synthetic/test runs with uniform scores)
      - mean score in (0.06, 0.94) (rejects degenerate all-high/all-low sessions)
      - p75 ≥ 0.15 (rejects sessions where almost everything was weak)

    Weight = exp(−age_days / 60): sessions decay to ~37% at 60 days, ~5% at 180 days.
    Requires ≥ 3 valid sessions before returning a non-None anchor.

    Returns (hist_p75, hist_p20, n_valid_sessions).
    """
    import json as _json
    import time as _time

    # ── Cache read ───────────────────────────────────────────────────────────
    try:
        if _HIST_ANCHOR_CACHE.exists():
            _cached = _json.loads(_HIST_ANCHOR_CACHE.read_text())
            _age    = _time.time() - _cached.get("timestamp", 0)
            if _age < _HIST_ANCHOR_TTL:
                _hp75 = _cached.get("hist_p75")
                _hp20 = _cached.get("hist_p20")
                _n    = int(_cached.get("n_sessions", 0))
                print(
                    f"[v2] Historical anchor: cache hit "
                    f"({_age / 3600:.1f}h old, {_n} sessions)"
                )
                return _hp75, _hp20, _n
    except Exception:
        pass   # corrupted cache — fall through to full query

    # ── Full LanceDB query ───────────────────────────────────────────────────
    try:
        import os as _os
        from collections import defaultdict as _dd
        from lance_store import query_all as _lance_all

        rows = _lance_all(min_score=0.01)
        if not rows:
            _write_anchor_cache(None, None, 0)
            return None, None, 0

        sessions: dict = _dd(list)
        for r in rows:
            folder = _os.path.dirname(r.get("path", ""))
            if folder:
                sessions[folder].append(r)

        now = _time.time()
        session_stats = []
        for folder, photos in sessions.items():
            scores = [float(p["score"]) for p in photos if (p.get("score") or 0) > 0.01]
            if len(scores) < 5:
                continue
            s_arr = np.array(scores, dtype=np.float32)
            score_range = float(s_arr.max() - s_arr.min())
            score_mean  = float(s_arr.mean())
            score_p75   = float(np.percentile(s_arr, 75))
            if score_range < 0.04:                              # uniform = test/synthetic
                continue
            if score_mean > 0.94 or score_mean < 0.06:         # degenerate session
                continue
            if score_p75 < 0.15:                                # everything graded weak
                continue

            # best timestamp: max EXIF ts across photos, fall back to file mtime
            ts_values = [p.get("exif_ts", 0.0) for p in photos if (p.get("exif_ts") or 0) > 1e8]
            if ts_values:
                session_ts = max(ts_values)
            else:
                try:
                    session_ts = _os.path.getmtime(photos[0]["path"])
                except Exception:
                    session_ts = now - 365 * 86400   # assume 1 year old if unknown

            age_days = (now - session_ts) / 86400.0
            weight   = float(np.exp(-age_days / 60.0))          # 60-day half-life

            session_stats.append({
                "p75": score_p75,
                "p20": float(np.percentile(s_arr, 20)),
                "weight": weight,
                "n": len(scores),
                "age_days": round(age_days, 1),
            })

        n_valid = len(session_stats)
        if n_valid < 3:                                          # cold-start: not enough history
            _write_anchor_cache(None, None, n_valid)
            return None, None, n_valid

        weights  = np.array([s["weight"] for s in session_stats], dtype=np.float32)
        p75s     = np.array([s["p75"]    for s in session_stats], dtype=np.float32)
        p20s     = np.array([s["p20"]    for s in session_stats], dtype=np.float32)
        w_sum    = float(weights.sum())
        if w_sum < 1e-9:
            _write_anchor_cache(None, None, 0)
            return None, None, 0

        hist_p75 = float((weights * p75s).sum() / w_sum)
        hist_p20 = float((weights * p20s).sum() / w_sum)
        print(
            f"[v2] Historical anchor: {n_valid} sessions  "
            f"hist_p75={hist_p75:.3f}  hist_p20={hist_p20:.3f}  "
            f"(youngest {min(s['age_days'] for s in session_stats):.0f}d, "
            f"oldest {max(s['age_days'] for s in session_stats):.0f}d)"
        )
        _write_anchor_cache(hist_p75, hist_p20, n_valid)
        return hist_p75, hist_p20, n_valid

    except Exception as _e:
        print(f"[v2] Historical anchor unavailable: {_e}")
        return None, None, 0


def _calibrate_thresholds(scores_arr: "np.ndarray") -> tuple[float, float]:
    """
    Hybrid grade thresholds: 60% current session quantiles + 40% recency-weighted
    historical anchor from past sessions in LanceDB.

    Cold start (< 3 valid historical sessions): pure current-session quantiles.
    Absolute floors: Strong ≥ 0.50, Mid ≥ 0.28, gap ≥ 0.12.
    """
    if len(scores_arr) < 4:
        return STRONG_THRESH, MID_THRESH

    q75 = float(np.percentile(scores_arr, 75))
    q20 = float(np.percentile(scores_arr, 20))

    hist_p75, hist_p20, n_sess = _historical_anchor()

    if hist_p75 is not None and n_sess >= 3:
        strong = max(0.60 * q75 + 0.40 * hist_p75, 0.50)
        mid    = max(0.60 * q20 + 0.40 * hist_p20, 0.28)
        print(
            f"[v2] Thresholds (calibrated {n_sess} sessions): "
            f"q75={q75:.3f} hist={hist_p75:.3f} → STRONG≥{strong:.2f}  "
            f"q20={q20:.3f} hist={hist_p20:.3f} → MID≥{mid:.2f}"
        )
    else:
        strong = max(q75, 0.50)
        mid    = max(q20, 0.28)
        reason = f"cold start ({n_sess} sessions)" if hist_p75 is None else "quantile only"
        print(
            f"[v2] Thresholds ({reason}): "
            f"q75={q75:.3f}→STRONG≥{strong:.2f}  q20={q20:.3f}→MID≥{mid:.2f}"
        )

    if strong - mid < 0.12:
        mid = max(strong - 0.12, 0.20)

    return strong, mid


def _np2py(v):
    """Convert numpy scalar to Python primitive; pass-through for everything else."""
    return v.item() if hasattr(v, "item") else v


def _sanitize_bd(d: dict) -> dict:
    """Return a copy of breakdown dict with all values cast to Python primitives."""
    return {k: _np2py(v) for k, v in d.items()}


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


# ── EXIF timestamp ─────────────────────────────────────────────────────────────

def _exif_ts(path: str) -> float:
    try:
        import piexif
        exif = piexif.load(path)
        raw = (
            exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
            or exif.get("0th",  {}).get(piexif.ImageIFD.DateTime)
        )
        if raw:
            from datetime import datetime
            return datetime.strptime(raw.decode(), "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    return 0.0


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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
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
    _p = progress or (lambda f, d: None)

    # ── Step 1: Discover images ───────────────────────────────────────────────
    _p(0.01, "Scanning folder…")
    folder = Path(folder_path)
    all_paths  = sorted(
        str(f) for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not all_paths:
        return {"error": "No images found in folder.", "gallery": [], "total": 0}

    # ── Incremental: skip already-graded images when force_rescan=False ───────
    import lance_store as _ls_diag
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
        _p(1.0, f"All {len(cached_rows)} images already graded (use Re-grade to force rescan)")
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
    _blur_disqualified:   set[str] = set()
    _yolo_disqualified:   set[str] = set()
    _yolo_soft_penalized: set[str] = set()
    try:
        from early_exit_gate import run_early_exit_gate
        try:
            from specvlm_pipeline import _cd_brief_implies_empty as _implies_empty
            _run_yolo = _implies_empty()
        except Exception:
            _run_yolo = False

        _p(0.015, "Early-exit gate: Laplacian blur check…")
        _survivors, _blur_disqualified, _yolo_disqualified, _yolo_soft_penalized = (
            run_early_exit_gate(paths, run_yolo=_run_yolo)
        )

        _n_early_fail = len(_blur_disqualified) + len(_yolo_disqualified)
        if _n_early_fail:
            _p(0.025, f"Early-exit: {_n_early_fail} images disqualified → score 0.00")
            print(
                f"[v2] Early-exit gate: {len(_blur_disqualified)} blur-failed, "
                f"{len(_yolo_disqualified)} YOLO-failed → score 0.00, IQA skipped"
            )
    except Exception as _ee_err:
        print(f"[v2] Early-exit gate skipped ({_ee_err})")

    # ── Pre-flush: commit fail records before GPU stages begin ────────────────
    # Persists disqualified images to LanceDB immediately so that if the GPU
    # pipeline aborts mid-run, score=0.00 records are already in the store and
    # won't re-enter the processing queue on the next run.
    _prefail_paths = list(_blur_disqualified | _yolo_disqualified)
    if _prefail_paths:
        _p(0.027, f"Pre-flushing {len(_prefail_paths)} fail records to LanceDB…")
        try:
            import lance_store as _ls_pf
            _ls_pf.upsert_batch([{
                "path":           p,
                "embedding":      [0.0] * 1536,
                "score":          0.00,
                "personal_score": 0.5,
                "grade":          GRADE_WEAK,
                "reasoning_log":  "",
                "breakdown":      {
                    "disqualified": True,
                    "reason": "blur" if p in _blur_disqualified else "yolo",
                },
                "exif_ts":        0.0,
            } for p in _prefail_paths])
            print(f"[v2] Pre-flushed {len(_prefail_paths)} fail records to LanceDB")
        except Exception as _e_pf:
            print(f"[v2] Fail record pre-flush skipped: {_e_pf}")

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
    _enc_reused             = False

    if _enc_singleton is not None:
        _p(0.03, "SigLIP-2 cached — encoding images directly…")
        try:
            embs = _enc_singleton.encode_images(paths, progress=_p)
            if _text_emb_cache:
                _pos_text_embs  = _text_emb_cache["pos"]
                _neg_text_embs  = _text_emb_cache["neg"]
                _aspect_names   = _text_emb_cache["aspect_names"]
                _aspect_pos     = _text_emb_cache["aspect_pos"]
                _aspect_neg     = _text_emb_cache["aspect_neg"]
                _genre_ref_embs  = _text_emb_cache.get("genre_ref_embs")
                _fine_art_anchor = _text_emb_cache.get("fine_art_anchor")
            try:
                from specvlm_pipeline import _CD_BRIEF as _brief_text
                if _brief_text and _brief_text.strip():
                    _p(0.49, "Encoding brief ensemble for semantic alignment…")
                    _brief_variants = _generate_brief_variants(_brief_text)
                    _brief_raw  = _enc_singleton.encode_text(_brief_variants)  # (V, 1536)
                    _prompt_emb = _brief_raw.mean(axis=0).astype(np.float64)
                    _prompt_emb /= (np.linalg.norm(_prompt_emb) + 1e-9)
                    _prompt_emb  = _prompt_emb.astype(np.float32)
                    print(f"[v2] Brief ensemble ({len(_brief_variants)} variants): '{_brief_text[:60]}'")
            except Exception as _e_brief:
                print(f"[v2] Brief embedding skipped: {_e_brief}")
            embed_dim   = 1536
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
            {"device": "auto", "quantize": True},   # 1st: GPU INT8/FP16
            {"device": "cpu",  "quantize": False},  # 2nd: CPU FP16 (slow but correct)
        ]):
            try:
                from siglip2_encoder import SigLIP2Encoder
                from specvlm_pipeline import _POS_PROMPTS, _NEG_PROMPTS, _ASPECT_PROMPTS
                enc  = SigLIP2Encoder(**_kwargs, progress=_p)
                embs = enc.encode_images(paths, progress=_p)   # (N, 1536)

                # Encode aesthetic text references and cache for subsequent runs
                # Augment positive prompts with any PDF reference phrases already ingested
                _pos_prompts_augmented = list(_POS_PROMPTS)
                try:
                    _rag_path = Path(__file__).resolve().parent.parent / "cache" / "rag_concepts.json"
                    if _rag_path.exists():
                        _rag_phrases = json.loads(_rag_path.read_text(encoding="utf-8")).get("phrases", [])
                        if _rag_phrases:
                            _pos_prompts_augmented = _pos_prompts_augmented + _rag_phrases
                            print(f"[v2] RAG: added {len(_rag_phrases)} PDF concept phrases to positive rubric")
                except Exception as _e_rag:
                    print(f"[v2] RAG load skipped: {_e_rag}")
                _p(0.48, "Encoding aesthetic reference prompts…")
                _pos_text_embs = enc.encode_text(_pos_prompts_augmented)    # (P+R, 1536)
                _neg_text_embs = enc.encode_text(_NEG_PROMPTS)              # (Q, 1536)
                _aspect_names  = list(_ASPECT_PROMPTS.keys())
                _aspect_pos    = enc.encode_text(
                    [v[0] for v in _ASPECT_PROMPTS.values()]                # (A, 1536)
                )
                _aspect_neg    = enc.encode_text(
                    [v[1] for v in _ASPECT_PROMPTS.values()]                # (A, 1536)
                )

                _text_emb_cache.update({
                    "pos":          _pos_text_embs,
                    "neg":          _neg_text_embs,
                    "aspect_names": _aspect_names,
                    "aspect_pos":   _aspect_pos,
                    "aspect_neg":   _aspect_neg,
                })

                # Cache "people" concept embedding for empty-brief creative direction
                _PEOPLE_PROMPTS = [
                    "people", "crowds", "pedestrians", "human figure", "faces",
                ]
                _ppl_raw  = enc.encode_text(_PEOPLE_PROMPTS)   # (5, 1536)
                _ppl_mean = _ppl_raw.mean(axis=0)
                _ppl_mean /= (np.linalg.norm(_ppl_mean) + 1e-9)
                try:
                    _cache_dir = Path("cache")
                    _cache_dir.mkdir(parents=True, exist_ok=True)
                    np.save(str(_cache_dir / "people_emb.npy"), _ppl_mean.astype(np.float32))
                    print("[v2] people_emb.npy saved for empty-brief CD gate")
                except Exception as _e_ppl:
                    print(f"[v2] people_emb save skipped: {_e_ppl}")

                # Encode low-contrast genre references for TOPIQ bias correction
                try:
                    _genre_raw   = enc.encode_text(_GENRE_REF_PROMPTS)          # (3, 1536)
                    _gnorms      = np.linalg.norm(_genre_raw, axis=1, keepdims=True)
                    _genre_ref_embs = (_genre_raw / (_gnorms + 1e-9)).astype(np.float32)
                    _text_emb_cache["genre_ref_embs"] = _genre_ref_embs
                    print("[v2] Genre reference embeddings cached for TOPIQ bias correction")
                except Exception as _e_genre_enc:
                    print(f"[v2] Genre ref encoding skipped: {_e_genre_enc}")

                # Encode fine-art pictorialism anchor (3-prompt ensemble, averaged + L2-norm)
                # Used for Vintage Lens Protocol and Soft-Focus Protection Gate.
                try:
                    _fa_raw   = enc.encode_text(_FINE_ART_PROMPTS)              # (3, 1536)
                    _fa_mean  = _fa_raw.mean(axis=0).astype(np.float64)
                    _fa_mean /= (np.linalg.norm(_fa_mean) + 1e-9)
                    _fine_art_anchor = _fa_mean.astype(np.float32)              # (1536,)
                    _text_emb_cache["fine_art_anchor"] = _fine_art_anchor
                    print("[v2] Fine-art anchor encoded and cached (3-prompt pictorialism ensemble)")
                except Exception as _e_fa:
                    print(f"[v2] Fine-art anchor encoding skipped: {_e_fa}")

                # Encode CD brief with prompt ensembling
                try:
                    from specvlm_pipeline import _CD_BRIEF as _brief_text
                    if _brief_text and _brief_text.strip():
                        _p(0.49, "Encoding brief ensemble for semantic alignment…")
                        _brief_variants = _generate_brief_variants(_brief_text)
                        _brief_raw  = enc.encode_text(_brief_variants)           # (V, 1536)
                        _prompt_emb = _brief_raw.mean(axis=0).astype(np.float64)
                        _prompt_emb /= (np.linalg.norm(_prompt_emb) + 1e-9)
                        _prompt_emb  = _prompt_emb.astype(np.float32)
                        print(f"[v2] Brief ensemble ({len(_brief_variants)} variants): '{_brief_text[:60]}'")
                except Exception as _e_brief:
                    print(f"[v2] Brief embedding skipped: {_e_brief}")

                # Street photography multi-probe aesthetic scoring.
                # Encodes once; per-image scores are pure dot products on existing embs.
                try:
                    _sp_raw  = enc.encode_text(_STREET_POS_PROBES)   # (P, 1536)
                    _sn_raw  = enc.encode_text(_STREET_NEG_PROBES)   # (Q, 1536)
                    _sp_nrm  = np.linalg.norm(_sp_raw, axis=1, keepdims=True)
                    _sn_nrm  = np.linalg.norm(_sn_raw, axis=1, keepdims=True)
                    _sp_embs = (_sp_raw / (_sp_nrm + 1e-9)).astype(np.float32)
                    _sn_embs = (_sn_raw / (_sn_nrm + 1e-9)).astype(np.float32)
                    _street_raw = (ethe mbs @ _sp_embs.T).mean(axis=1) - (embs @ _sn_embs.T).mean(axis=1)
                    _s_min, _s_max = float(_street_raw.min()), float(_street_raw.max())
                    if _s_max > _s_min:
                        street_aesthetic_scores = ((_street_raw - _s_min) / (_s_max - _s_min)).astype(np.float32)
                    else:
                        street_aesthetic_scores = np.full(n, 0.5, dtype=np.float32)
                    print(f"[v2] Street aesthetic scores: min={street_aesthetic_scores.min():.3f}  "
                          f"max={street_aesthetic_scores.max():.3f}  mean={street_aesthetic_scores.mean():.3f}  "
                          f"({len(_STREET_POS_PROBES)}pos/{len(_STREET_NEG_PROBES)}neg probes)")
                except Exception as _e_sp:
                    print(f"[v2] Street probe scoring skipped: {_e_sp}")
                    street_aesthetic_scores = np.full(n, 0.5, dtype=np.float32)

                _enc_singleton = enc   # keep in VRAM — evicted by release_grading_models()
                embed_dim = 1536
                siglip_ok = True
                _tag = "GPU" if _kwargs["device"] == "auto" else "CPU fallback"
                _p(0.50, "SigLIP-2 done — cached as singleton…")
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

        # SigLIP-2 (1536-d) is required — all legacy encoders removed in Frontier 2026.
        if embed_dim != 1536:
            raise RuntimeError(
                f"SigLIP-2 failed to load on both GPU and CPU.\n"
                f"Reason: {_siglip_last_err}"
            )

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
    _arch_cache_path      = Path("cache") / "archetype_embs.npy"
    _arch_hash_path       = Path("cache") / "archetype_embs.hash"
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
        print(f"[v2] Archetype embeddings: loaded from cache {archetype_embs.shape}")
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
            "[v2] Archetype embeddings cache missing and _enc_singleton is None. "
            "The encoder must be loaded before this stage. "
            "Delete cache/archetype_embs.npy and re-grade to regenerate."
        )

    # Flush caching allocator — singleton weights remain resident in VRAM
    _vram_clear()

    # ── Step 3: Duplicate detection ───────────────────────────────────────────
    _p(0.50, "Detecting duplicates…")
    cluster_ids:     list[int] = [-1] * n
    sim_flags:       list[str] = [""] * n
    to_rate_indices: list[int] = list(range(n))
    _comp_eligible:  set[str]  = set(paths)   # default: all paths eligible for composition

    if siglip_ok and n >= 2:
        try:
            from collections import defaultdict as _dd
            norms  = np.linalg.norm(embs, axis=1, keepdims=True)
            normed = embs / (norms + 1e-9)
            sims   = normed @ normed.T

            SIM_THRESH = 0.96   # true burst duplicates only (same frame ±ms)

            parent = list(range(n))
            def _find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            # Vectorized: find all above-threshold pairs in one numpy call
            dup_i, dup_j = np.where(np.triu(sims > SIM_THRESH, k=1))
            for i, j in zip(dup_i.tolist(), dup_j.tolist()):
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

    # Stamp all early-exit disqualified photos (score 0.00, skip IQA)
    _all_disqualified = _blur_disqualified | _yolo_disqualified
    for i, p in enumerate(paths):
        if p in _all_disqualified:
            scores[i] = 0.00

    # Exclude disqualified images from IQA scoring
    to_rate_indices = [i for i in to_rate_indices if paths[i] not in _all_disqualified]
    paths_to_rate = [paths[i] for i in to_rate_indices]

    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _props = _torch.cuda.get_device_properties(0)
            _free  = (_props.total_memory - _torch.cuda.memory_reserved(0)) / 1e9
            print(f"[v2] VRAM before IQA heads: {_free:.2f} GB free / {_props.total_memory/1e9:.2f} GB total")
        del _torch
    except Exception:
        pass

    # ── Step 4a: Vision grading (Qwen2.5-VL-3B primary · SpecVLM CLIP fallback) ──
    # Primary:  Qwen2.5-VL looks at each image and outputs absolute aspect scores
    #           (0-100) directly from vision — not cosine similarity against text.
    #           RAG concept phrases from uploaded PDFs are injected into the prompt.
    # Fallback: SpecVLMPipeline CLIP cosine similarity (instant, no extra model).
    # scan_mode always uses SpecVLM CLIP for speed.
    _p(0.51, "Vision grading…")
    vlm_scores_rated  = np.full(len(paths_to_rate), 0.5, dtype=np.float32)
    comp_scores_rated = np.full(len(paths_to_rate), 0.5, dtype=np.float32)
    _raw_comp_by_path: dict[str, float] = {}
    _vlm_ran = False   # True when Qwen scored — skips SpecVLM CLIP fallback

    if not scan_mode:
        try:
            from qwen_vlm_grader import QwenVLMGrader
            if True:
                try:
                    from pdf_rag import load_concepts as _load_rag
                    _rag_phrases = _load_rag()
                except Exception:
                    _rag_phrases = []

                if _qwen_singleton is None:
                    _p(0.51, "Loading Vision Engine (first run downloads ~6 GB)…")
                    # Evict SigLIP-2 before Qwen loads — embeddings already in NumPy.
                    if _enc_singleton is not None:
                        try:
                            _enc_singleton.unload()
                        except Exception:
                            pass
                        _enc_singleton = None
                    _vram_clear()
                    _qwen_singleton = QwenVLMGrader(progress=_p)
                    print("[v2] Qwen singleton created — will reuse across runs")
                else:
                    _p(0.51, "Vision Engine ready…")
                    print("[v2] Reusing Qwen singleton")

                _qwen_results = _qwen_singleton.grade_images_scored(
                    paths_to_rate,
                    mode        = preset,
                    rag_phrases = _rag_phrases,
                    progress    = _p,
                )
                # Do NOT unload — keep singleton warm for next run.
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

                _vlm_ran = True
                _p(0.65, "Qwen2.5-VL grading done — running IQA heads…")
                print(
                    f"[v2] Qwen2.5-VL scores: min={vlm_scores_rated.min():.3f}  "
                    f"max={vlm_scores_rated.max():.3f}  mean={vlm_scores_rated.mean():.3f}"
                    + (f"  rag={len(_rag_phrases)} phrases" if _rag_phrases else "")
                )
            else:
                print("[v2] Qwen2.5-VL weights not cached — using SpecVLM CLIP scoring")
        except Exception as _e_qwen:
            print(f"[v2] Qwen2.5-VL grading failed ({_e_qwen}) — using SpecVLM CLIP")

    if not _vlm_ran:
        # ── SpecVLM CLIP scoring (scan mode or Qwen2.5-VL unavailable) ───────
        try:
            from specvlm_pipeline import SpecVLMPipeline
            pipeline        = SpecVLMPipeline()
            specvlm_results = pipeline.grade_images(
                paths_to_rate,
                progress        = _p,
                scan_mode       = scan_mode,
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

            _p(0.55, "SpecVLM done — running IQA heads…")
            print(
                f"[v2] SpecVLM scores: min={vlm_scores_rated.min():.3f}  "
                f"max={vlm_scores_rated.max():.3f}  mean={vlm_scores_rated.mean():.3f}"
            )

        except Exception as e_clip:
            import traceback as _tb_clip
            print(f"[v2] SpecVLM scoring FATAL — full traceback:")
            _tb_clip.print_exc()
            print("\n--- CRASH LOCAL VARIABLES ---")
            for key, value in locals().items():
                if key not in ['self', 'encoder_model']:
                    print(f"{key}: {value}")
            print("-----------------------------\n")
            raise

    _vram_clear()  # Free grader VRAM before IQA heads load

    # Pre-compute luminance stats — shared by composition analysis, ChiaroscuroHead,
    # and Vintage Lens Protocol in Step 4d.
    _p(0.555, "Computing luminance stats…")

    def _lum_stats(path: str):
        try:
            from PIL import Image as _PILI
            with _PILI.open(path) as _raw:
                img = _raw.convert("RGB")
            img.thumbnail((128, 128), _PILI.LANCZOS)   # 10-100× faster; lum stats are invariant to scale
            _arr = np.array(img, dtype=np.float32)
            _Y   = 0.299 * _arr[:, :, 0] + 0.587 * _arr[:, :, 1] + 0.114 * _arr[:, :, 2]
            return float(_Y.mean()), float(_Y.std())
        except Exception:
            return 128.0, 60.0   # neutral defaults — no VLP trigger

    import os as _os_lum
    from concurrent.futures import ThreadPoolExecutor as _TPELUM
    _lum_workers = min(_os_lum.cpu_count() or 8, max(len(paths_to_rate), 1))
    with _TPELUM(max_workers=_lum_workers) as _lpool:
        lum_stats_rated = list(_lpool.map(_lum_stats, paths_to_rate))

    # ── Step 4b: Vision IQA Head (UniQA unified backbone) ───────────────────────
    # scan_mode bypasses IQA — Scan uses composition scores (already set in Step 4a).
    # Full grading runs:
    #   1. run_composition_analysis (Depth → Seg → Chiaroscuro) — within run_vision_heads
    #   2. UniQAHead with YOLO11s-seg routing (empty-scene / layered-frame / standard)
    composition_overrides: dict[str, float] = {}
    chiaroscuro_flags:     dict[str, bool]  = {}
    person_detected_dict:  dict[str, bool]  = {}

    if scan_mode:
        _p(0.84, "Scan mode — IQA heads skipped, using SpecVLM scores…")
        print(f"[v2] Scan mode: IQA skipped, {len(paths_to_rate)} photos at CLIP speed")
        tech_scores_rated      = vlm_scores_rated.copy()
        aesthetic_scores_rated = vlm_scores_rated.copy()
    else:
        _p(0.56, f"IQA heads — scoring {len(paths_to_rate)} images…")
        try:
            from vision_grading_heads import run_vision_heads

            iqa_embs  = embs[np.array(to_rate_indices)]   # (M, 1536)
            _vlm_bds  = [per_photo_breakdowns[idx] for idx in to_rate_indices]

            iqa_out = run_vision_heads(
                image_paths         = paths_to_rate,
                image_embeddings    = iqa_embs,
                prompt_embedding    = _prompt_emb,
                clip_scores         = vlm_scores_rated,
                genre_ref_embs      = _genre_ref_embs,
                lum_stats           = lum_stats_rated,
                progress            = _p,
                comp_eligible_paths = _comp_eligible,
                vlm_breakdowns      = _vlm_bds,
            )

            tech_scores_rated        = iqa_out["quality"]                      # (M,) UniQA technical
            aesthetic_scores_rated   = vlm_scores_rated                        # (M,) VLM aesthetic (Step 4a)
            iqa_breakdowns           = iqa_out["breakdowns"]                   # list[dict]
            composition_overrides    = iqa_out.get("composition_overrides",  {})
            chiaroscuro_flags        = iqa_out.get("chiaroscuro_flags",      {})
            person_detected_dict     = iqa_out.get("person_detected",        {})
            _framing_obstruction_dict = iqa_out.get("framing_obstruction", {})
            _subject_bboxes_dict     = iqa_out.get("subject_bboxes",       {})

            for local_i, idx in enumerate(to_rate_indices):
                per_photo_breakdowns[idx].update(iqa_breakdowns[local_i])
                # Apply over-the-shoulder portrait composition override
                _opath = paths[idx]
                if _opath in composition_overrides:
                    per_photo_breakdowns[idx]["Composition"] = composition_overrides[_opath]

            _p(0.84, "IQA heads done — releasing singletons…")
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
            import traceback as _tb_iqa
            print(f"[v2] IQA heads FATAL — full traceback:")
            _tb_iqa.print_exc()
            print("\n--- CRASH LOCAL VARIABLES ---")
            for key, value in locals().items():
                if key not in ['self', 'encoder_model']:  # Skip dumping massive model objects
                    print(f"{key}: {value}")
            print("-----------------------------\n")
            raise

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
        _fa_sims_rated = _fine_art_sims_all[np.array(to_rate_indices)]
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
    # similarity ≥ 0.75 → enforce overall_score = max(score, 0.65).
    # This guarantees compositionally elite or fine-art-aligned photos can never drop
    # below the Strong bucket threshold due to IQA penalties.
    _AES_VLP_THRESHOLD    = 0.556   # TOPIQ NR threshold for VLP trigger
    _AES_ANCHOR_THRESHOLD = 0.72    # TOPIQ NR threshold for Anchor Floor (raised from 0.611 — old value was calibrated for broken UniQA that always returned 0.5)
    _ANCHOR_FLOOR         = 0.65
    _FA_ANCHOR_THRESHOLD  = 0.75                 # normalised fine-art sim threshold

    _p(0.86, "Fusing scores (vectorised: Archetype Projection + VLP + smooth gates)…")

    _ARCHETYPE_TEMP = 0.15

    M = len(to_rate_indices)

    # ── Pre-extract per-image arrays ──────────────────────────────────────────
    def _bd_arr(key: str, default: float = 0.5) -> np.ndarray:
        return np.array(
            [float(per_photo_breakdowns[idx].get(key, default)) for idx in to_rate_indices],
            dtype=np.float32,
        )

    arr_t     = tech_scores_rated                           # (M,) raw IQA
    arr_a     = aesthetic_scores_rated                      # (M,)
    arr_fa    = fine_art_scores_rated                       # (M,)
    arr_lum   = np.array([ls[0] for ls in lum_stats_rated], dtype=np.float32)
    arr_std   = np.array([ls[1] for ls in lum_stats_rated], dtype=np.float32)
    arr_comp  = _bd_arr("Composition")
    arr_light = _bd_arr("Lighting")
    arr_hc    = _bd_arr("Human/Culture")
    arr_narr  = _bd_arr("Narrative")
    arr_tech  = np.array(
        [float(per_photo_breakdowns[idx].get("Technical", float(arr_t[li])))
         for li, idx in enumerate(to_rate_indices)],
        dtype=np.float32,
    )
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
    arr_street = _sa[np.array(to_rate_indices)]                                  # (M,)

    # ── Batch archetype projection: one matrix multiply for all M images ──────
    rated_embs  = embs[np.array(to_rate_indices)].astype(np.float32)          # (M, 1536)
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

    # ── Per-archetype formula scores (all vectorized) ─────────────────────────
    fused_geo     = arr_comp * 0.40 + arr_light * 0.30 + arr_a * 0.30
    # Night: removed artificial comp-floor (was max(comp, 0.70)) and over-generous +0.075
    # constant. Added arr_a (VLM holistic score) so mediocre night shots can't ride the
    # comp floor to Strong. Constant reduced to +0.05 (ambient light bonus only).
    fused_night   = arr_hc * 0.30 + arr_comp * 0.30 + arr_a * 0.20 + arr_t * 0.15 + 0.05
    # Layered + Max_doc: added arr_a (Qwen holistic verdict) so direct VLM quality
    # judgment tempers over-reliance on individual aspects that Qwen may rate loosely.
    fused_layered = arr_comp * 0.35 + arr_hc * 0.35 + arr_narr_eff * 0.15 + arr_a * 0.15
    fused_messy   = np.maximum(0.0, arr_t * 0.35 + arr_a * 0.65 - 0.08)
    fused_max_doc = arr_hc * 0.35 + arr_narr_eff * 0.30 + arr_a * 0.20 + arr_t * 0.15

    fused = (
        w_geo     * fused_geo     +
        w_night   * fused_night   +
        w_layered * fused_layered +
        w_messy   * fused_messy   +
        w_max_doc * fused_max_doc
    )                                                                           # (M,) blend

    # Street aesthetic genre-fit modifier: ±0.06 max based on 304-probe cosine scoring.
    # Centered at 0.50 so neutral images receive no adjustment.
    fused = np.clip(fused + 0.12 * (arr_street - 0.50), 0.0, 1.0)

    # ── Post-fusion gates (vectorized) ────────────────────────────────────────

    # 1. Vintage Lens Protocol — blend fine-art similarity for dark/pictorialist work
    vlp_mask = arr_is_chiaroscuro | (((arr_lum < 80.0) | (arr_std < 35.0)) & (arr_a >= _AES_VLP_THRESHOLD))
    fused    = np.where(vlp_mask, fused * 0.75 + arr_fa * 0.25, fused)

    # 2. YOLO soft penalty for dark-scene silhouettes (waived for chiaroscuro)
    yolo_pen_mask = arr_yolo_soft & ~arr_is_chiaroscuro
    fused = np.where(yolo_pen_mask, np.maximum(0.0, fused - 0.15), fused)

    # 3. Anchor Floor — elite photos cannot fall below 0.65
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
    comp_exempt = np.isin(dom_arch, [0, 1, 4])

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

    # 7. Archetype strong floors (absolute last word on minimums)
    fused = np.where(night_gate & (arr_tech >= 0.50),
                     np.maximum(fused, 0.68), fused)
    fused = np.where((dom_arch == 0) & (arr_tech >= 0.50) & (arr_comp >= 0.55) & (arr_a >= 0.55),
                     np.maximum(fused, 0.66), fused)
    fused = np.where(max_gate & (arr_tech >= 0.55) & (arr_light >= 0.35),
                     np.maximum(fused, 0.64), fused)
    geo_assist = (w_geo >= 0.18) & (dom_arch != 3) & ~night_gate & ~max_gate & ~narr_clutter & (arr_t >= 0.38)
    fused = np.where(geo_assist, np.maximum(fused, 0.62), fused)

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
        print(f"[v2] Max-doc floor: {_max_gate_count} → 0.64")
    if _geo_floor_count:
        print(f"[v2] Geo floor: {_geo_floor_count} → 0.66")
    if _night_floor_count:
        print(f"[v2] Night floor: {_night_floor_count} → 0.68")
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
    _DIAG_TARGETS = {"TPE26-10.jpg", "TPE26-102.jpg"}
    for _li, _idx in enumerate(to_rate_indices):
        if Path(paths[_idx]).name in _DIAG_TARGETS:
            print(
                f"\n[DIAG 4d] {Path(paths[_idx]).name}  "
                f"fused={fused[_li]:.4f}  dom={dom_arch[_li]}  "
                f"geo={w_geo[_li]:.2f} night={w_night[_li]:.2f} "
                f"layer={w_layered[_li]:.2f} messy={w_messy[_li]:.2f} max={w_max_doc[_li]:.2f}  "
                f"vlp={bool(vlp_mask[_li])}  anchor={bool(anchor_mask[_li])}  "
                f"clutter={_clutter_density[_li]:.3f}  r2b={bool(route2b_mask[_li])}\n"
            )

    scores_arr = np.array(scores, dtype=np.float32)
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
    if not scan_mode:
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
                _p(0.865, f"VLM fast-scan ({len(paths_to_rate)} images) — {_active_scan_model}…")

                print(f"[v2] Step 4e: pre-loading VLM models…")
                warmup_vlm_models()
                print(f"[v2] Step 4e: models resident — starting {_active_scan_model} batch")

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

                        per_photo_breakdowns[idx]["Composition"]  = round(_comp_fit, 3)
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
                    print(
                        f"[v2] Qwen VLM fast-scan: {_vlm4e_ok}/{len(paths_to_rate)} scored — "
                        f"min={scores_arr.min():.3f}  max={scores_arr.max():.3f}  "
                        f"mean={scores_arr.mean():.3f}"
                    )
                    # Soft batch normalisation — if VLM scores cluster tightly
                    # (range < 0.20), stretch them toward [0.18, 0.88] so that
                    # relative differences produce visible grade separation.
                    # Only applied when ≥4 images were scored; not applied when
                    # the range is already healthy (≥ 0.20) to avoid distorting
                    # genuinely discriminating scores.
                    _scored_idx = [i for i in range(n) if scores[i] > 0.01]
                    if len(_scored_idx) >= 4:
                        _raw = np.array([scores[i] for i in _scored_idx], dtype=np.float32)
                        _rng = float(_raw.max() - _raw.min())
                        if 0.005 < _rng < 0.20:
                            _stretched = 0.18 + (_raw - _raw.min()) / _rng * 0.70
                            for _j, _si in enumerate(_scored_idx):
                                scores[_si] = float(np.clip(_stretched[_j], 0.0, 1.0))
                            scores_arr = np.array(scores, dtype=np.float32)
                            print(
                                f"[v2] Score stretch applied: raw range {_rng:.3f} → "
                                f"[{scores_arr[_scored_idx].min():.3f}, "
                                f"{scores_arr[_scored_idx].max():.3f}]"
                            )
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
        _yolo_fb = _subject_bboxes_dict.get(paths[_sidx], [])
        if not _yolo_fb:
            continue
        _yb = _yolo_fb[0]  # [x1n, y1n, x2n, y2n] normalised [0,1]
        try:
            from PIL import Image as _PILI_DIM
            with _PILI_DIM.open(paths[_sidx]) as _dim_img:
                _iw, _ih = _dim_img.size
            per_photo_breakdowns[_sidx]["vlm_bboxes"] = [{
                "label": "anchor_subject",
                "bbox_2d": [
                    int(_yb[0] * _iw), int(_yb[1] * _ih),
                    int(_yb[2] * _iw), int(_yb[3] * _ih),
                ]
            }]
            _bbox_synth_count += 1
        except Exception:
            pass
    if _bbox_synth_count:
        print(f"[v2] Anchor-subject bbox synthesised from YOLO for {_bbox_synth_count} images")

    # ── Step 5: PersonalHead adjustment ──────────────────────────────────────
    _p(0.87, "Applying personal preference…")
    pers         = np.full(n, 0.5, dtype=np.float32)
    final_scores = scores_arr.copy()  # copy so Soft-Focus gate doesn't mutate scores_arr
    _ph_weights  = Path("cache/personal_head.pt")
    if _ph_weights.exists():
        print("[v2] PersonalHead weights found — blending 80/20")
        try:
            import personal_head as ph
            pers         = ph.score(embs)
            final_scores = 0.80 * scores_arr + 0.20 * pers
        except Exception as _e:
            import traceback
            print(f"[v2] PersonalHead blend failed: {_e}")
            traceback.print_exc()
            print("\n--- CRASH LOCAL VARIABLES ---")
            for key, value in locals().items():
                if key not in ['self', 'encoder_model']:  # Skip dumping massive model objects
                    print(f"{key}: {value}")
            print("-----------------------------\n")
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
                f"(fine_art_sim > 0.68, base_score ≥ 0.42)"
            )

    # ── Step 5b: Duplicate sim-flag assignment based on final_scores ──────────
    _p(0.88, "Flagging duplicates…")
    try:
        from collections import defaultdict as _dd2
        groups2: dict = _dd2(list)
        for i, cid in enumerate(cluster_ids):
            if cid >= 0:
                groups2[cid].append(i)
        for members in groups2.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda i: float(final_scores[i]), reverse=True)
            best_fn = Path(paths[members[0]]).name
            best_sc = float(final_scores[members[0]])
            for rank, idx in enumerate(members):
                if rank == 0:
                    sim_flags[idx] = f"★ Best of {len(members)} similar shots (score {best_sc:.2f})"
                else:
                    diff = best_sc - float(final_scores[idx])
                    sim_flags[idx] = (
                        f"\U0001f501 Duplicate — {best_fn} is better: higher overall score (+{diff:.2f})"
                    )
    except Exception as e:
        import traceback
        print(f"[v2] Sim-flag assignment failed: {e}")
        traceback.print_exc()
        print("\n--- CRASH LOCAL VARIABLES ---")
        for key, value in locals().items():
            if key not in ['self', 'encoder_model']:  # Skip dumping massive model objects
                print(f"{key}: {value}")
        print("-----------------------------\n")

    # ── Step 6: Absolute grade thresholds ────────────────────────────────────
    # Strong ≥ 0.60  |  Mid 0.41–0.59  |  Weak ≤ 0.40
    final_scores = np.clip(np.nan_to_num(final_scores, nan=0.15), 0.10, 1.0)
    final_scores = np.round(final_scores, 2)

    _p(0.89, "Applying grade thresholds…")
    print(
        f"[v2] final scores — min={final_scores.min():.2f}  "
        f"max={final_scores.max():.2f}  mean={final_scores.mean():.2f}  "
        f"median={float(np.median(final_scores)):.2f}"
    )
    # Calibrate on pre-gate Qwen/CLIP scores for rated images only (excludes
    # disqualified 0.00s and Soft-Focus inflation that would skew the regime detection)
    _rated_idx = np.array(to_rate_indices, dtype=np.int32)
    _s_thresh, _m_thresh = _calibrate_thresholds(scores_arr[_rated_idx] if len(_rated_idx) >= 4 else scores_arr)
    print(f"[v2] Thresholds — Weak < {_m_thresh:.2f}  |  Mid {_m_thresh:.2f}–{_s_thresh - 0.01:.2f}  |  Strong ≥ {_s_thresh:.2f}")

    grades = []
    for i, s in enumerate(final_scores):
        if s >= _s_thresh:
            g = GRADE_STRONG
        elif s >= _m_thresh:
            g = GRADE_MID
        else:
            g = GRADE_WEAK
        grades.append(g)
        print(f"[v2]   {Path(paths[i]).name}: {s:.2f} → {g}")

    # ── Step 7: EXIF + LanceDB ────────────────────────────────────────────────
    _p(0.90, "Reading EXIF…")
    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=min(16, len(paths) or 1)) as _pool:
        timestamps = list(_pool.map(_exif_ts, paths))

    _p(0.92, "Writing to LanceDB (bulk upsert)…")
    lance_ok = False
    try:
        import lance_store as ls
        import traceback as _tb_lance
        print(f"[v2] LanceDB WRITE START — {n} records → {ls._DB_DIR}")
        # Build all records in memory first, then a single vectorised upsert.
        # Per-photo breakdown includes all CLIP aspect dimensions, not just
        # the high-level aesthetic/personal summary.
        lance_records: list[dict] = []
        for i in range(n):
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
                "reasoning_log":  "",          # LLM layer removed; field kept for schema compat
                "breakdown":      bd,
                "exif_ts":        timestamps[i],
            })
        ls.upsert_batch(lance_records)
        ls.compact_after_write()
        ls.close_table()
        lance_ok = True
        print(f"[v2] LanceDB WRITE OK — {len(lance_records)} records committed")
    except Exception as _e_lance:
        import traceback as _tb_lance
        print(f"[v2] !!! LanceDB WRITE FAILED: {_e_lance}")
        _tb_lance.print_exc()
        lance_ok = False

    # ── Step 8: Gallery response ──────────────────────────────────────────────
    _p(0.94, "Building gallery…")
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
            "stars":           0,
            "reject":          cluster_ids[i] >= 0 and not sim_flags[i].startswith("★"),
            "sim_flag":        sim_flags[i],
            "cluster_id":      int(cluster_ids[i]),
        })

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
    try:
        import time as _cat_time
        _cat_dir  = Path(__file__).resolve().parent.parent / "cache"
        _cat_dir.mkdir(parents=True, exist_ok=True)
        _cat_path = _cat_dir / "catalog.json"
        _cat_photos  = [{k: v for k, v in g.items() if k != "embedding"} for g in gallery]
        _cat_folders = list(dict.fromkeys(str(Path(g["path"]).parent) for g in gallery))
        _cat_payload = json.dumps({
            "photos":   _cat_photos,
            "folders":  _cat_folders,
            "saved_at": _cat_time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, default=_np2py)
        _cat_tmp = _cat_path.with_suffix(".json.tmp")
        _cat_tmp.write_text(_cat_payload, encoding="utf-8")
        _cat_tmp.replace(_cat_path)
        print(f"[v2] catalog.json → {len(_cat_photos)} photos (atomic write)")
    except Exception as _e_cat:
        import traceback
        print(f"[v2] catalog.json write skipped: {_e_cat}")
        traceback.print_exc()
        print("\n--- CRASH LOCAL VARIABLES ---")
        for key, value in locals().items():
            if key not in ['self', 'encoder_model']:  # Skip dumping massive model objects
                print(f"{key}: {value}")
        print("-----------------------------\n")

    # ── Step 9: NSGA-III multi-objective sequencing ───────────────────────────
    _p(0.96, "Running NSGA-III (strict literal constraints)…")
    mogco_seq:   list[dict] = []
    mogco_error: str        = ""
    if siglip_ok and lance_ok:
        try:
            from nsga3_sequencer import run_nsga3_sequence_with_vlm, SequencerConstraintError

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

            selected = run_nsga3_sequence_with_vlm(
                seq_candidates,
                target     = mogco_target,
                progress   = _p,
                brief      = _seq_brief,
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

        except SequencerConstraintError as e:
            mogco_error = str(e)
            print(f"[v2] NSGA-III constraint error: {e}")
        except Exception as e:
            import traceback
            print(f"[v2] NSGA-III sequencing failed: {e}")
            traceback.print_exc()
            print("\n--- CRASH LOCAL VARIABLES ---")
            for key, value in locals().items():
                if key not in ['self', 'encoder_model']:  # Skip dumping massive model objects
                    print(f"{key}: {value}")
            print("-----------------------------\n")

    _p(1.0, "Done")

    all_grades = [g["grade"] for g in gallery]
    strong = sum(1 for g in all_grades if g == GRADE_STRONG)
    mid    = sum(1 for g in all_grades if g == GRADE_MID)
    weak   = sum(1 for g in all_grades if g == GRADE_WEAK)
    print(f"[v2] SUMMARY: {len(gallery)} photos → Strong={strong}  Mid={mid}  Weak={weak}  (new={n}  cached={len(cached_rows)})")

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
        description="Street Story Curator — vision grading pipeline",
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
        "--force_rescan", action="store_true", default=True,
        help="Re-grade even if images are already in LanceDB (default: True)",
    )
    _args = _parser.parse_args()

    # Ensure src/ is on the path when called from project root
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 62)
    print(f"  Street Story Curator — pipeline test run")
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
