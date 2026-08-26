"""Export routes — moved verbatim from server_impl.py (Milestone 4 split).

Decorators retargeted app -> router; every bare name that used to live in
server_impl resolves lazily through the module __getattr__ below (PEP 562),
so request-time access always sees the fully-initialised app without
circular imports. FastAPI names are imported eagerly because decorator-time
evaluation (parameter defaults like Query(...)) runs at import.
"""
from fastapi import (
    APIRouter, Body, Depends, File, Form, HTTPException, Query, Request,
    Response, UploadFile,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, PlainTextResponse,
    StreamingResponse,
)
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator, validator, model_validator

from server_impl import (  # shared state & helpers
    Path, RECENTLY_GENERATED, _DATA_DIR, _IMAGE_EXTS, _ensure_worker, _get_editorial_fns, _safe_dir_path, _safe_image_path, _worker_proc, analyzer, asyncio, gpu_lock, os, sys,
)
from routers.library import GradeRequest

router = APIRouter()


def __getattr__(name):
    # Eager bindings above cover every static reference; this only serves
    # dynamic accesses (e.g. late-bound state added after the split).
    import server_impl as _si
    return getattr(_si, name)


@router.post("/api/clear_history")
async def clear_generation_history():
    global RECENTLY_GENERATED
    RECENTLY_GENERATED.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Niche recommendation
# ---------------------------------------------------------------------------

@router.post("/api/recommend")
async def analyze_niche(payload: dict):
    import math

    results = payload.get("photos", [])
    if not results:
        return {"preset": "Classic Street", "confidence": 0, "reason": "No images provided."}

    # ── Extract named breakdown values by label (not positional index) ────────
    LABEL_MAP = {
        "Technical":"tech","News Sharpness":"tech","Cleanliness":"tech","Execution":"tech",
        "Detail Retention":"tech","Exposure":"tech","Sharpness & Detail":"tech",
        "Composition":"comp","Framing":"comp","Context":"comp","Geometry & Balance":"comp",
        "Negative Space":"comp","Framing Instinct":"comp","Layered Depth":"comp",
        "Lighting":"light","Atmosphere":"light","Natural Light":"light","Mood & Tone":"light",
        "Tonal Purity":"light","Contrast Purity":"light","Available Light":"light",
        "Natural Light Quality":"light",
        "Decisive Moment":"auth","Cultural Depth":"auth","Journalistic Integrity":"auth",
        "Narrative Suggestion":"auth","Conceptual Weight":"auth","Reduction":"auth",
        "Authenticity":"auth","Immediacy":"auth","Environmental Truth":"auth",
        "Subject Isolation":"human","Sense of Place":"human","Human Impact":"human",
        "Character Presence":"human","Emotional Resonance":"human","Scale Element":"human",
        "Human/Culture":"human","Presence":"human","Scale & Life":"human",
    }

    totals = {"tech":0.0,"comp":0.0,"light":0.0,"auth":0.0,"human":0.0}
    counts = {"tech":0,"comp":0,"light":0,"auth":0,"human":0}
    scores_all, faces_list = [], []
    n_items = 0

    for item in results:
        d = item if isinstance(item, dict) else (item[1] if isinstance(item,(list,tuple)) else {})
        b = d.get("breakdown", {})
        for label, val in b.items():
            key = LABEL_MAP.get(label)
            if key:
                totals[key] += float(val)
                counts[key] += 1
        scores_all.append(float(d.get("score", 0.0)))
        faces_list.append(int(d.get("faces", 0)))
        n_items += 1

    if n_items == 0:
        return {"preset": "Classic Street", "confidence": 0, "reason": "No scoreable images."}

    avg = {k: (totals[k]/counts[k] if counts[k] else 0.5) for k in totals}
    t, c, l, a, h = avg["tech"], avg["comp"], avg["light"], avg["auth"], avg["human"]

    # ── Derived signals ───────────────────────────────────────────────────────
    valid_scores  = [s for s in scores_all if s > 0]
    avg_score     = sum(valid_scores) / len(valid_scores) if valid_scores else 0.5
    score_std     = math.sqrt(sum((x-avg_score)**2 for x in valid_scores)/len(valid_scores)) if valid_scores else 0.0
    avg_faces     = sum(faces_list) / n_items
    frac_with_faces = sum(1 for f in faces_list if f > 0) / n_items
    strong_frac   = sum(1 for s in scores_all if s > 0.65) / n_items
    weak_frac     = sum(1 for s in scores_all if s < 0.45) / n_items

    # Interaction terms that are the actual discriminant features
    tech_x_auth   = t * a            # high = documentary/press; low tech only = snapshot
    comp_x_light  = c * l            # high = landscape/cinematic
    human_x_auth  = h * a            # high = travel/humanist; low = landscape/minimalist
    low_tech_flag = max(0.0, 0.45 - t)   # how far below the "deliberate" threshold
    no_people     = max(0.0, 0.30 - h)   # how strongly people are absent
    people_heavy  = max(0.0, h - 0.55)   # how strongly people dominate

    # ── Discriminant scoring ──────────────────────────────────────────────────
    # Each archetype is scored on the cross-dimension patterns that uniquely
    # identify it, with penalties for patterns that contradict it.
    def clamp(x): return max(0.0, min(1.0, x))

    raw = {}

    # SNAPSHOT: raw immediacy signature — low tech OR high variance, high auth
    # Key: (low_tech OR high_std) AND high_auth
    snapshot_tech_signal = clamp(low_tech_flag * 2.2)
    snapshot_var_signal  = clamp(score_std * 4.0)
    snapshot_trigger     = clamp(max(snapshot_tech_signal, snapshot_var_signal * 0.8))
    raw["Snapshot / Point-and-Shoot"] = (
        0.40 * snapshot_trigger * a +
        0.25 * clamp(weak_frac * 1.5) +
        0.20 * clamp(frac_with_faces) +
        0.15 * clamp(1.0 - strong_frac * 1.5)
    ) - clamp((t - 0.60) * 2.0) * 0.35    # penalise if actually technically sharp

    # STREET - EDITORIAL: balanced auth + comp + human, all above floor
    # Use min-of-three so any below-threshold dimension suppresses the score without
    # collapsing to near-zero when two dimensions are just above threshold.
    street_balance = min(clamp(a - 0.30), clamp(c - 0.30), clamp(h - 0.30))
    raw["Classic Street"] = (
        0.40 * clamp(street_balance * 5.0) +   # needs all three
        0.25 * clamp(a) +
        0.20 * clamp(h) +
        0.15 * clamp(c)
    ) - clamp(low_tech_flag * 1.5) * 0.20      # slight penalty for very low tech

    # WORLD PRESS DOC: tech + auth is the signature, human context required
    # Key: high tech AND high auth, people present
    raw["Photojournalism"] = (
        0.40 * clamp(tech_x_auth * 2.0) +
        0.30 * clamp((t - 0.50) * 3.0) +        # tech must be genuinely high
        0.20 * clamp(h) +
        0.10 * clamp(a)
    ) - clamp((0.50 - t) * 3.0) * 0.40          # hard penalty if tech is low

    # TRAVEL EDITOR: auth + human together, place/environment matters
    # Key: both auth AND human high (cultural immersion pattern)
    raw["Travel Editor"] = (
        0.45 * clamp(human_x_auth * 2.2) +
        0.25 * clamp(l) +
        0.20 * clamp(frac_with_faces) +
        0.10 * clamp(c)
    ) - clamp(no_people * 2.5) * 0.35           # penalise if few people

    # HUMANIST / EVERYDAY: people-dominant, warmth, auth — highest human of all
    # Key: human is the single dominant dimension, auth supports it
    raw["Humanist/Everyday"] = (
        0.50 * clamp(people_heavy * 3.5) +       # human must be very high
        0.25 * clamp(human_x_auth * 2.0) +
        0.15 * clamp(frac_with_faces) +
        0.10 * clamp(avg_faces / 2.0)
    ) - clamp(no_people * 3.0) * 0.50           # hard penalty without people

    # CINEMATIC / EDITORIAL: light is the dominant axis, mood over sharpness
    # Key: light far above other dimensions
    light_dominance = clamp(l - max(t, c, a, h) + 0.10)
    raw["Cinematic/Editorial"] = (
        0.45 * clamp(l * 1.4) +
        0.30 * clamp(light_dominance * 3.0) +
        0.15 * clamp(comp_x_light * 1.5) +
        0.10 * clamp(1.0 - abs(h - 0.45))       # some human but not dominant
    ) - clamp((0.55 - l) * 3.0) * 0.40          # hard penalty if light is not high

    # LANDSCAPE WITH ELEMENTS: high light + high comp, very low human
    # Key: comp_x_light interaction AND absence of people
    raw["Landscape with Elements"] = (
        0.40 * clamp(comp_x_light * 2.0) +
        0.30 * clamp(no_people * 3.0) +          # no people is a positive signal here
        0.20 * clamp(l * 1.3) +
        0.10 * clamp(c * 1.3)
    ) - clamp(frac_with_faces * 2.0) * 0.40     # penalise if faces appear often

    # MINIMALIST / URBEX: comp dominates everything, low people, controlled palette
    # Key: comp is the single highest dimension, auth/human low
    comp_dominance = clamp(c - max(t, l, a, h) + 0.10)
    raw["Minimalist/Urbex"] = (
        0.45 * clamp(c * 1.4) +
        0.30 * clamp(comp_dominance * 3.5) +
        0.15 * clamp(no_people * 2.0) +
        0.10 * clamp(t)
    ) - clamp(frac_with_faces * 1.5) * 0.30     # penalise faces

    # FINE ART / CONTEMPORARY: comp + intentionality, not purely about people or place
    # Key: comp high, auth genuinely low (staged/conceptual, not candid street)
    # clamp(max(0, 0.50-a) * 3) only rewards when auth is truly low; drops to 0 at auth >= 0.50
    raw["Fine Art/Contemporary"] = (
        0.40 * clamp(c * 1.3) +
        0.25 * clamp(l) +
        0.20 * clamp(max(0.0, 0.50 - a) * 3.0) +
        0.15 * clamp(t)
    ) - clamp(a - 0.65) * 0.45                  # harder penalty if clearly candid

    # London Street: atmosphere + human, urban mood, between street and cinematic
    # Needs both human presence AND decent light  penalise if either is absent
    # (Both penalties are INSIDE the assignment: the light-quality term used to
    # sit on its own line as a bare expression statement, so it was evaluated
    # and discarded — the intended penalty never applied.)
    raw["London Street"] = (
        0.35 * clamp(l * 1.2) +
        0.30 * clamp(human_x_auth * 1.8) +
        0.20 * clamp(h) +
        0.15 * clamp(a)
        - clamp(no_people * 2.0) * 0.35         # penalise if very few people
        - clamp((0.40 - l) * 3.0) * 0.25        # penalise if light quality is poor
    )

    # ── Normalise scores to [0, 1] ────────────────────────────────────────────
    min_r = min(raw.values())
    max_r = max(raw.values())
    spread = max(max_r - min_r, 0.01)
    normalised = {name: (v - min_r) / spread for name, v in raw.items()}

    ranked = sorted(normalised.items(), key=lambda x: x[1], reverse=True)
    best_preset = ranked[0][0]
    # Confidence = how far the winner leads the runner-up (not just that it won).
    # Gap of 0.40+ → 99%; gap of 0.20 → ~50%; gap of 0.08 → ~20%.
    _gap = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    _confidence = int(round(min(_gap * 2.5, 1.0) * 99))

    REASONS = {
        "Snapshot / Point-and-Shoot":
            "Batch shows raw immediacy — imperfect technique, high candid energy, variable quality. The moment is the priority.",
        "Classic Street":
            "Decisive moments, deliberate framing, and human presence in balance. Classic street photography benchmark.",
        "Photojournalism":
            "High technical sharpness combined with strong authenticity and human impact. Aligns with documentary standards.",
        "Travel Editor":
            "Strong cultural presence and authentic immersion. People and place work together across the batch.",
        "Humanist/Everyday":
            "People dominate the frame throughout. Warm, candid, dignity-driven — the human subject is the story.",
        "Cinematic/Editorial":
            "Light is the dominant force. Atmospheric, mood-driven, with cinematic colour and tonal direction.",
        "Landscape with Elements":
            "Natural light and compositional depth without human subjects. Foreground-layered environmental storytelling.",
        "Minimalist/Urbex":
            "Composition is the single strongest signal. Clean reduction, negative space, and structural purity.",
        "Fine Art/Contemporary":
            "Compositional intent over candid capture. Conceptual framing and tonal control elevate it beyond documentation.",
        "London Street":
            "Urban atmosphere and human presence in soft, directional light. Between street photography and cinematic mood.",
    }

    # ── Per-niche actionable guidance ──────────────────────────────────────────
    GUIDANCE = {
        "Classic Street": {
            "submit":  ["World Street Photography Awards", "Burn Magazine", "6 Mois"],
            "market":  "Editorial agencies (Panos, VII), documentary publishers, photobook imprints, festival circuits (Visa Pour l'Image).",
            "study":   ["Vivian Maier", "Alex Webb", "Daido Moriyama"],
        },
        "Travel Editor": {
            "submit":  ["Travel Photographer of the Year", "National Geographic Open Call", "Wanderlust Photo Awards", "Condé Nast Traveler"],
            "market":  "Travel magazines, tourism boards, airline in-flight media, hotel and hospitality brands.",
            "study":   ["Steve McCurry", "Ami Vitale", "Jonas Bendiksen"],
        },
        "Photojournalism": {
            "submit":  ["World Press Photo", "POYi", "Pictures of the Year International", "Bayeux-Calvados Award", "W. Eugene Smith Grant"],
            "market":  "Wire agencies (AP, Reuters, Getty), daily newspapers, long-form digital editorial, documentary book publishers.",
            "study":   ["James Nachtwey", "Lynsey Addario", "Sebastião Salgado"],
        },
        "Cinematic/Editorial": {
            "submit":  ["LensCulture Art Photography Awards", "Sony World Photography (Creative)", "1854 Media Awards", "IPA Advertising"],
            "market":  "Advertising agencies, film and TV production, fashion editorial, luxury brand campaigns.",
            "study":   ["Gregory Crewdson", "Philip-Lorca diCorcia", "Saul Leiter"],
        },
        "Fine Art/Contemporary": {
            "submit":  ["Paris Photo", "Rencontres d'Arles", "LensCulture Emerging Talent", "Foam Talent Call", "Aperture Summer Open"],
            "market":  "Gallery representation, museum acquisitions, art collectors, photobook publishers (Mack, Loose Joints, SPBH).",
            "study":   ["Wolfgang Tillmans", "Alec Soth", "Stephen Shore"],
        },
        "Minimalist/Urbex": {
            "submit":  ["Mono Awards", "Tokyo International Photo Awards", "B&W Spider Awards", "Chromatic Awards (Architecture)"],
            "market":  "Interior design publications, architectural practices, fine art print collectors, corporate art acquisitions.",
            "study":   ["Fan Ho", "Michael Kenna", "Hiroshi Sugimoto"],
        },
        "London Street": {
            "submit":  ["Street Foto San Francisco", "Sony World Photography (Street)", "Street Photo Prize"],
            "market":  "UK cultural institutions, editorial press, documentary photobooks, urban lifestyle brands.",
            "study":   ["Nick Turpin", "Matt Stuart", "Jesse Marlow"],
        },
        "Snapshot / Point-and-Shoot": {
            "submit":  ["Dazed Photography Awards", "It's Nice That", "Shoot Film Co Annual", "Superchief Gallery Open"],
            "market":  "Youth and lifestyle brands, music press, zine and independent publishers, social-first editorial.",
            "study":   ["Nan Goldin", "Wolfgang Tillmans", "Ryan McGinley"],
        },
        "Landscape with Elements": {
            "submit":  ["Landscape Photographer of the Year (UK)", "GDT European Wildlife", "Nature TTL Photographer of the Year", "Outdoor Photographer of the Year"],
            "market":  "Calendar publishers, tourism and national park bodies, outdoor gear brands, fine art print galleries.",
            "study":   ["Michael Kenna", "Charlie Waite", "Art Wolfe"],
        },
        "Humanist/Everyday": {
            "submit":  ["Taylor Wessing Portrait Prize", "Head On Portrait Prize", "Sony World Photography (Portraits)", "Humanity Photo Awards"],
            "market":  "NGO and charity publishers, editorial press (colour supplements), portrait documentary books, cultural foundations.",
            "study":   ["Dorothea Lange", "Mary Ellen Mark", "Platon"],
        },
    }

    # ── Dimension-specific coaching ────────────────────────────────────────────
    # Improve: keyed to the photographer's weakest average dimension.
    DIM_IMPROVE = {
        "tech":  "Technical execution is your floor to raise — sharper focus and cleaner exposure separate keepers from near-misses. Shoot in better light or use a faster shutter.",
        "comp":  "Compositional intentionality is what separates your shots from everyone else's — look for geometric tension, layering, and negative space before you press the shutter.",
        "light": "Light quality transforms good subjects into great photographs. Extend your sessions into early morning and late afternoon. Overcast diffusion is underrated.",
        "auth":  "Wait one beat longer. The decisive moment is usually a half-second ahead of where most photographers fire — resist the urge to shoot on the approach.",
        "human": "Close the distance. Proximity and genuine presence create the human connection missing from these frames. Engage before you raise the camera.",
    }

    # Strength: keyed to the strongest average dimension.
    DIM_STRENGTH = {
        "tech":  "Technical precision is your competitive floor — sharp, clean frames give editors nothing to reject on technical grounds.",
        "comp":  "Compositional instinct is your signature — your frames show geometry and intentionality that stop the edit.",
        "light": "Light is your strongest tool — atmospheric, directional, mood-driven exposures recur consistently across the batch.",
        "auth":  "Decisive-moment capture is where you stand out — peak gesture, unguarded expression, unrepeatable timing.",
        "human": "Human presence and cultural depth are your clearest signal — subjects feel authentic, unposed, and alive.",
    }

    dim_avgs   = {"tech": t, "comp": c, "light": l, "auth": a, "human": h}
    weakest    = min(dim_avgs, key=dim_avgs.get)
    strongest  = max(dim_avgs, key=dim_avgs.get)
    guidance   = GUIDANCE.get(best_preset, {})

    return {
        "preset":     best_preset,
        "confidence": _confidence,
        "reason":     REASONS.get(best_preset, "Best match for this batch's visual signature."),
        "ranking":    [{"preset": n, "score": round(s, 3)} for n, s in ranked],
        "submit":     guidance.get("submit", []),
        "market":     guidance.get("market", ""),
        "study":      guidance.get("study", []),
        "improve":    DIM_IMPROVE.get(weakest, ""),
        "strength":   DIM_STRENGTH.get(strongest, ""),
        "weakest":    weakest,
        "strongest":  strongest,
    }


# ---------------------------------------------------------------------------
# Pre-grade niche detection
# ---------------------------------------------------------------------------
# Lets the pre-grade picker auto-select the ideal niche BEFORE the full cull.
# Runs a fast scan_mode pass (CLIP composition scores — no IQA, no Ollama) over
# the folder through the persistent worker, then scores niches with the same
# engine as /api/recommend. Non-streaming and gpu_lock-serialised, so it never
# races a real grade for the GPU or the shared response queue, and never touches
# gallery state.
#
# The recommender works in a 10-archetype space; the picker speaks the 20-niche
# registry (src/niche_registry.py). This lossy map projects the archetype onto
# the closest registry slug so setPreset() actually matches a dropdown option.
# Images to scan for detection — a representative sample, capped so the pass
# finishes in ~seconds on a warm worker no matter how large the folder is.
_DETECT_SAMPLE = 24

_ARCHETYPE_TO_NICHE = {
    "Classic Street":             "classic_street",
    "Snapshot / Point-and-Shoot": "classic_street",
    "Photojournalism":            "photojournalism",
    "Travel Editor":              "travel_cultural",
    "Humanist/Everyday":          "documentary",
    "Cinematic/Editorial":        "liminal",
    "Landscape with Elements":    "landscape",
    "Minimalist/Urbex":           "minimalist",
    "Fine Art/Contemporary":      "fine_art",
    "London Street":              "urban_city",
}


async def _scan_folder_for_data(all_folders: list, preset: str, sample_limit: int) -> list:
    """Fast scan_mode pass through the persistent worker; returns the graded
    photo dicts (with breakdowns) for niche scoring, or [] if the worker dies /
    produces nothing. force_rescan=False so an already-graded folder resolves
    instantly on re-open."""
    import queue as _std_queue
    _gl = gpu_lock
    if _gl is not None:
        await _gl.acquire()
    try:
        loop = asyncio.get_running_loop()
        req_q, resp_q = await loop.run_in_executor(None, _ensure_worker)
        req_q.put({
            "folders":      all_folders,
            "preset":       preset or "Classic Street",
            "force_rescan": False,
            "scan_mode":    True,     # fast CLIP pass — no IQA, no Ollama gate
            "catalog_path": str(_CATALOG_PATH),
            "data_dir":     str(_DATA_DIR),
            "mogco_target": 0,
            "sample_limit": sample_limit,     # cap to a representative subset → ~seconds
            "detect_only":  True,             # never clears/writes catalog.json
        })

        def _try_get():
            try:
                return resp_q.get(timeout=1.0)
            except _std_queue.Empty:
                return None

        _deadline = asyncio.get_running_loop().time() + 240.0   # hard stop: a
        # worker that is alive but wedged must not hold gpu_lock forever — the
        # caller falls back to the CPU detector and the UI stays responsive.
        while True:
            try:
                msg = await asyncio.wait_for(loop.run_in_executor(None, _try_get), timeout=20.0)
            except asyncio.TimeoutError:
                msg = None
            if msg is None:
                if _worker_proc is None or not _worker_proc.is_alive():
                    return []                      # worker died — caller falls back
                if asyncio.get_running_loop().time() > _deadline:
                    print("[server] scan_mode pass timed out after 240 s — falling back")
                    return []
                continue
            if msg.get("error"):
                return []
            if msg.get("done"):
                return msg.get("data", []) or []
    finally:
        if _gl is not None:
            _gl.release()


@router.post("/api/recommend-niche")
async def recommend_niche(req: GradeRequest):
    """Instant pre-grade niche recommendation for the picker.

    Uses the warm CPU CLIP ViT-B/32 detector (src/fast_niche_detector.py) over a
    small, size-adaptive sample of the folder — no GPU, no grading pipeline, and
    no gpu_lock, so it can never stall previews or crash a grade, and returns in
    well under 3 s. Returns a registry slug the dropdown can select directly;
    falls back to classic_street so the picker is never blocked."""
    all_folders = [str(Path(fp).resolve()) for fp in req.folder_paths if os.path.isdir(fp)]
    if not all_folders and req.folder_path and os.path.isdir(req.folder_path):
        all_folders = [str(Path(req.folder_path).resolve())]
    if not all_folders:
        return {"preset": "classic_street", "confidence": 0, "detected": False,
                "reason": "No valid folder to scan."}

    # Gather candidate images (non-recursive, same rule as /api/browse-folder).
    img_paths: list[str] = []
    for d in all_folders:
        try:
            for p in Path(d).iterdir():
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    img_paths.append(str(p))
        except (PermissionError, OSError):
            continue
    if not img_paths:
        return {"preset": "classic_street", "confidence": 0, "detected": False,
                "reason": "No images found to scan."}
    img_paths.sort()

    try:
        from fast_niche_detector import detect as _detect_niche
        loop = asyncio.get_running_loop()
        rec = await loop.run_in_executor(
            None, lambda: _detect_niche(img_paths, req.sample_limit)
        )
    except Exception as e:
        return {"preset": "classic_street", "confidence": 0, "detected": False,
                "reason": f"Detection failed: {e}"}
    if not rec:
        return {"preset": "classic_street", "confidence": 0, "detected": False,
                "reason": "Could not read images — pick a niche manually."}
    rec["detected"] = True
    return rec


# ---------------------------------------------------------------------------
# Export magazine carousel
# ---------------------------------------------------------------------------

@router.post("/api/export/magazine")
async def export_magazine(payload: dict):
    try:
        images = payload.get("images", [])
        if len(images) < 5:
            raise HTTPException(400, "Need 5 images")
        clean_data = [
            {"path": i["path"], "rationale": i.get("rationale", ""), "presenter": "Curator"}
            for i in images
        ]
        generate_magazine_carousel, _ = _get_editorial_fns()
        zip_path = generate_magazine_carousel(clean_data)
        return FileResponse(zip_path, media_type="application/x-zip-compressed",
                            filename="Magazine_Carousel.zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Export full-res photos by grade
# ---------------------------------------------------------------------------

@router.post("/api/export/grades")
async def export_by_grade(payload: dict):
    import shutil
    photos    = payload.get("photos", [])       # [{path, grade}, ...]
    dest      = payload.get("dest", "").strip()
    grades    = set(payload.get("grades", []))  # e.g. ["Strong ✅", "Mid ⚠️"]

    if not dest:
        raise HTTPException(400, "dest folder is required")
    if not grades:
        raise HTTPException(400, "at least one grade must be selected")

    dest_root = Path(dest)
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(400, f"Cannot create destination folder: {e}")

    copied, skipped, errors = 0, 0, []
    for item in photos:
        src_path = item.get("path", "")
        grade    = item.get("grade", "")
        if grade not in grades:
            skipped += 1
            continue
        src = Path(src_path)
        if not src.exists():
            errors.append(src_path)
            continue
        # Subfolder per grade, strip emoji for safe dir name. Grade strings are
        # already "{Word} {emoji}" (e.g. "Mid ⚠️") — replacing the emoji WITH
        # the word duplicated it ("Mid Mid"); just strip the emoji instead.
        safe_grade = grade.replace("✅", "").replace("⚠️", "").replace("❌", "").strip()
        out_dir = dest_root / safe_grade
        out_dir.mkdir(exist_ok=True)
        dest_file = out_dir / src.name
        # Avoid silent overwrite — append suffix if name collides
        counter = 1
        while dest_file.exists():
            dest_file = out_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        try:
            shutil.copy2(src, dest_file)
            copied += 1
        except Exception as e:
            errors.append(f"{src_path}: {e}")

    return {
        "copied":  copied,
        "skipped": skipped,
        "errors":  errors,
        "dest":    str(dest_root),
    }


# ---------------------------------------------------------------------------
# Editorial endpoint (slot-based selection + render)
# ---------------------------------------------------------------------------

@router.post("/api/editorial")
async def generate_editorial(payload: dict, fmt: str = Query("portrait")):
    import random
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity as _cos
    from datetime import datetime

    items_raw      = payload.get("photos", [])
    excluded_paths = set(payload.get("excluded_paths", []))

    if not items_raw:
        raise HTTPException(400, "No photos provided")

    scored = [
        {"path": p["path"], "score": p.get("score", 0), "grade": p.get("grade", ""),
         "breakdown": p.get("breakdown", {}),
         "emb": np.array(analyzer.cache.get(p["path"], {}).get("embedding", [0.0] * 384))}
        for p in items_raw
        if p["path"] not in excluded_paths and p.get("score", 0) > 0
    ]
    if len(scored) < 5:
        # Reset exclusions if pool too small
        scored = [
            {"path": p["path"], "score": p.get("score", 0), "grade": p.get("grade", ""),
             "breakdown": p.get("breakdown", {}),
             "emb": np.array(analyzer.cache.get(p["path"], {}).get("embedding", [0.0] * 384))}
            for p in items_raw if p.get("score", 0) > 0
        ]
    if len(scored) < 5:
        raise HTTPException(400, "Need at least 5 scoreable photos")

    scored.sort(key=lambda x: x["score"], reverse=True)
    pool = scored[:max(int(len(scored) * 0.7), 15)]
    rng  = random.Random(random.randint(0, 999_999))

    slot_roles = [
        {"Composition": 0.5, "Technical": 0.3, "Lighting": 0.2},
        {"human": 0.5, "auth": 0.4, "comp": 0.1},
        {"tech": 0.6, "comp": 0.3, "light": 0.1},
        {"light": 0.6, "auth": 0.3, "comp": 0.1},
        {},
    ]

    def _role_score(it, weights):
        b    = it.get("breakdown", {})
        vals = list(b.values())
        if not vals: return it["score"]
        pos  = {"tech": 0, "comp": 1, "light": 2, "auth": 3, "human": 4,
                "Composition": 1, "Technical": 0, "Lighting": 2}
        s = 0.0
        for k, w in weights.items():
            if k in b:                            s += b[k] * w
            elif k in pos and pos[k] < len(vals): s += vals[pos[k]] * w
        return s

    selected, used = [], set()
    for weights in slot_roles:
        candidates = [s for s in pool if s["path"] not in used]
        if not candidates: break
        if not weights:
            if selected:
                sel_embs = np.stack([s["emb"] for s in selected])
                best, best_d = None, -1.0
                for cand in candidates:
                    d = 1.0 - float(_cos(cand["emb"].reshape(1, -1), sel_embs).min())
                    if d > best_d: best_d, best = d, cand
                pick = best
            else:
                pick = rng.choice(candidates)
        else:
            ranked = sorted(candidates,
                            key=lambda s: _role_score(s, weights) + rng.uniform(0, 0.10),
                            reverse=True)
            pick = rng.choice(ranked[:min(4, len(ranked))])
        selected.append(pick)
        used.add(pick["path"])

    slot_labels = ["Opening", "Human Moment", "Detail", "Mood", "Closing"]
    for i, s in enumerate(selected):
        s["rationale"] = analyzer.cache.get(s["path"], {}).get("rationale", "") or slot_labels[i]

    out_dir = Path("output/editorial") / datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        _, render_editorial_carousel = _get_editorial_fns()
        out_paths, zip_path = render_editorial_carousel(selected, out_dir, fmt=fmt)
    except Exception as e:
        raise HTTPException(500, str(e))

    return JSONResponse([
        {"path": p, "source_path": selected[i]["path"],
         "score": selected[i]["score"], "grade": selected[i]["grade"],
         "rationale": selected[i]["rationale"], "zip": zip_path}
        for i, p in enumerate(out_paths)
    ])


# ---------------------------------------------------------------------------
# Batch ZIP export — one download for "Download all" instead of N anchor-clicks
# (Chromium/WebView2 blocks automatic multi-downloads after the first).
# ---------------------------------------------------------------------------

@router.post("/api/export/batch-zip")
async def export_batch_zip(payload: dict):
    """Zip a list of photo paths into output/batch_<ts>.zip. Body: {paths: [...]}."""
    import zipfile
    from datetime import datetime

    raw_paths = payload.get("paths") or []
    if not raw_paths:
        raise HTTPException(400, "No photos provided")

    srcs = []
    for p in raw_paths:
        try:
            srcs.append(_safe_image_path(str(p)))
        except HTTPException:
            continue  # skip missing/deleted files rather than failing the batch
    if not srcs:
        raise HTTPException(400, "None of the provided photos could be found")

    out_dir = _OUTPUT_DIR_ZIP / "batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    def _build() -> str:
        # Unique names inside the archive (photos from multiple folders can collide).
        used: set = set()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for src in srcs:
                name = src.name
                stem, ext = src.stem, src.suffix
                i = 1
                while name in used:
                    name = f"{stem}_{i}{ext}"
                    i += 1
                used.add(name)
                zf.write(str(src), arcname=name)
        return str(zip_path)

    zip_str = await run_in_threadpool(_build)
    return JSONResponse({"zip": zip_str, "count": len(srcs), "skipped": len(raw_paths) - len(srcs)})


# ---------------------------------------------------------------------------
# Native folder picker (used by Edge app mode — no pywebview js_api available)
# ---------------------------------------------------------------------------

@router.get("/api/pick-folder")
async def pick_folder_dialog():
    """Opens a native OS folder-picker dialog and returns the chosen path."""
    import asyncio, subprocess, sys, os, tempfile, ctypes

    # Use ctypes to call Windows API directly - no subprocess needed
    def _show_dialog():
        try:
            # Windows API constants
            BIF_RETURNONLYFSDIRS = 0x00000001
            BIF_NEWDIALOGSTYLE = 0x00000040
            
            # Define BROWSEINFO structure
            class BROWSEINFO(ctypes.Structure):
                _fields_ = [
                    ('hwndOwner', ctypes.c_void_p),
                    ('pidlRoot', ctypes.c_void_p),
                    ('pszDisplayName', ctypes.c_char_p),
                    ('lpszTitle', ctypes.c_char_p),
                    ('ulFlags', ctypes.c_uint),
                    ('lpfn', ctypes.c_void_p),
                    ('lParam', ctypes.c_void_p),
                    ('iImage', ctypes.c_int)
                ]
            
            # Get the folder path using Windows API
            ctypes.windll.shell32.Shell32_SHBrowseForFolderW.argtypes = [ctypes.POINTER(BROWSEINFO)]
            ctypes.windll.shell32.Shell32_SHBrowseForFolderW.restype = ctypes.c_void_p
            
            # Use a simpler approach - create a temporary Python script that uses tkinter
            # but runs with pythonw.exe to avoid console window
            script = '''
import tkinter as tk
import tkinter.filedialog as fd
import sys
import os

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', True)
root.focus_force()

# Try to use the newer dialog style
try:
    p = fd.askdirectory(
        title='Select Photo Folder',
        parent=root,
        initialdir=os.path.expanduser('~')
    )
except:
    p = fd.askdirectory(title='Select Photo Folder', parent=root)

root.destroy()
print(p if p else '', end='')
'''
            
            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                temp_script = f.name
            
            try:
                # Use pythonw.exe to run the script
                _py = sys.executable
                if os.name == "nt" and _py.lower().endswith("python.exe"):
                    _pyw = _py[:-10] + "pythonw.exe"
                    if os.path.exists(_pyw):
                        _py = _pyw
                
                proc = subprocess.Popen(
                    [_py, temp_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    close_fds=True
                )
                stdout, _ = proc.communicate(timeout=120)
                path = stdout.decode('utf-8').strip()
                return path if path else None
            finally:
                try:
                    os.unlink(temp_script)
                except:
                    pass
                    
        except Exception as e:
            return None

    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, _show_dialog)
    return JSONResponse({"path": path})


# ---------------------------------------------------------------------------
# XMP / JSON sidecar export
# ---------------------------------------------------------------------------

@router.post("/api/export/metadata")
async def export_metadata_endpoint(payload: dict):
    """
    Write XMP/JSON sidecars for a list of graded photos.
    payload: { photos: [{path, grade, score, critique, breakdown, nima_score}],
               dest: optional output folder }
    Returns list of {source, sidecar} pairs.
    """
    from engine_utils import export_metadata
    photos  = payload.get("photos", [])
    raw_dest = payload.get("dest") or None
    dest = str(_safe_dir_path(raw_dest)) if raw_dest else None
    if not photos:
        raise HTTPException(400, "No photos provided")
    results = []
    for p in photos:
        try:
            src = _safe_image_path(p["path"])
            sidecar = export_metadata(str(src), p, out_dir=dest)
            results.append({"source": p["path"], "sidecar": sidecar})
        except Exception as e:
            results.append({"source": p["path"], "error": str(e)})
    return JSONResponse({"exported": len([r for r in results if "sidecar" in r]),
                         "results": results})


# ---------------------------------------------------------------------------
# Incremental folder watch
# ---------------------------------------------------------------------------

_folder_watcher: "FolderWatcher | None" = None   # type: ignore[name-defined]
_watched_folder: str = ""

