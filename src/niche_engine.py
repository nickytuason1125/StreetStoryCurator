def detect_niche_breakdown(breakdown):
    """
    Tiered photography niche classifier.

    Uses raw image signals (_n_faces, histogram fractions, DINOv2 decisive, etc.)
    in addition to style-adjusted dimensions so that the adjusted floors in
    Street/Environmental/Night contexts don't bleed into niche assignment.

    Priority order (most specific → most general):
      1. Face-driven       — Wedding/Event, Portrait/People
      2. Tonal extremes    — Fine Art, Silhouette, High Key, Night/Nocturnal
      3. Motion            — Sports/Action, Motion/Long Exposure
      4. Technical extremes— Macro/Detail, Product/Commercial
      5. Environment       — Minimalist/Urbex, Architecture, Landscape/Nature, Food/Culinary
      6. Human-narrative   — Street/Urban, Documentary/Travel
      7. Fallback          — General/Mixed
    """
    # Style-adjusted dimensions (used where raw signals are absent)
    h = breakdown.get("Human/Culture", 0.0)
    c = breakdown.get("Composition",   0.0)
    l = breakdown.get("Lighting",      0.0)
    t = breakdown.get("Technical",     0.0)
    m = breakdown.get("Mood/Color",    0.0)

    # Boolean tonal / motion signals
    high_key         = bool(breakdown.get("_high_key",         False))
    backlit          = bool(breakdown.get("_backlit",           False))
    directional_blur = bool(breakdown.get("_directional_blur",  False))
    chiaroscuro      = bool(breakdown.get("_chiaroscuro",       False))

    # Raw measurement signals (richer, not style-adjusted)
    n_faces      = int(  breakdown.get("_n_faces",            0))
    subj_prom    = float(breakdown.get("_subject_prominence",  0.0))
    decisive     = float(breakdown.get("_decisive",            0.0))
    bright_frac  = float(breakdown.get("_bright_frac",        0.0))
    shadow_dark  = float(breakdown.get("_shadow_dark",        0.0))
    center_mean  = float(breakdown.get("_center_mean",        128.0))
    contrast_sc  = float(breakdown.get("_contrast_score",     0.5))
    noise_level  = float(breakdown.get("_noise_level",        0.0))
    best_sharp   = float(breakdown.get("_best_sharp",         999.0))
    blur_cv      = float(breakdown.get("_blur_cv",            1.0))

    # ── Tier 1: Face-driven ───────────────────────────────────────────────────
    # Wedding/Event: crowd of faces + strong human presence + decent light
    if n_faces >= 3 and h > 0.60 and l > 0.45:
        return "Wedding/Event"
    # Portrait: confirmed face(s) with meaningful human scoring
    if n_faces >= 1 and h > 0.55:
        return "Portrait/People"
    # Metric fallbacks for scenes where face detection misses (backs, hats, distance)
    if h > 0.75 and l > 0.50:
        return "Wedding/Event"
    if h > 0.70:
        return "Portrait/People"

    # ── Tier 2: Tonal extremes / lighting specials ────────────────────────────
    # Fine Art / Abstract: strong compositional intent + tonal marker, no dominant subject
    if (c > 0.65 and h < 0.35 and
            (high_key or backlit or chiaroscuro or m > 0.65 or l > 0.65)):
        return "Fine Art"

    # Silhouette: bright surround, centre dark, no faces — intentional counter-jour
    if backlit and n_faces == 0 and c > 0.48:
        return "Silhouette"

    # High Key: majority bright pixels + compositional intent
    if high_key and c > 0.50:
        return "High Key"

    # Night/Nocturnal: use histogram signals (much more reliable than the lighting
    # dimension which gets style-context floors applied in Street/Environmental)
    if shadow_dark > 0.35 and bright_frac < 0.25 and center_mean < 80:
        return "Night/Nocturnal"
    if l < 0.35 and m > 0.60:          # dimension fallback
        return "Night/Nocturnal"

    # ── Tier 3: Motion ───────────────────────────────────────────────────────
    # Sports/Action: directional motion + faces + high decisiveness
    if directional_blur and n_faces >= 1 and decisive > 0.55:
        return "Sports/Action"
    # Motion/Long Exposure: directional blur + compositional intent (no faces needed)
    if directional_blur and c > 0.52:
        return "Motion/Long Exposure"

    # ── Tier 4: Technical extremes ────────────────────────────────────────────
    # Macro/Detail: very sharp + rich composition + no people + tight DOF signal
    if t > 0.80 and c > 0.65 and h < 0.25 and best_sharp > 200:
        return "Macro/Detail"
    # Product/Commercial: very clean execution + strong composition + no people
    if t > 0.85 and c > 0.70 and h < 0.20:
        return "Product/Commercial"

    # ── Tier 5: Environment-driven ────────────────────────────────────────────
    # Minimalist/Urbex: high geometry, no faces, low colour energy
    if c > 0.70 and h < 0.20 and n_faces == 0 and m < 0.50:
        return "Minimalist/Urbex"
    # Architecture: dominant geometry, no/minimal people
    if c > 0.72 and h < 0.25 and n_faces == 0:
        return "Architecture"
    if c > 0.75 and h < 0.20:          # metric fallback
        return "Architecture"
    # Landscape/Nature: good light + composition, no people
    if l > 0.60 and h < 0.30 and n_faces == 0:
        return "Landscape/Nature"
    if l > 0.65 and h < 0.30:          # metric fallback
        return "Landscape/Nature"
    # Food/Culinary: warm saturated mood + decent tech + no faces
    if m > 0.65 and t > 0.50 and h < 0.30 and n_faces == 0:
        return "Food/Culinary"

    # ── Tier 6: Human-narrative ───────────────────────────────────────────────
    # Street/Urban: human presence with mood or decisive energy
    if h > 0.50 and m > 0.50:
        return "Street/Urban"
    if h > 0.45 and decisive > 0.50:
        return "Street/Urban"
    # Documentary/Travel: human + compositional intent, lower energy than street
    if h > 0.40 and c > 0.55:
        return "Documentary/Travel"
    if h > 0.45:
        return "Street/Urban"

    return "General/Mixed"


def classify_with_fallback(ai_result: str, ai_conf: float, breakdown: dict):
    """AI only overrides metrics if confidence is very high."""
    metric_niche = detect_niche_breakdown(breakdown)
    if ai_conf > 0.85 and ai_result != metric_niche:
        return ai_result, "ai_override"
    return metric_niche, "metric_driven"
