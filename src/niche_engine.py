def detect_niche_breakdown(breakdown):
    """Deterministic photography niche classifier using existing metrics."""
    h = breakdown.get("Human/Culture", 0.0)
    c = breakdown.get("Composition",   0.0)
    l = breakdown.get("Lighting",      0.0)
    t = breakdown.get("Technical",     0.0)
    m = breakdown.get("Mood/Color",    0.0)

    # Extra signals from culling/quality gates
    high_key         = bool(breakdown.get("_high_key",         False))
    backlit          = bool(breakdown.get("_backlit",           False))
    directional_blur = bool(breakdown.get("_directional_blur",  False))
    chiaroscuro      = bool(breakdown.get("_chiaroscuro",       False))

    # Priority order: specific → broad (prevents overlaps)
    if h > 0.75 and l > 0.5:  return "Wedding/Event"
    if h > 0.7:                return "Portrait/People"

    # Fine Art / Abstract — strong compositional intent, not primarily people,
    # often paired with tonal extremes (high-key, chiaroscuro, backlit, mono).
    if (c > 0.65 and h < 0.35 and
            (high_key or backlit or chiaroscuro or m > 0.65 or l > 0.65)):
        return "Fine Art"

    # High Key — intentionally bright image, compositional intent
    if high_key and c > 0.50:
        return "High Key"

    # Macro / Detail — very sharp, compositionally rich, no people
    if t > 0.80 and c > 0.65 and h < 0.25:
        return "Macro/Detail"

    # Motion / Long Exposure — directional blur as artistic tool
    if directional_blur and c > 0.52:
        return "Motion/Long Exposure"

    if h > 0.5 and m > 0.55:  return "Street/Urban"
    if m > 0.65 and t > 0.5 and h < 0.3: return "Food/Culinary"
    if c > 0.75 and h < 0.2:  return "Architecture"
    if l > 0.6  and h < 0.3:  return "Landscape/Nature"
    if l < 0.35 and m > 0.7:  return "Night/Nocturnal"
    if t > 0.85 and c > 0.7:  return "Product/Commercial"
    if h > 0.4  and c > 0.6:  return "Documentary/Travel"
    return "General/Mixed"

def classify_with_fallback(ai_result: str, ai_conf: float, breakdown: dict):
    """AI only overrides metrics if confidence is very high."""
    metric_niche = detect_niche_breakdown(breakdown)
    if ai_conf > 0.85 and ai_result != metric_niche:
        return ai_result, "ai_override"
    return metric_niche, "metric_driven"
