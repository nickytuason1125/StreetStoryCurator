def detect_niche_breakdown(breakdown):
    """Deterministic photography niche classifier using existing metrics."""
    h = breakdown.get("Human/Culture", 0.0)
    c = breakdown.get("Composition", 0.0)
    l = breakdown.get("Lighting", 0.0)
    t = breakdown.get("Technical", 0.0)
    m = breakdown.get("Mood/Color", 0.0)

    # Priority order: specific → broad (prevents overlaps)
    if h > 0.75 and l > 0.5: return "Wedding/Event"
    if h > 0.7: return "Portrait/People"
    if h > 0.5 and m > 0.55: return "Street/Urban"
    if m > 0.65 and t > 0.5 and h < 0.3: return "Food/Culinary"
    if c > 0.75 and h < 0.2: return "Architecture"
    if l > 0.6 and h < 0.3: return "Landscape/Nature"
    if l < 0.35 and m > 0.7: return "Night/Nocturnal"
    if t > 0.85 and c > 0.7: return "Product/Commercial"
    if h > 0.4 and c > 0.6: return "Documentary/Travel"
    return "General/Mixed"

def classify_with_fallback(ai_result: str, ai_conf: float, breakdown: dict):
    """AI only overrides metrics if confidence is very high."""
    metric_niche = detect_niche_breakdown(breakdown)
    if ai_conf > 0.85 and ai_result != metric_niche:
        return ai_result, "ai_override"
    return metric_niche, "metric_driven"
