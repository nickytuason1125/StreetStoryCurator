import gradio as gr
import os, json, socket, random, tempfile, hashlib
from pathlib import Path
from PIL import Image
from lightweight_analyzer import LightweightStreetScorer, PRESET_RULES
import tkinter as tk
from tkinter import filedialog

analyzer   = LightweightStreetScorer()
OUTPUT_DIR = Path("./output")

# ---------------------------------------------------------------------------
# Thumbnail cache
# ---------------------------------------------------------------------------

THUMB_DIR      = Path("cache/thumbs")
MAX_THUMB_SIZE = 600
THUMB_DIR.mkdir(parents=True, exist_ok=True)

def get_or_create_thumb(img_path: str) -> str:
    # Hash the full path so images with the same filename in different subfolders
    # never collide, and the thumb is always inside THUMB_DIR (Gradio-servable).
    path_hash  = hashlib.md5(img_path.encode()).hexdigest()[:10]
    safe_name  = f"{Path(img_path).stem.replace(' ', '_')}_{path_hash}.webp"
    thumb_path = THUMB_DIR / safe_name
    if not thumb_path.exists():
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img.thumbnail((MAX_THUMB_SIZE, MAX_THUMB_SIZE), Image.Resampling.LANCZOS)
                img.save(str(thumb_path), "WEBP", quality=75, optimize=True)
        except Exception:
            return None   # caller must handle None — never return an unservable path
    return str(thumb_path)

def generate_gallery_data(results):
    out = []
    for r in results:
        thumb = get_or_create_thumb(r[0])
        if thumb:
            out.append((thumb, r[1]["grade"]))
    return out

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port(start=7860, end=7875):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start

def _to_rows(table_data):
    """Normalise gr.DataFrame output (pandas DataFrame or list) to list-of-lists."""
    if table_data is None:
        return []
    try:
        import pandas as pd
        if isinstance(table_data, pd.DataFrame):
            return table_data.values.tolist()
    except ImportError:
        pass
    return table_data if isinstance(table_data, list) else []

def _safe_folder(folder):
    """Coerce cancelled gr.update() dicts or None to empty string."""
    return folder.strip() if isinstance(folder, str) else ""

def _fmt_rationale(lines):
    """Format sequence_story rationale list as Markdown string."""
    return "\n\n".join(r for r in lines if r)

# ---------------------------------------------------------------------------
# Instagram carousel helpers
# ---------------------------------------------------------------------------

INSTAGRAM_DIR = Path("./output/instagram")
_IG_FORMATS = {
    "Square (1:1)  1080×1080":     (1080, 1080),
    "Portrait (4:5) 1080×1350":    (1080, 1350),
    "Landscape (1.91:1) 1080×566": (1080, 566),
}

def _crop_to_ratio(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h   = img.size
    target_ratio   = target_w / target_h
    src_ratio      = src_w   / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left  = (src_w - new_w) // 2
        img   = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top   = (src_h - new_h) // 2
        img   = img.crop((0, top, src_w, top + new_h))
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

# ---------------------------------------------------------------------------
# Allowed paths for Gradio file serving
# ---------------------------------------------------------------------------

def _allowed_paths():
    dirs = [
        os.path.abspath("."),
        str(THUMB_DIR.resolve()),
        str(INSTAGRAM_DIR.resolve()),
        tempfile.gettempdir(),
        str(Path.home() / "Desktop"),
        str(Path.home() / "Pictures"),
        str(Path.home() / "Downloads"),
        r"C:\Users\Nicky Tuason\Desktop",
    ]
    return list({os.path.abspath(d) for d in dirs})

# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def open_folder_dialog():
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.deiconify(); root.lift(); root.focus_force()
        root.after(0, root.withdraw)
        path = filedialog.askdirectory(title="Select Street Photo Folder")
        root.destroy()
        return path if path else gr.update()
    except Exception:
        return gr.update()

def detect_median_niche_sync(folder):
    """Detect median niche instantly when folder is selected."""
    if not folder or not os.path.isdir(folder):
        return gr.update(visible=False)
    
    try:
        # Get image paths
        exts = (".jpg", ".jpeg", ".png", ".webp")
        paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        
        if len(paths) < 2:
            return gr.update(visible=False)
        
        # Load analyzer and detect niche
        analyzer = LightweightStreetScorer()
        results = analyzer.analyze_folder(folder, preset=None, progress=None)
        
        if not results or len(results) < 2:
            return gr.update(visible=False)
        
        # Detect niche from results
        best_preset, confidence, reason = _detect_niche(results)
        
        niche_md = (
            f"### 🎯 Niche Detected: **{best_preset}** · {int(confidence*100)}% match\n"
            f"> {reason}\n\n"
            + f"*Auto-detected niche. Click **Score & Grade Photos** to grade with this niche.*"
        )
        return gr.update(value=niche_md, visible=True)
    except Exception as e:
        return gr.update(visible=False)


_ARCHETYPES = {
    "Cinematic/Editorial":   {"tech":0.3, "comp":0.4, "light":0.9, "auth":0.5, "human":0.5},
    "Travel Editor":         {"tech":0.4, "comp":0.4, "light":0.6, "auth":0.9, "human":0.8},
    "World Press Doc":       {"tech":0.8, "comp":0.5, "light":0.5, "auth":0.8, "human":0.7},
    "Fine Art/Contemporary": {"tech":0.5, "comp":0.7, "light":0.6, "auth":0.4, "human":0.9},
    "Minimalist/Urbex":      {"tech":0.6, "comp":0.9, "light":0.5, "auth":0.3, "human":0.4},
    "Humanist/Everyday":     {"tech":0.4, "comp":0.5, "light":0.5, "auth":0.7, "human":0.8},
    "Street - Magnum":       {"tech":0.6, "comp":0.7, "light":0.5, "auth":0.7, "human":0.6},
}
_ARCHETYPE_REASONS = {
    "Cinematic/Editorial":   "High atmospheric lighting & mood contrast detected.",
    "Travel Editor":         "Strong cultural authenticity & sense of place detected.",
    "World Press Doc":       "High technical clarity & journalistic context detected.",
    "Fine Art/Contemporary": "Strong geometric abstraction & emotional resonance detected.",
    "Minimalist/Urbex":      "High compositional purity & negative space detected.",
    "Humanist/Everyday":     "High candid intimacy & human warmth detected.",
    "Street - Magnum":       "Balanced decisive moments & candid layering detected.",
}

def _detect_niche(results):
    """Cosine similarity against archetype vectors. Returns (preset, confidence, reason)."""
    # Map new breakdown keys to archetype keys
    key_map = {
        "Technical": "tech",
        "Composition": "comp",
        "Lighting": "light",
        "Narrative": "auth",
        "Human/Culture": "human"
    }
    metrics = {"tech": 0.0, "comp": 0.0, "light": 0.0, "auth": 0.0, "human": 0.0}
    count = len(results)
    if count == 0:
        return "Street - Magnum", 0.0, "No images."
    for _, d in results:
        breakdown = d.get("breakdown", {})
        for new_key, old_key in key_map.items():
            if new_key in breakdown:
                metrics[old_key] += breakdown[new_key]
    for k in metrics:
        metrics[k] /= count
    best_preset, best_score = "Street - Magnum", -1.0
    for name, ideal in _ARCHETYPES.items():
        dot    = sum(metrics[k] * ideal[k] for k in metrics.keys())
        norm_a = sum(v**2 for v in metrics.values()) ** 0.5 + 1e-6
        norm_b = sum(v**2 for v in ideal.values())   ** 0.5 + 1e-6
        sim    = dot / (norm_a * norm_b)
        if sim > best_score:
            best_score, best_preset = sim, name
    return best_preset, round(best_score, 2), _ARCHETYPE_REASONS.get(best_preset, "")


def _build_table(results):
    return [
        [
            r[0],
            r[1]["grade"],
            r[1]["score"],
            " | ".join(f"{k}: {v}" for k, v in r[1].get("breakdown", {}).items()),
            r[1]["critique"],
        ]
        for r in results
    ]


def load_and_score(folder, sort_exif, human_mode, progress=gr.Progress()):
    folder = _safe_folder(folder)
    if not folder or not os.path.isdir(folder):
        yield gr.update(), gr.update(), "❌ Invalid or empty folder path.", gr.update(visible=False)
        return

    # ── Grade with auto-detected niche ────────────────────────────────────
    results = analyzer.analyze_folder(folder, preset=None, progress=progress)
    if not results:
        yield gr.update(), gr.update(), "⚠️ No supported images found (jpg/png/webp).", gr.update(visible=False)
        return

    if sort_exif:
        try:
            from exif_handler import sort_by_timeline
            results = sort_by_timeline(results)
        except Exception:
            pass

    # ── Niche detection ───────────────────────────────────────────────────
    best_preset, confidence, reason = _detect_niche(results)
    applied_preset = best_preset

    strong = sum(1 for r in results if "Strong" in r[1]["grade"])
    mid    = sum(1 for r in results if "Mid"    in r[1]["grade"])
    weak   = sum(1 for r in results if "Weak"   in r[1]["grade"])

    niche_md = (
        f"### 🎯 Niche Detected: **{best_preset}** · {int(confidence*100)}% match\n"
        f"> {reason}\n\n"
        + f"*Auto-detected niche applied. Graded with **{best_preset}** criteria.*"
    )

    yield (
        generate_gallery_data(results),
        _build_table(results),
        f"✅ **{applied_preset}** — {len(results)} images · Strong {strong} / Mid {mid} / Weak {weak}",
        gr.update(value=niche_md, visible=True),
    )


def filter_gallery(table_data, filter_grade):
    rows = _to_rows(table_data)
    if not rows:
        return gr.update()
    filtered = (
        rows if filter_grade == "All"
        else [r for r in rows if len(r) > 1 and filter_grade in str(r[1])]
    )
    if not filtered:
        return []
    out = []
    for r in filtered:
        if len(r) >= 2:
            thumb = get_or_create_thumb(str(r[0]))
            if thumb:
                out.append((thumb, str(r[1])))
    return out


def generate_sequence(table_data, filter_grade):
    rows = _to_rows(table_data)
    if not rows:
        return gr.update(), gr.update(value="⚠️ Grade photos first.", visible=True), gr.update(interactive=False), []

    filtered = (
        rows if filter_grade == "All"
        else [r for r in rows if len(r) > 1 and filter_grade in str(r[1])]
    )
    if len(filtered) < 5:
        return (
            gr.update(),
            gr.update(value=f"⚠️ Need at least 5 images in the **{filter_grade}** tier.", visible=True),
            gr.update(interactive=False),
            [],
        )

    # Rebuild (path, dict) tuples — table rows only carry grade/score strings;
    # embeddings and breakdown come from the live cache.
    results_for_seq = []
    for r in filtered:
        path  = str(r[0])
        score = float(r[2]) if len(r) > 2 and r[2] is not None else 0.0
        cached = analyzer.cache.get(path, {})
        results_for_seq.append((path, {
            "score":     score,
            "grade":     str(r[1]),
            "embedding": cached.get("embedding", [0.0] * 256),
            "breakdown": cached.get("breakdown", {}),
        }))

    fresh_seed = random.randint(1000, 999_999)
    seq_paths, rationale = analyzer.sequence_story(results_for_seq, target=5, seed=fresh_seed)

    if not seq_paths:
        return gr.update(), gr.update(value="⚠️ Not enough diverse images.", visible=True), gr.update(interactive=False), []

    preview = [
        (get_or_create_thumb(p), f"Slide {i+1} · {Path(p).name}")
        for i, p in enumerate(seq_paths)
        if get_or_create_thumb(p)
    ]
    return (
        preview,
        gr.update(value=_fmt_rationale(rationale), visible=True),
        gr.update(interactive=len(seq_paths) == 5),
        seq_paths,
    )


def export_scorecard(seq_paths, preset):
    if not seq_paths:
        return gr.update(visible=False), gr.update(value="⚠️ Generate a sequence first.", visible=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_scores = dict(analyzer.cache)
    try:
        from pdf_exporter import generate_pdf
        out = generate_pdf(seq_paths, all_scores, preset, str(OUTPUT_DIR))
        return gr.update(value=str(out), visible=True), gr.update(value=f"✅ Saved: {Path(out).name}", visible=True)
    except Exception as exc:
        return gr.update(visible=False), gr.update(value=f"❌ PDF error: {exc}", visible=True)


def generate_instagram_carousel(table_data, fmt):
    import zipfile, numpy as np
    from sklearn.metrics.pairwise import cosine_similarity as _cos
    from datetime import datetime

    rows = _to_rows(table_data)
    if not rows:
        return gr.update(), gr.update(visible=False), gr.update(value="⚠️ Score some photos first.", visible=True)

    scored = []
    for r in rows:
        if len(r) < 3:
            continue
        path  = str(r[0])
        score = float(r[2]) if r[2] is not None else 0.0
        emb   = analyzer.cache.get(path, {}).get("embedding", None)
        if emb and score > 0:
            scored.append({"path": path, "score": score, "emb": np.array(emb)})

    if len(scored) < 1:
        return gr.update(), gr.update(visible=False), gr.update(value="⚠️ No valid scored images found.", visible=True)

    # Wide pool — top 60% of all scoreable photos (strong + mid), min 15
    scored.sort(key=lambda x: x["score"], reverse=True)
    pool = scored[:max(int(len(scored) * 0.6), 15)]

    if len(pool) < 1:
        return gr.update(), gr.update(visible=False), gr.update(value="⚠️ Not enough photos.", visible=True)

    # Per-generation seed so every click produces different results
    rng = random.Random(random.randint(0, 999_999))

    # Slot roles — each slot scores candidates on different breakdown keys
    # so each position draws from a different visual character
    slot_roles = [
        ("Hook",          {"Composition": 0.5, "Technical": 0.3, "Lighting": 0.2}),
        ("Human Moment",  {"Human/Culture": 0.5, "Authenticity": 0.4, "Composition": 0.1}),
        ("Detail",        {"Technical": 0.6, "Composition": 0.3, "Lighting": 0.1}),
        ("Mood",          {"Lighting": 0.6, "Authenticity": 0.3, "Composition": 0.1}),
        ("Closer",        {}),  # Pure diversity — most different from all prior picks
    ]
    # Also accept preset-specific label names by falling back to positional values
    def _slot_score(item, weights):
        b = analyzer.cache.get(item["path"], {}).get("breakdown", {})
        vals = list(b.values())
        keys = list(b.keys())
        if not vals:
            return item["score"]
        score = 0.0
        for label, w in weights.items():
            # Try exact label match first, then positional fallback
            if label in b:
                score += b[label] * w
            elif label == "Human/Culture" and len(vals) > 4:
                score += vals[4] * w
            elif label == "Authenticity" and len(vals) > 3:
                score += vals[3] * w
            elif label == "Lighting" and len(vals) > 2:
                score += vals[2] * w
            elif label == "Composition" and len(vals) > 1:
                score += vals[1] * w
            elif label == "Technical" and len(vals) > 0:
                score += vals[0] * w
        return score

    selected  = []
    used_paths = set()

    for slot_idx, (role, weights) in enumerate(slot_roles):
        candidates = [s for s in pool if s["path"] not in used_paths]
        if not candidates:
            break

        if not weights:
            # Closer: pick most visually diverse from all selected
            if selected:
                sel_embs = np.stack([s["emb"] for s in selected])
                best, best_dist = None, -1.0
                for cand in candidates:
                    sim  = float(_cos(cand["emb"].reshape(1, -1), sel_embs).min())
                    dist = 1.0 - sim
                    if dist > best_dist:
                        best_dist, best = dist, cand
                pick = best
            else:
                pick = rng.choice(candidates)
        else:
            # Score each candidate by role fit + small random jitter for freshness
            ranked = sorted(
                candidates,
                key=lambda s: _slot_score(s, weights) + rng.uniform(0, 0.08),
                reverse=True,
            )
            # Pick from top 3 so repeated generations vary
            pick = rng.choice(ranked[:min(3, len(ranked))])

        selected.append(pick)
        used_paths.add(pick["path"])

    INSTAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    target_w, target_h = _IG_FORMATS.get(fmt, (1080, 1080))
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = INSTAGRAM_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    for i, item in enumerate(selected, 1):
        try:
            with Image.open(item["path"]) as img:
                img = img.convert("RGB")
                img = _crop_to_ratio(img, target_w, target_h)
                out = run_dir / f"carousel_{i:02d}_{Path(item['path']).stem}.jpg"
                img.save(str(out), "JPEG", quality=92, optimize=True)
                out_paths.append(str(out))
        except Exception:
            out_paths.append(item["path"])

    zip_path = str(run_dir / f"instagram_carousel_{ts}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_paths:
            zf.write(p, Path(p).name)

    gallery_items = [
        (p, f"Slide {i+1} · {Path(selected[i]['path']).name}  score={selected[i]['score']:.2f}")
        for i, p in enumerate(out_paths)
    ]
    status = (
        f"✅ **Carousel ready** — {len(out_paths)} slides · "
        f"{fmt.split()[0]} format · `{run_dir.name}/`"
    )
    return gallery_items, gr.update(value=zip_path, visible=True), gr.update(value=status, visible=True)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_CSS = """
.wrap-inner     { pointer-events: auto !important; cursor: pointer !important; }
.form .checkbox { z-index: 10 !important; }
.row, .column   { z-index: 1 !important; }
"""

def build_ui():
    with gr.Blocks(title="📸 Street Story Curator") as demo:
        gr.Markdown("### 🎞️ Offline grading & cinematic sequencing for street photography.")

        # ── Folder ────────────────────────────────────────────────────────
        with gr.Row():
            folder_path = gr.Textbox(
                label="📁 Local Image Folder",
                placeholder="Paste path or click Browse",
                interactive=True, scale=3,
            )
            browse_btn = gr.Button("📂 Browse", variant="secondary", scale=1)

        # ── Options (flat Row — avoids Svelte pointer-event bug) ─
        with gr.Row(equal_height=True):
            exif_cb  = gr.Checkbox(label="🕰️ Sort by EXIF time",    value=False, interactive=True, scale=1)
            human_cb = gr.Checkbox(label="👁️ Human Preference Mode", value=True,  interactive=True, scale=1)

        score_btn  = gr.Button("🔍 Score & Grade Photos", variant="primary", size="lg")
        status_txt = gr.Markdown("")
        niche_out  = gr.Markdown("", visible=False)

        # ── Gallery + filter ──────────────────────────────────────────────
        gallery_out = gr.Gallery(label="🖼️ Graded Gallery", columns=4, height=350, object_fit="cover")
        filter_dd   = gr.Dropdown(
            choices=["All", "Strong ✅", "Mid ⚠️", "Weak ❌"],
            value="All", label="🔍 Filter by Grade", interactive=True,
        )

        # ── Scores table ──────────────────────────────────────────────────
        table_out = gr.DataFrame(
            headers=["Path", "Grade", "Score", "Breakdown", "Critique"],
            label="📋 Jury Scores", wrap=True,
        )

        gr.Markdown("---")

        # ── Instagram carousel ────────────────────────────────────────────
        gr.Markdown("### 📱 Instagram Carousel Generator")
        gr.Markdown(
            "Picks the **5 most visually diverse** high-scoring images and exports "
            "them at Instagram resolution."
        )
        with gr.Row(equal_height=True):
            ig_format_dd = gr.Dropdown(
                choices=list(_IG_FORMATS.keys()),
                value="Portrait (4:5) 1080×1350",
                label="📐 Format", interactive=True, scale=2,
            )
            ig_btn = gr.Button("✨ Generate Carousel", variant="primary", scale=1)

        ig_gallery = gr.Gallery(label="📱 Carousel Preview (5 slides)",
                                columns=5, height=300, object_fit="cover")
        ig_zip     = gr.File(label="⬇️ Download All 5 Slides (.zip)", visible=False)
        ig_status  = gr.Markdown("", visible=False)

        # ── Wiring ────────────────────────────────────────────────────────
        browse_btn.click(open_folder_dialog, outputs=[folder_path])
        
        # Detect median niche instantly when folder changes
        folder_path.change(
            detect_median_niche_sync,
            inputs=[folder_path],
            outputs=[niche_out],
        )

        score_btn.click(
            load_and_score,
            inputs=[folder_path, exif_cb, human_cb],
            outputs=[gallery_out, table_out, status_txt, niche_out],
        )

        filter_dd.change(
            filter_gallery,
            inputs=[table_out, filter_dd],
            outputs=[gallery_out],
        )

        ig_btn.click(
            generate_instagram_carousel,
            inputs=[table_out, ig_format_dd],
            outputs=[ig_gallery, ig_zip, ig_status],
        )

    return demo

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = _find_free_port()
    print(f"Launching on http://127.0.0.1:{port}")
    demo = build_ui()
    demo.queue(default_concurrency_limit=2).launch(
        server_name="127.0.0.1",
        server_port=port,
        allowed_paths=_allowed_paths(),
        theme=gr.themes.Soft(),
        css=_CSS,
        prevent_thread_lock=False,
        inline=False,
    )
