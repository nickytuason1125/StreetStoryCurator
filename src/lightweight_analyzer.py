import sys, cv2, numpy as np, json, os, hashlib, random, threading, asyncio, gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_loader import get_sessions
from reference_bank import ReferenceBank
from fast_io import bgr_to_chw, normalize_imagenet, IMAGENET_MEAN, IMAGENET_STD
from niche_engine import classify_with_fallback

def _cosine_similarity(a, b):
    # Pure numpy — no sklearn/joblib/loky, no worker-process spawning on Windows.
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def load_image_fast(path: str, max_side: int = 720) -> np.ndarray | None:
    """
    Load and downscale an image using pyvips (JPEG thumbnail decode — up to 4× faster
    than cv2.imread on full-size RAW/JPEG).  Falls back to cv2 transparently.

    Returns: BGR uint8 ndarray (cv2-compatible), None on failure.
    Pyvips buffers are freed when the vips Image goes out of scope — no leak.
    """
    try:
        import pyvips
        # thumbnail() uses JPEG shrink-on-load — decodes at reduced resolution
        # without reading the full pixel data, so it's fast even on 50 MP files.
        # Reduced from 1080 to 720 for faster grading (2.25x fewer pixels)
        vimg = pyvips.Image.thumbnail(path, max_side, size="down")
        if vimg.bands == 1:                          # grayscale → BGR
            vimg = vimg.colourspace("srgb")
        elif vimg.bands == 4:                        # RGBA → RGB
            vimg = vimg.flatten()
        arr = np.frombuffer(vimg.write_to_memory(), dtype=np.uint8).reshape(
            vimg.height, vimg.width, 3
        )
        return arr[:, :, ::-1].copy()               # RGB → BGR
    except Exception:
        # Fallback: unicode-safe cv2 load
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        ih, iw = img.shape[:2]
        if max(ih, iw) > max_side:
            scale = max_side / max(ih, iw)
            img   = cv2.resize(img, (int(iw * scale), int(ih * scale)),
                               interpolation=cv2.INTER_AREA)
        return img

PRESET_RULES = {
    # weights keys match _style_presets in _analyze exactly — "narrative" replaces old "auth"
    # so the displayed breakdown bars reflect what the scorer actually computed.
    "Classic Street": {
        "weights": {"tech":0.10, "comp":0.18, "light":0.15, "narrative":0.32, "human":0.25},
        "labels": {"tech":"Technical", "comp":"Composition", "light":"Lighting", "narrative":"Decisive Moment", "human":"Subject Isolation"},
        "guide": "Prioritises decisive moments, layering, and candid geometry.",
        "critique_high": "Peak timing & confident framing. Clear narrative tension.",
        "critique_mid": "Solid moment; slight repositioning or light direction could sharpen the story.",
        "critique_low": "Lacks a clear focal point or decisive gesture. Consider waiting for peak interaction."
    },
    "Travel Editor": {
        "weights": {"tech":0.12, "comp":0.13, "light":0.18, "narrative":0.27, "human":0.30},
        "labels": {"tech":"Technical", "comp":"Framing", "light":"Atmosphere", "narrative":"Cultural Depth", "human":"Sense of Place"},
        "guide": "Rewards authenticity, local context, and environmental storytelling.",
        "critique_high": "Strong sense of place & cultural immersion. Candid and unposed.",
        "critique_mid": "Good context; leaning slightly touristy. Capture more candid interaction.",
        "critique_low": "Feels staged or landmark-centric. Step closer to daily life or rituals."
    },
    "Photojournalism": {
        "weights": {"tech":0.13, "comp":0.13, "light":0.14, "narrative":0.40, "human":0.20},
        "labels": {"tech":"News Sharpness", "comp":"Context", "light":"Natural Light", "narrative":"Journalistic Integrity", "human":"Human Impact"},
        "guide": "Prioritises factual clarity, impact, and minimal manipulation.",
        "critique_high": "Clear impact & factual clarity. Technique serves the story.",
        "critique_mid": "Strong subject; distracting background elements reduce narrative focus.",
        "critique_low": "Over-processed or lacks contextual anchors. Prioritise raw clarity."
    },
    "Cinematic/Editorial": {
        "weights": {"tech":0.10, "comp":0.20, "light":0.30, "narrative":0.20, "human":0.20},
        "labels": {"tech":"Cleanliness", "comp":"Framing", "light":"Mood & Tone", "narrative":"Narrative Suggestion", "human":"Character Presence"},
        "guide": "Rewards mood, color tone, and narrative suggestion over literal documentation.",
        "critique_high": "Evocative mood & cinematic pacing. Light direction drives emotion.",
        "critique_mid": "Good atmosphere; flat contrast reduces depth. Enhance shadow/highlight balance.",
        "critique_low": "Clinical lighting kills mood. Wait for directional light or golden hour."
    },
    "Fine Art/Contemporary": {
        "weights": {"tech":0.10, "comp":0.25, "light":0.20, "narrative":0.15, "human":0.30},
        "labels": {"tech":"Execution", "comp":"Geometry & Balance", "light":"Tonal Purity", "narrative":"Conceptual Weight", "human":"Emotional Resonance"},
        "guide": "Values abstraction, negative space, and artistic intent over literal capture.",
        "critique_high": "Strong conceptual depth & tonal harmony. Composition feels intentional.",
        "critique_mid": "Visually competent; lacks abstraction or negative space to elevate it.",
        "critique_low": "Too literal. Crop tightly or wait for geometric alignment to simplify."
    },
    "Minimalist/Urbex": {
        "weights": {"tech":0.15, "comp":0.30, "light":0.20, "narrative":0.10, "human":0.25},
        "labels": {"tech":"Detail Retention", "comp":"Negative Space", "light":"Contrast Purity", "narrative":"Reduction", "human":"Scale Element"},
        "guide": "Penalises clutter. Rewards simplicity, symmetry, and tonal purity.",
        "critique_high": "Clean reduction & strong geometry. Distractions successfully eliminated.",
        "critique_mid": "Good structure; edge clutter breaks minimalism. Crop tighter or angle lower.",
        "critique_low": "Overly busy. Seek uniform surfaces, clean lines, or isolated subjects."
    },
    "LSPF (London Street)": {
        "weights": {"tech":0.07, "comp":0.17, "light":0.25, "narrative":0.26, "human":0.25},
        "labels": {"tech":"Technical", "comp":"Composition", "light":"Lighting", "narrative":"Authenticity", "human":"Human/Culture"},
        "guide": "Balances light atmosphere with candid human presence in urban settings.",
        "critique_high": "Strong atmosphere & candid street energy. Light and subject work together.",
        "critique_mid": "Good urban feel; light or subject placement could be stronger.",
        "critique_low": "Lacks street energy or atmosphere. Look for stronger light or human interaction."
    },
    "Snapshot / Point-and-Shoot": {
        "weights": {"tech":0.08, "comp":0.12, "light":0.15, "narrative":0.40, "human":0.25},
        "labels": {"tech":"Exposure", "comp":"Framing Instinct", "light":"Available Light", "narrative":"Immediacy", "human":"Presence"},
        "guide": "Rewards raw immediacy over craft. Blur, grain, and imperfect framing are acceptable — the moment is everything.",
        "critique_high": "Pure decisive capture. Imperfections serve the urgency of the moment.",
        "critique_mid": "Good instinct; slight hesitation in timing or framing dilutes the rawness.",
        "critique_low": "The moment is missing. Technical flaws without emotional payoff produce neither documentary nor art."
    },
    "Landscape with Elements": {
        "weights": {"tech":0.18, "comp":0.32, "light":0.35, "narrative":0.05, "human":0.10},
        "labels": {"tech":"Sharpness & Detail", "comp":"Layered Depth", "light":"Natural Light Quality", "narrative":"Environmental Truth", "human":"Scale & Life"},
        "guide": "Rewards compositional depth — foreground interest, mid-ground structure, sky drama. Human or organic elements anchor scale.",
        "critique_high": "Excellent environmental layering. Foreground, mid-ground, and sky all earn their place.",
        "critique_mid": "Strong light but flat structure. Add foreground interest or wait for a layered moment.",
        "critique_low": "Empty frame with no visual hierarchy. Find a foreground anchor or wait for dramatic light."
    },
}

def get_preset_config(name):
    return PRESET_RULES.get(name, PRESET_RULES["Classic Street"])


def detect_focal_hierarchy(gray):
    # Saliency proxy: high-contrast edges + face/line convergence
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=gray.shape[1]//4, maxLineGap=15)
    line_density = 0.0 if lines is None else len(lines) / 50.0
    return min(line_density, 1.0)


def color_mood_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    warmth = np.mean(h[h < 30]) / 180.0 if np.any(h < 30) else 0.0
    saturation_harmony = 1.0 - np.std(s) / 128.0
    contrast_mood = np.sqrt(np.var(v)) / 128.0
    return 0.4*warmth + 0.3*saturation_harmony + 0.3*contrast_mood


def exif_compatibility(width, height, focal_proxy=35):
    # Simple aspect-ratio vs focal length heuristic
    ratio = max(width, height) / min(width, height)
    if focal_proxy <= 24: return 1.0 - abs(ratio - 1.5)   # wide prefers 3:2
    if focal_proxy >= 85: return 1.0 - abs(ratio - 1.33)  # telephoto prefers 4:3
    return 1.0 - abs(ratio - 1.5) * 0.5


_CACHE_VER     = 2                                    # bump to invalidate stale caches
_MAX_FILE_MB   = 50                                   # hard cap per image
_SAFE_PIL_FMTS = frozenset({"JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF"})

class LightweightStreetScorer:
    def __init__(self, cache_path="cache/light_scores.json"):
        # Use absolute path relative to this file's directory
        self.cache_path = Path(__file__).parent.parent / cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # ONNX sessions — lazy so startup stays instant
        self._ort_sessions  = None
        self._est_input     = None
        self._comp_input    = None
        self._nima_input    = None   # None until nima.onnx is generated by nima_setup.py
        self._seq_scorer    = None   # None until train_sequence_scorer.py has been run
        self._seq_input     = None
        # Face detection — YuNet ONNX (modern) with Haar cascade fallback
        self._face_yn       = None
        self._face_casc     = None
        self._profile_casc  = None
        # Reference bank — loads from disk at startup, empty until user indexes exemplars
        self._ref_bank      = ReferenceBank()
        # NicheClassifier — built lazily after first analyze_folder call
        self._niche_clf     = None
        # VLMNicheDetector — lazy; only instantiated when model file is present
        self._vlm_instance  = None
        # Style presets for _find_best_preset()
        self._STYLE_PRESETS = {
            "Classic Street":             {"tech": 0.10, "comp": 0.18, "light": 0.15, "narrative": 0.32, "human": 0.25},
            "Travel Editor":               {"tech": 0.12, "comp": 0.13, "light": 0.18, "narrative": 0.27, "human": 0.30},
            "Photojournalism":             {"tech": 0.13, "comp": 0.13, "light": 0.14, "narrative": 0.40, "human": 0.20},
            "Cinematic/Editorial":         {"tech": 0.10, "comp": 0.20, "light": 0.30, "narrative": 0.20, "human": 0.20},
            "Fine Art/Contemporary":       {"tech": 0.10, "comp": 0.25, "light": 0.20, "narrative": 0.15, "human": 0.30},
            "Minimalist/Urbex":            {"tech": 0.15, "comp": 0.30, "light": 0.20, "narrative": 0.10, "human": 0.25},
            "LSPF (London Street)":        {"tech": 0.07, "comp": 0.17, "light": 0.25, "narrative": 0.26, "human": 0.25},
            "Snapshot / Point-and-Shoot":  {"tech": 0.08, "comp": 0.12, "light": 0.15, "narrative": 0.40, "human": 0.25},
            "Landscape with Elements":     {"tech": 0.18, "comp": 0.32, "light": 0.35, "narrative": 0.05, "human": 0.10},
        }
        self.cache = self._load_cache()

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        """Load cache with version check and checksum verification."""
        if not self.cache_path.exists():
            return {}
        try:
            d = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if d.get("v") != _CACHE_VER:
                return {}                              # stale version — rebuild
            stored_chk = d.get("chk")
            payload    = {k: v for k, v in d.items() if k != "chk"}
            expected   = hashlib.md5(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
            if stored_chk != expected:
                return {}                              # tampered / corrupt — rebuild
            return {k: v for k, v in payload.items() if k != "v"}
        except Exception:
            return {}

    def _ensure_sessions(self):
        if self._ort_sessions is None:
            self._ort_sessions = get_sessions()
            self._est_input    = self._ort_sessions["aesthetic"].get_inputs()[0].name
            self._comp_input   = self._ort_sessions["composition"].get_inputs()[0].name
            if "nima" in self._ort_sessions:
                self._nima_input = self._ort_sessions["nima"].get_inputs()[0].name
        if self._seq_scorer is None:
            seq_path = Path("models/onnx/sequence_scorer.onnx")
            if seq_path.exists():
                try:
                    import onnxruntime as ort
                    self._seq_scorer = ort.InferenceSession(str(seq_path))
                    self._seq_input  = self._seq_scorer.get_inputs()[0].name
                except Exception:
                    self._seq_scorer = None
        if self._face_yn is None and self._face_casc is None:
            yn_path = Path("models/face_detection_yunet_2023mar.onnx")
            if yn_path.exists():
                try:
                    self._face_yn = cv2.FaceDetectorYN.create(
                        str(yn_path), "", (320, 320),
                        score_threshold=0.6, nms_threshold=0.3, top_k=5000,
                    )
                except Exception:
                    self._face_yn = None
            if self._face_yn is None:
                # Haar fallback — legacy but functional when YuNet model absent
                self._face_casc    = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                self._profile_casc = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_profileface.xml")
                _w = np.zeros((32, 32), dtype=np.uint8)
                self._face_casc.detectMultiScale(_w, 1.3, 3)
                self._profile_casc.detectMultiScale(_w, 1.3, 3)

    def _save_cache(self):
        """Atomic write with version tag and MD5 checksum for corruption detection."""
        try:
            payload = {**self.cache, "v": _CACHE_VER}
            payload["chk"] = hashlib.md5(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_path)              # atomic on POSIX; near-atomic on Windows
        except Exception:
            pass

    def _build_niche_anchors(self) -> int:
        """Build visual prototype anchors for NicheClassifier from cached embeddings."""
        try:
            from niche_classifier import NicheClassifier
            if self._niche_clf is None:
                self._niche_clf = NicheClassifier()
            return self._niche_clf.build_anchors(
                self.cache,
                self._COMP_KEYS, self._TECH_KEYS,
                self._HUMAN_KEYS, self._LIGHT_KEYS, self._AUTH_KEYS,
            )
        except Exception:
            return 0

    def _detect_style_context(self, b):
        h, l, m, t, c = (b.get("Human/Culture", 0), b.get("Lighting", 0),
                          b.get("Mood/Color", 0), b.get("Technical", 0), b.get("Composition", 0))
        if h > 0.6 and l < 0.45 and m > 0.65: return "Documentary/Low-Key"
        if h > 0.5 and c > 0.7:                return "Street/Environmental"
        if l < 0.3 and m > 0.7:                return "Night/Nocturnal"
        if t > 0.8 and h < 0.2:                return "Commercial/Studio"
        # Atmospheric/gritty catch-all: intentionally low-tech but purposeful framing
        # and real light — classic documentary/street pattern missed by the stricter branches.
        if t < 0.38 and c > 0.45 and l > 0.40: return "Documentary/Low-Key"
        return "Standard"

    def _analyze(self, path, preset="Classic Street"):
        # ── Input safety gates ────────────────────────────────────────────────
        try:
            if os.path.getsize(path) > _MAX_FILE_MB * 1024 * 1024:
                return self._fallback(f"File exceeds {_MAX_FILE_MB} MB limit")
        except OSError:
            return self._fallback("Cannot access file")
        try:
            from PIL import Image as _PILFmt
            with _PILFmt.open(path) as _pf:
                if _pf.format and _pf.format not in _SAFE_PIL_FMTS:
                    return self._fallback(f"Unsupported format: {_pf.format}")
        except Exception:
            pass                                      # PIL can't read RAW — let cv2 try

        self._ensure_sessions()
        try:
            img = load_image_fast(path, max_side=720)
            if img is None: return self._fallback("Failed to load")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            if h == 0 or w == 0: return self._fallback("Invalid dims")

            # ── TECHNICAL: multi-region sharpness + exposure + noise ─────────
            # 3×3 grid: find the sharpest zone — soft BG is fine if subject is sharp
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            cell_h, cell_w = max(h // 3, 1), max(w // 3, 1)
            region_vars = []
            for ri in range(3):
                for ci in range(3):
                    patch = gray[ri*cell_h:(ri+1)*cell_h, ci*cell_w:(ci+1)*cell_w]
                    if patch.size > 0:
                        region_vars.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
            best_sharp = max([lap_var] + region_vars)

            # ── Intentional-blur detection ────────────────────────────────────
            # Vintage lenses, Holga, Leica wide-open, film cameras — deliberately
            # soft images — have a characteristic signature:
            #   • ALL 3×3 regions are similarly low-sharpness (uniform softness)
            #   • Shadow noise is elevated (film grain, sensor noise at high ISO)
            #   • No single region is dramatically sharper than another
            # Accidental blur (camera shake, missed focus) looks different:
            #   • High variance between regions (one zone sharp, others smeared)
            #   • Or a directional motion smear across all regions
            # We detect "globally uniform softness" via the coefficient of variation
            # of the 3×3 region Laplacian variances.  Low CV = uniformly soft.
            if len(region_vars) > 2:
                rv = np.array(region_vars, dtype=np.float32)
                blur_cv = float(rv.std() / (rv.mean() + 1e-6))
            else:
                blur_cv = 1.0   # can't tell — assume accidental

            hist    = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            total   = float(hist.sum()) or 1.0
            blown   = float(hist[245:].sum()) / total
            blocked = float(hist[:8].sum())   / total   # pure-black pixels (0–7)
            midtone_frac = float(hist[50:210].sum()) / total
            # Wider shadow fraction for chiaroscuro detection.
            # hist[:8] catches only pure-black (0–7); bare-bulb market scenes, candlelit
            # interiors, and street-at-night shots typically have dark-gray shadows in
            # the 8–40 range that would never trip the narrow threshold.
            shadow_dark  = float(hist[:35].sum()) / total  # pixels 0–34 = "dark shadow"

            # ── Chiaroscuro detection ─────────────────────────────────────────
            # Single-source dramatic lighting (bare bulb, window shaft, candle,
            # street lamp at night) produces intentional dark surrounds alongside
            # a well-lit centre.  This is a recognised artistic style — Rembrandt,
            # Caravaggio, documentary market shots, street-at-night photography.
            # Signature: large dark surround (high shadow_dark) BUT centre is well-lit
            # AND blown area is small (it's one point source, not overexposure).
            cy_c1, cy_c2 = h // 4, 3 * h // 4
            cx_c1, cx_c2 = w // 4, 3 * w // 4
            center_mean  = float(gray[cy_c1:cy_c2, cx_c1:cx_c2].mean())
            chiaroscuro  = (
                shadow_dark  > 0.18        # substantial dark areas (0–34 range)
                and center_mean > 65       # but centre is properly lit
                and blown      < 0.12      # light source, not overexposure
                and (center_mean - float(gray.mean())) > 18  # centre clearly brighter
            )

            if chiaroscuro:
                # Score on centre brightness, not the dark-biased global mean.
                # Ignore the blocked penalty entirely — dark surround is intentional.
                # Keep only a small blown penalty for the light source itself.
                exp_qual     = float(np.clip(1.0 - abs(center_mean - 110) / 120.0, 0.0, 1.0))
                exposure_pen = min(blown * 2.0, 0.15)
            else:
                exposure_pen = min(blown * 4.0, 0.60) + min(blocked * 3.0, 0.40)
                exposure_pen = min(exposure_pen, 0.70)

            blur5    = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_map = np.abs(gray.astype(np.float32) - blur5.astype(np.float32))
            shadow_mask = gray < 55
            noise_level = float(noise_map[shadow_mask].mean()) if shadow_mask.any() else 0.0
            noise_pen   = min(noise_level / 12.0, 0.50)

            # intentional_soft = globally uniform softness + grain present
            # Thresholds: blur_cv < 0.55 (all regions similar) AND noise_level > 1.5
            # (grain/noise) AND image isn't completely unusable (some variance exists)
            intentional_soft = (
                best_sharp < 250          # soft enough to matter
                and blur_cv   < 0.55      # all regions similarly soft (not shake/miss)
                and noise_level > 1.5     # grain present (film / high-ISO)
                and best_sharp > 8        # not completely black / solid colour
            )

            # Sharpness score:
            # • Intentionally soft: floor at 0.38 — still lower than average but not
            #   treated as broken.  DINOv2 subject prominence already compensates for
            #   the apparent lack of classical sharpness.
            # • Accidentally blurred: hard floor at 0.18 (technically unusable).
            sharpness_score = min(best_sharp / 400.0, 1.0)
            if best_sharp < 60:
                if intentional_soft:
                    sharpness_score = max(sharpness_score, 0.38)
                else:
                    sharpness_score = min(sharpness_score, 0.18)  # hard floor: unusable

            tech = float(np.clip(
                sharpness_score
                * (1.0 - exposure_pen * 0.55)
                * (1.0 - noise_pen   * 0.35)
                * (0.80 + 0.20 * min(midtone_frac / 0.65, 1.0)),
                0.0, 1.0
            ))

            # ── COMPOSITION: DINOv2 patch-norm CV + rule-of-thirds ───────────
            comp_in = normalize_imagenet(bgr_to_chw(img, 224))[np.newaxis, ...]
            comp_out = self._ort_sessions["composition"].run(None, {self._comp_input: comp_in})[0]

            # Extract normalised CLS token now — reused for both composition scoring
            # and reference bank lookup later in the pipeline.
            _cls_f32  = comp_out[0, 0, :].astype(np.float32)
            _cls_f32 /= (np.linalg.norm(_cls_f32) + 1e-9)

            patches     = comp_out[0, 1:, :]           # (256, 384) — skip CLS token
            patch_norms = np.linalg.norm(patches, axis=1)  # (256,)
            dino_cv     = float(patch_norms.std() / (patch_norms.mean() + 1e-6))
            # CV ≈ 0.05 (flat/featureless) → 0.5+ (strong spatial hierarchy)
            dino_comp = float(np.clip(dino_cv / 0.40, 0.0, 1.0))

            # Subject prominence — ratio of peak patch activation to the batch mean.
            # A photo with a clear dominant subject has one or a few patches with much
            # higher activation than the rest. A static/empty scene has flat activations.
            # This is later combined with the Laplacian decisive proxy for a more
            # reliable "is there something happening here?" signal.
            _peak_norm       = float(patch_norms.max())
            _mean_norm       = float(patch_norms.mean()) + 1e-6
            # Scale: prominence=0 if all patches equal, ~1 if peak is 2.5× the mean
            subject_prominence = float(np.clip((_peak_norm / _mean_norm - 1.0) / 1.5, 0.0, 1.0))

            # Rule-of-thirds energy as secondary signal
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad = np.sqrt(gx**2 + gy**2)
            gmean = float(grad.mean()) + 1e-6
            margin = max(h // 12, 6)
            rot_vals = [
                float(grad[max(0,ty-margin):min(h,ty+margin),
                           max(0,tx-margin):min(w,tx+margin)].mean())
                for ty in [h//3, 2*h//3] for tx in [w//3, 2*w//3]
            ]
            rot_score = min(max(rot_vals) / (gmean * 1.8), 1.0)

            # Keep composition independent of technical quality — circular dependency
            # caused blurry shots to be double-penalized (once in tech, once here).
            # In complex/busy scenes (market, crowd) dino_cv is suppressed because
            # many patches activate equally. rot_score (rule-of-thirds energy at
            # subject positions) is more reliable in those cases, so we blend
            # adaptively: when dino_comp is weak, rot_score carries more weight.
            dino_weight = 0.55 + 0.20 * dino_comp   # 0.55–0.75 depending on confidence
            rot_weight  = 1.0 - dino_weight
            comp_score = float(np.clip(
                dino_weight * dino_comp + rot_weight * rot_score,
                0.0, 1.0
            ))

            # ── LIGHTING: exposure quality + contrast + mood ─────────────────
            mean_bright = float(gray.mean())
            if chiaroscuro:
                # exp_qual already set above using centre brightness — don't overwrite
                pass
            elif mean_bright < 40:
                exp_qual = (mean_bright / 40.0) * 0.45
            elif mean_bright > 210:
                exp_qual = max(0.0, (255 - mean_bright) / 45.0) * 0.45
            else:
                exp_qual = 1.0 - abs(mean_bright - 118) / 170.0

            contrast_score = min(float(np.std(gray)) / 52.0, 1.0)
            ps = max(h // 10, 16)
            rh, rw = (h // ps) * ps, (w // ps) * ps
            if rh > 0 and rw > 0:
                blocks = gray[:rh, :rw].reshape(rh // ps, ps, rw // ps, ps)
                local_contrast = min(float(blocks.std(axis=(1, 3)).mean()) / 38.0, 1.0)
            else:
                local_contrast = 0.4
            mood = float(np.clip(color_mood_score(img), 0.0, 1.0))
            light = float(np.clip(
                0.30 * exp_qual + 0.28 * contrast_score + 0.22 * local_contrast + 0.20 * mood,
                0.0, 1.0
            ))

            # ── AUTHENTICITY: MobileViT subject-clarity + decisive-moment ────
            rgb_feat = bgr_to_chw(img, 256)[np.newaxis, ...]
            est_out  = self._ort_sessions["aesthetic"].run(None, {self._est_input: rgb_feat})[0]
            logits   = est_out.flatten().astype(np.float64)
            softmax  = np.exp(logits - logits.max()); softmax /= softmax.sum()
            top3_mass = float(np.sort(softmax)[-3:].sum())
            clarity   = float(np.clip(top3_mass * 1.5, 0.0, 1.0))

            # Decisive-moment proxy — two signals combined:
            #
            # Signal A (Laplacian): centre zone sharper than periphery.
            #   Catches front-to-back sharpness falloff but is noisy — a sharp background
            #   or an intentionally blurred subject degrades this unfairly.
            cy1, cy2 = h//3, 2*h//3
            cx1, cx2 = w//4, 3*w//4
            ctr_v  = float(cv2.Laplacian(gray[cy1:cy2, cx1:cx2], cv2.CV_64F).var()) + 1.0
            strips = [gray[:h//6,:], gray[5*h//6:,:], gray[:,:w//8], gray[:,7*w//8:]]
            edg_v  = float(np.mean([float(cv2.Laplacian(s, cv2.CV_64F).var())
                                    if s.size > 0 else 0.0 for s in strips])) + 1.0
            lap_decisive = float(min(np.log1p(ctr_v / edg_v) / np.log1p(4.0), 1.0))

            # Signal B (DINOv2 subject prominence): computed above from patch norms.
            #   A genuine decisive moment has a clear spatial anchor — the subject
            #   activates far more strongly than the background.  A static, empty, or
            #   cluttered scene has flat activations.  This signal is more robust than
            #   the Laplacian ratio because DINOv2 encodes semantics, not raw sharpness.

            # Equal weight — each catches what the other misses.
            decisive = float(np.clip(0.50 * lap_decisive + 0.50 * subject_prominence, 0.0, 1.0))

            # Authenticity: decisive-moment is the dominant signal.
            # clarity (MobileViT top-3 mass) measures subject legibility, not action.
            # Weight 60/40 toward decisive — still decisive-first, but less brittle than
            # the previous 75/25 split that let a single noisy proxy flip the grade.
            auth = float(np.clip(0.40 * clarity + 0.60 * decisive, 0.0, 1.0))

            # ── HUMAN / CULTURAL PRESENCE ─────────────────────────────────────
            # YuNet (ResNet-based ONNX) when model present; Haar cascade fallback.
            # YuNet detects profiles, partial faces, and low-light faces that Haar misses.
            fd_scale = min(1.0, 320 / max(h, w, 1))
            fd_w, fd_h = max(1, int(w * fd_scale)), max(1, int(h * fd_scale))
            try:
                if self._face_yn is not None:
                    img_fd = cv2.resize(img, (fd_w, fd_h)) if fd_scale < 1.0 else img
                    self._face_yn.setInputSize((fd_w, fd_h))
                    _, faces = self._face_yn.detect(img_fd)
                    n_faces = 0 if faces is None else len(faces)
                else:
                    gray_fd = cv2.resize(gray, (fd_w, fd_h)) if fd_scale < 1.0 else gray
                    n_front   = len(self._face_casc.detectMultiScale(gray_fd, 1.25, 4, minSize=(14,14)))
                    n_profile = len(self._profile_casc.detectMultiScale(gray_fd, 1.25, 3, minSize=(14,14)))
                    n_faces   = n_front + n_profile
            except Exception:
                n_faces = 0
                
            # Add a buffer for human/culture scoring to prevent architectural/landscape shots from being nerfed
            human_buffer = 0.15  # Buffer value to ensure non-human shots don't get overly penalized
            
            if n_faces >= 1:
                # Faces are necessary but not sufficient. Top judges penalise
                # "faces with nothing happening" — require a decisive-moment signal
                # alongside face detection so a static portrait doesn't score as high
                # as a caught moment. Face count still matters but decisiveness carries 40%.
                face_weight = min(n_faces, 3) / 3.0   # 0.33 → 1.0
                human_env   = float(np.clip(
                    0.42 + 0.18 * face_weight + 0.40 * decisive, 0.0, 1.0))
            else:
                # No faces detected — covers: figures from behind, silhouettes,
                # gestures/pointing, profiles, hands, small figures (Hopper style).
                #
                # Key insight: a strong decisive signal (≥ 0.45) without face
                # detection almost always means active human interaction that the
                # Haar cascade missed — pointing, gesturing, turning, market
                # transactions. Treat decisive as a direct human-presence signal
                # in this path rather than ignoring it.
                decisive_contrib = 0.26 * max(0.0, decisive - 0.35) / 0.65 \
                                   if decisive > 0.35 else 0.0
                human_base = 0.10 + 0.16 * comp_score + 0.18 * subject_prominence \
                             + decisive_contrib
                # Cap: decisive shot without faces → same ceiling as 1-face shot
                human_cap  = (0.62 if decisive > 0.52
                              else 0.55 if subject_prominence > 0.55
                              else 0.46 if subject_prominence > 0.30
                              else 0.36)
                human_env  = float(np.clip(human_base, 0.0, human_cap))
                
            # Apply the human buffer to prevent architectural/landscape shots from being overly penalized
            human_env = float(np.clip(human_env + human_buffer, 0.0, 1.0))

            # ── STYLE-ADAPTIVE GRADING ────────────────────────────────────────
            style = self._detect_style_context({
                "Human/Culture": human_env,
                "Lighting":      light,
                "Mood/Color":    mood,
                "Technical":     tech,
                "Composition":   comp_score,
            })

            if style == "Documentary/Low-Key":
                tech_adj        = max(tech, 0.35)
                # contrast_score as floor so cold-toned / B&W shots aren't hurt by low warmth
                light_adj       = max(mood, contrast_score * 0.75)
                comp_adj        = 0.5 + (human_env * 0.5)
                narrative_boost = 0.18
            elif style == "Street/Environmental":
                tech_adj        = max(tech, 0.40)
                light_adj       = light
                comp_adj        = max(comp_score, 0.55)
                narrative_boost = 0.12
            elif style == "Night/Nocturnal":
                tech_adj        = max(tech, 0.30)
                light_adj       = float(np.clip(mood * 1.1, 0.0, 1.0))
                comp_adj        = comp_score
                narrative_boost = 0.15
            elif style == "Commercial/Studio":
                tech_adj        = max(tech, 0.55)   # clean execution required
                light_adj       = light
                comp_adj        = max(comp_score, 0.50)
                narrative_boost = 0.03
            else:
                tech_adj        = tech
                light_adj       = light
                comp_adj        = comp_score
                narrative_boost = 0.05

            # For non-warm styles, use contrast as mood floor so cold-toned / blue-hour
            # shots aren't penalised by the warmth-biased color_mood_score.
            if style in ("Night/Nocturnal", "Documentary/Low-Key", "Street/Environmental"):
                mood_for_narrative = max(mood, contrast_score * 0.65)
            else:
                mood_for_narrative = mood

            narrative_score = float(np.clip(
                0.6 * human_env + 0.3 * mood_for_narrative + 0.1 * light_adj, 0.0, 1.0
            ))

            _style_presets = {
                "Classic Street":             {"tech": 0.10, "comp": 0.18, "light": 0.15, "narrative": 0.32, "human": 0.25},
                "Travel Editor":               {"tech": 0.12, "comp": 0.13, "light": 0.18, "narrative": 0.27, "human": 0.30},
                "Photojournalism":             {"tech": 0.13, "comp": 0.13, "light": 0.14, "narrative": 0.40, "human": 0.20},
                "Cinematic/Editorial":         {"tech": 0.10, "comp": 0.20, "light": 0.30, "narrative": 0.20, "human": 0.20},
                "Fine Art/Contemporary":       {"tech": 0.10, "comp": 0.25, "light": 0.20, "narrative": 0.15, "human": 0.30},
                "Minimalist/Urbex":            {"tech": 0.15, "comp": 0.30, "light": 0.20, "narrative": 0.10, "human": 0.25},
                "LSPF (London Street)":        {"tech": 0.07, "comp": 0.17, "light": 0.25, "narrative": 0.26, "human": 0.25},
                "Snapshot / Point-and-Shoot":  {"tech": 0.08, "comp": 0.12, "light": 0.15, "narrative": 0.40, "human": 0.25},
                "Landscape with Elements":     {"tech": 0.18, "comp": 0.32, "light": 0.35, "narrative": 0.05, "human": 0.10},
            }
            
            # Niche-to-preset mapping for adaptive grading
            _NICHE_PRESETS = {
                "Portrait/People":    "Fine Art/Contemporary",  # Human-focused, high human weight
                "Portrait":           "Fine Art/Contemporary",
                "Wedding/Event":      "Travel Editor",          # Human + light + comp
                "Street/Urban":       "Classic Street",        # Narrative + human
                "Travel/Documentary": "Travel Editor",          # Human + narrative + light
                "Architecture":       "Minimalist/Urbex",       # Comp + tech
                "Real Estate":        "Cinematic/Editorial",    # Comp + light
                "Food/Culinary":      "Cinematic/Editorial",    # Tech + light
                "Product/Commercial": "Cinematic/Editorial",    # Tech + comp
                "Landscape/Nature":   "Landscape with Elements", # Light + comp
                "Nature/Landscape":   "Landscape with Elements",
                "Night/Nocturnal":    "Snapshot / Point-and-Shoot", # Narrative + human
                "Sports/Action":      "Photojournalism",        # Narrative + human
                "Macro/Detail":       "Minimalist/Urbex",       # Comp + tech
                "Fine Art":           "Fine Art/Contemporary",  # Narrative + human
                "Documentary/Low-Key":"Photojournalism",        # Narrative focused
                "Street/Environmental":"Classic Street",       # Narrative + human
            }
            
            # Build breakdown dictionary for grading
            bd = {
                "Technical":     round(float(tech_adj),        2),
                "Composition":   round(float(comp_adj),         2),
                "Lighting":      round(float(light_adj),        2),
                "Mood/Color":    round(float(mood),             2),
                "Narrative":     round(float(narrative_score),  2),
                "Human/Culture": round(float(human_env),        2),
            }
            
            # Detect niche from breakdown
            detected_niche, _ = classify_with_fallback("", 0.0, bd)
            bd["Detected_Niche"] = detected_niche
            
            # Use detected niche for adaptive preset
            adaptive_preset = _NICHE_PRESETS.get(detected_niche, preset)
            pw = _style_presets.get(adaptive_preset, _style_presets["Classic Street"])
            
            # Calculate score with median niche preset (kept for tracking)
            raw_median = (pw["tech"] * tech_adj + pw["comp"] * comp_adj + pw["light"] * light_adj
                          + pw["narrative"] * narrative_score + pw["human"] * human_env
                          + narrative_boost)

            # Find best preset for tracking purposes
            best_preset, best_raw = self._find_best_preset(bd)

            # Track both scores for transparency
            median_score = float(np.clip(1.0 - (1.0 - raw_median) ** 1.2, 0.0, 1.0))
            best_score = float(np.clip(1.0 - (1.0 - best_raw) ** 1.2, 0.0, 1.0))

            # Final score = average of the 5 independent base categories
            # (Narrative excluded — derived from other dims, would double-count)
            # 0.90 deflation factor reduces strong bias by ~10%
            raw = (tech_adj + comp_adj + light_adj + mood + human_env) / 5.0 * 0.90

            if best_sharp < 60 and not intentional_soft:
                raw = min(raw, 0.28)
            if blown > 0.25 or (blocked > 0.35 and not chiaroscuro):
                raw *= 0.72

            # ── NIMA nudge (AVA-trained aesthetic signal) ─────────────────────
            nima_score = None
            if self._nima_input is not None:
                try:
                    nima_out = self._ort_sessions["nima"].run(
                        None, {self._nima_input: comp_in}
                    )[0]
                    probs = nima_out[0].astype(np.float64)
                    probs = np.exp(probs - probs.max()); probs /= probs.sum()
                    mean_rating = float(np.dot(probs, np.arange(1, 11)))
                    nima_score  = (mean_rating - 1.0) / 9.0
                    raw = float(np.clip(
                        raw + np.clip((nima_score - 0.50) * 0.22, -0.08, 0.10),
                        0.0, 1.0
                    ))
                except Exception:
                    nima_score = None

            # ── Reference bank nudge (curated exemplar similarity) ────────────
            if nima_score is None:
                ref_sim = self._ref_bank.score(_cls_f32)
                if ref_sim is not None:
                    raw = float(np.clip(
                        raw + np.clip((ref_sim - 0.65) * 0.30, -0.05, 0.06),
                        0.0, 1.0
                    ))

            final = float(np.clip(1.0 - (1.0 - raw) ** 1.2, 0.0, 1.0))

            if style == "Documentary/Low-Key" and final > 0.59:
                grade   = "Strong \u2705"
                critique = ("Compelling documentary moment. Intentional low-key lighting "
                            "& environmental context drive narrative weight.")
            elif final > 0.59:
                grade   = "Strong \u2705"
                critique = "Peak timing & confident framing. Clear visual hierarchy."
            elif final > 0.40:
                grade   = "Mid \u26a0\ufe0f"
                critique = "Solid moment; slight repositioning or light direction could sharpen the story."
            else:
                grade   = "Weak \u274c"
                critique = "Lacks focal clarity or decisive gesture."

            issues = []
            if best_sharp < 150 and not intentional_soft:
                issues.append("focus or motion blur degrading sharpness")
            elif intentional_soft and best_sharp < 150:
                issues.append("globally soft rendering — consistent with vintage / old glass")
            if blown > 0.12:
                issues.append("highlights clipping")
            if blocked > 0.20 and not chiaroscuro:
                issues.append("shadows crushed")
            if comp_adj < 0.30:
                issues.append("weak compositional structure")
            if human_env < 0.20:
                issues.append("no discernible human or subject presence")
            if issues:
                critique += " Issues: " + "; ".join(issues) + "."

            # Update breakdown with additional tracking fields
            bd["Style Context"] = style
            bd["Applied_Preset"] = adaptive_preset  # Track which preset was used for median niche grading
            bd["Best_Preset"] = best_preset  # Track the preset that gave the strongest score
            bd["Median_Score"] = round(median_score, 3)  # Score with median niche preset
            bd["Best_Score"] = round(best_score, 3)  # Score with optimal preset
            # Detected_Niche is already set from earlier in the function

            embedding = _cls_f32.astype(np.float64).tolist()

            exif_ts: float | None = None
            try:
                from PIL import Image as _PILImage
                from PIL.ExifTags import TAGS as _TAGS
                with _PILImage.open(path) as _im:
                    _exif = _im._getexif()
                if _exif:
                    for _tag_id, _val in _exif.items():
                        if _TAGS.get(_tag_id) in ("DateTimeOriginal", "DateTime"):
                            import time as _time
                            exif_ts = float(_time.mktime(
                                _time.strptime(str(_val), "%Y:%m:%d %H:%M:%S")))
                            break
            except Exception:
                pass

            result = {"score": round(final, 3), "grade": grade,
                    "human_perception": round(top3_mass, 3),
                    "nima_score": round(nima_score, 3) if nima_score is not None else None,
                    "critique": critique, "dims": (int(w), int(h)), "faces": int(n_faces),
                    "breakdown": bd, "embedding": embedding,
                    "exif_ts": exif_ts}
            return result
        except Exception as e:
            return self._fallback(str(e))

    def _fallback(self, err):
        return {"score": 0.0, "grade": "Error \u274c", "human_perception": 0.0, "critique": err,
                "breakdown": {"Technical": 0, "Composition": 0, "Lighting": 0,
                               "Authenticity": 0, "Human/Culture": 0},
                "dims": (0, 0), "faces": 0, "embedding": [0.0] * 384}

    def _find_best_preset(self, breakdown: dict) -> tuple[str, float]:
        """
        Find the best preset for an image based on its breakdown scores.
        Returns (preset_name, score) for the optimal preset.
        
        This enables "strongest score" grading - the system evaluates all presets
        and returns the one that gives the highest score for this specific image.
        """
        if not breakdown:
            return "Classic Street", 0.0
        
        # Extract scores from breakdown (handle label variations)
        def _get(b, keys):
            return next((v for k, v in b.items() if k in keys), 0.0)
        
        tech = _get(breakdown, self._TECH_KEYS)
        comp = _get(breakdown, self._COMP_KEYS)
        light = _get(breakdown, self._LIGHT_KEYS)
        human = _get(breakdown, self._HUMAN_KEYS)
        narrative = _get(breakdown, self._AUTH_KEYS)
        
        # Calculate score for each preset
        best_preset = "Classic Street"
        best_score = 0.0
        
        for preset_name, weights in self._STYLE_PRESETS.items():
            score = (
                weights["tech"] * tech +
                weights["comp"] * comp +
                weights["light"] * light +
                weights["narrative"] * narrative +
                weights["human"] * human
            )
            if score > best_score:
                best_score = score
                best_preset = preset_name
        
        return best_preset, best_score

    # Breakdown keys that map to the "tech" dimension, regardless of preset label
    _TECH_KEYS = frozenset({
        "Technical", "News Sharpness", "Cleanliness", "Execution",
        "Detail Retention", "Exposure", "Sharpness & Detail",
    })

    _COMP_KEYS = frozenset({
        "Composition","Framing","Context","Geometry & Balance",
        "Negative Space","Framing Instinct","Layered Depth",
    })
    _AUTH_KEYS = frozenset({
        "Decisive Moment","Cultural Depth","Journalistic Integrity",
        "Narrative Suggestion","Conceptual Weight","Reduction",
        "Authenticity","Immediacy","Environmental Truth",
        "Narrative",
    })
    _HUMAN_KEYS = frozenset({
        "Human/Culture","Sense of Place","Human Impact",
        "Character Presence","Emotional Resonance","Scale Element",
        "Presence","Scale & Life",
        "Subject Isolation",   # Classic Street preset label
    })
    _LIGHT_KEYS = frozenset({
        "Lighting","Atmosphere","Natural Light","Mood & Tone",
        "Tonal Purity","Contrast Purity","Available Light",
        "Natural Light Quality",
    })

    # Per-type role constraints for each of the 5 narrative slots.
    # "trait": callable(hv, lv, cv, tv) → bool — True means the photo satisfies
    #          the role; False triggers the penalty subtraction from role_score.
    # "penalty": magnitude subtracted when the trait check fails (positive float).
    # None trait = role is enforced by the transition matrix, not the photo itself.
    _ROLE_REQUIREMENTS: dict = {
        "street": [
            {"trait": lambda hv, lv, cv, tv, av: cv >= 0.40,              "penalty": 0.15},  # Establishing: wide comp
            {"trait": lambda hv, lv, cv, tv, av: hv >= 0.35 or av >= 0.50,"penalty": 0.20},  # Moment Anchor: human OR decisive gesture
            {"trait": lambda hv, lv, cv, tv, av: tv >= 0.38,              "penalty": 0.10},  # Detail: sharpness
            {"trait": None,                                                 "penalty": 0.15},  # Contrast: matrix-driven
            {"trait": lambda hv, lv, cv, tv, av: lv >= 0.40,              "penalty": 0.10},  # Resolution: light
        ],
        "nature": [
            {"trait": lambda hv, lv, cv, tv, av: lv >= 0.45,                "penalty": 0.15},  # Scene Opener: light is required — no pure-comp bypass
            {"trait": lambda hv, lv, cv, tv, av: lv >= 0.42 or cv >= 0.50, "penalty": 0.15},  # Landscape Anchor: light preferred; comp fallback only if strong
            {"trait": lambda hv, lv, cv, tv, av: tv >= 0.40 or av >= 0.45, "penalty": 0.10},  # Detail / Wildlife: sharpness or decisiveness
            {"trait": lambda hv, lv, cv, tv, av: lv >= 0.42,                "penalty": 0.15},  # Mood & Atmosphere: pure light
            {"trait": lambda hv, lv, cv, tv, av: lv >= 0.36 or cv >= 0.44, "penalty": 0.10},  # Quiet Close
        ],
        "portrait": [
            {"trait": lambda hv, lv, cv, tv, av: hv >= 0.42,              "penalty": 0.15},  # Subject intro
            {"trait": lambda hv, lv, cv, tv, av: hv >= 0.52 or av >= 0.55,"penalty": 0.20},  # Eye contact or unguarded
            {"trait": lambda hv, lv, cv, tv, av: cv >= 0.40,              "penalty": 0.10},  # Environment context
            {"trait": lambda hv, lv, cv, tv, av: hv >= 0.38 or av >= 0.48,"penalty": 0.15},  # Unguarded moment
            {"trait": lambda hv, lv, cv, tv, av: lv >= 0.40,              "penalty": 0.10},  # Defining frame
        ],
        "architecture": [
            {"trait": lambda hv, lv, cv, tv, av: cv >= 0.50,              "penalty": 0.15},  # Facade & scale
            {"trait": lambda hv, lv, cv, tv, av: tv >= 0.45 or cv >= 0.52,"penalty": 0.15},  # Geometric detail or strong comp
            {"trait": lambda hv, lv, cv, tv, av: lv >= 0.46 or av >= 0.50,"penalty": 0.15},  # Light/shadow or decisive geometric moment
            {"trait": lambda hv, lv, cv, tv, av: hv >= 0.15 or av >= 0.30,"penalty": 0.10},  # Human scale — any human presence counts
            {"trait": lambda hv, lv, cv, tv, av: cv >= 0.45,              "penalty": 0.10},  # Abstract close
        ],
    }
    # Fallback for "auto" — use street defaults
    _ROLE_REQUIREMENTS["auto"] = _ROLE_REQUIREMENTS["street"]

    def _apply_batch_grades(self, results):
        """
        Post-processing pass over a completed batch.

        Design philosophy (revised):
        The per-photo sigmoid score from _analyze already encodes all quality
        signals — the compound gate in the previous version added hard binary
        cliffs based on individual noisy metrics, causing bidirectional errors
        (strong shots demoted, weak shots promoted) depending on which proxy
        happened to land just above or below a threshold.

        This pass now does only two things:
          1. Hard floor: if the image is technically unusable (severe blur,
             clipping, or noise) it is always Weak regardless of score.
          2. Critique text is updated to match the final grade so the UI
             doesn't show a "Weak" critique for a Strong photo.

        Grade boundaries are set by the per-photo sigmoid in _analyze
        (Strong > 0.65, Mid > 0.38) and are not overridden here.
        """
        if not results: return

        for r in results:
            bd       = r.get("breakdown", {})
            tech_val = next((v for k, v in bd.items() if k in self._TECH_KEYS), None)

            # Only hard override: technically unusable → always Weak.
            # Note: intentionally-soft vintage shots are exempt because _analyze
            # raises their sharpness_score floor to 0.38, producing tech > 0.18.
            # Only accidental blur (camera shake, missed focus) trips this gate.
            if tech_val is not None and tech_val < 0.18:
                r["grade"]    = "Weak \u274c"
                r["critique"] = "Technically unusable — severe blur, clipping, or noise."
                continue

            # Sync critique text to grade — check for weak-sounding language in any grade band
            _c = r.get("critique", "").lower()
            _weak_words = ("lacks", "weak", "missing", "unusable", "no discernible")
            if "Strong" in r["grade"] and any(w in _c for w in _weak_words):
                r["critique"] = "Peak timing & confident framing. Clear visual hierarchy."
            elif "Mid" in r["grade"] and any(w in _c for w in ("technically unusable", "lacks focal clarity")):
                r["critique"] = "Solid moment; slight repositioning or light direction could sharpen the story."

    def _detect_similar_shots(self, results, sim_threshold=0.78):
        """
        Cluster near-duplicate shots using DINOv2 semantic embeddings.
        Union-find handles transitive similarity (A~B and B~C → same cluster).
        """
        n = len(results)
        for r in results:
            r.setdefault("sim_flag",   "")
            r.setdefault("cluster_id", -1)
        if n < 2:
            return

        raw_embs = [r.get("embedding", None) for r in results]
        # Skip photos whose embeddings are missing or all-zero (ONNX failure)
        valid_mask = [
            e is not None and len(e) > 0 and np.linalg.norm(e) > 1e-6
            for e in raw_embs
        ]
        if sum(valid_mask) < 2:
            import sys
            print(f"[sim] only {sum(valid_mask)}/{n} valid embeddings — skipping", file=sys.stderr)
            return

        embs = np.array(
            [e if v else [0.0]*len(raw_embs[next(i for i,v in enumerate(valid_mask) if v)])
             for e, v in zip(raw_embs, valid_mask)],
            dtype=np.float32
        )
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs  = embs / (norms + 1e-9)
        sims  = (embs @ embs.T)

        # Mask out invalid rows/cols so they never cluster
        for i, v in enumerate(valid_mask):
            if not v:
                sims[i, :] = 0.0
                sims[:, i] = 0.0

        # Log similarity stats to crash.log once per grade run
        import sys
        upper = sims[np.triu_indices(n, k=1)]
        pairs_above = int((upper > sim_threshold).sum())
        print(f"[sim] n={n} valid={sum(valid_mask)} max_sim={float(upper.max()):.3f} "
              f"pairs>{sim_threshold}={pairs_above}", file=sys.stderr)

        # Union-find with path compression
        parent = list(range(n))
        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def _union(x, y):
            parent[_find(x)] = _find(y)

        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] > sim_threshold:
                    _union(i, j)

        # Group indices by cluster root
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for i in range(n):
            groups[_find(i)].append(i)

        def _dim_val(r, keys):
            b = r.get("breakdown", {})
            return next((v for k, v in b.items() if k in keys), 0.0)

        def _best_rank(r):
            # Multi-dimensional ranking: overall score + weighted tech/auth/comp
            # Tech matters most in burst selection (sharpness varies most shot-to-shot);
            # auth catches peak decisive moment; comp rewards the better reframe.
            return (
                0.40 * r.get("score", 0.0)
              + 0.30 * _dim_val(r, self._TECH_KEYS)
              + 0.20 * _dim_val(r, self._AUTH_KEYS)
              + 0.10 * _dim_val(r, self._COMP_KEYS)
            )

        def _why_better(best_r, other_r):
            """Return a short human-readable reason why best_r beats other_r."""
            bt = _dim_val(best_r, self._TECH_KEYS)
            ot = _dim_val(other_r, self._TECH_KEYS)
            ba = _dim_val(best_r, self._AUTH_KEYS)
            oa = _dim_val(other_r, self._AUTH_KEYS)
            bc = _dim_val(best_r, self._COMP_KEYS)
            oc = _dim_val(other_r, self._COMP_KEYS)

            score_diff = best_r.get("score", 0) - other_r.get("score", 0)
            if score_diff < 0.02:
                return "marginal edge — keep both for review"
            if bt - ot >= 0.08:
                return "sharper focus / cleaner exposure"
            if ba - oa >= 0.08:
                return "stronger decisive moment"
            if bc - oc >= 0.08:
                return "better framing"
            return f"higher overall score (+{score_diff:.2f})"

        cid = 0
        for root, members in groups.items():
            if len(members) < 2:
                continue   # unique shot — keep defaults
            members.sort(key=lambda i: _best_rank(results[i]), reverse=True)
            best_idx = members[0]
            best_r   = results[best_idx]
            best_name = os.path.basename(best_r.get("path", "best"))

            for rank, idx in enumerate(members):
                results[idx]["cluster_id"] = cid
                if rank == 0:
                    results[idx]["sim_flag"] = (
                        f"\u2605 Best of {len(members)} similar shots "
                        f"(score {best_r['score']:.2f})"
                    )
                else:
                    reason = _why_better(best_r, results[idx])
                    results[idx]["sim_flag"] = (
                        f"\U0001f501 Duplicate \u2014 {best_name} is better: {reason}"
                    )
            cid += 1

    def _detect_median_niche(self, results):
        """
        Detect the median niche across all images in a folder.
        Returns the most common niche detected.
        """
        if not results:
            return "Street/Urban"
        
        niche_counts = {}
        for r in results:
            b = r.get("breakdown", {})
            niche = b.get("Detected_Niche", "")
            if niche:
                niche_counts[niche] = niche_counts.get(niche, 0) + 1
        
        if not niche_counts:
            return "Street/Urban"
        
        # Return the most common niche (mode)
        return max(niche_counts.items(), key=lambda x: x[1])[0]

    def analyze_folder(self, folder_path, preset=None, progress=None, force_rescan=True):
        """
        Analyze a folder of images with automatic median niche detection.
        
        Args:
            folder_path: Path to folder containing images
            preset: Optional preset override (if None, uses median niche)
            progress: Progress callback
            force_rescan: Whether to re-scan all images
        """
        folder = Path(folder_path)
        if not folder.is_dir(): return []
        exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff",
                ".JPG", ".JPEG", ".PNG", ".WEBP", ".raw", ".cr2", ".nef", ".arw"}
        all_paths = [str(p) for p in folder.rglob("*") if p.suffix.lower() in exts]
        if not all_paths: return []

        # _analyze always picks the best-fitting preset internally via _find_best_preset,
        # so the preset parameter only affects the fallback niche label — not the score.
        # A separate first pass to detect the median niche is therefore redundant.
        effective_preset = preset or "Classic Street"

        # Grade with the selected preset
        new = all_paths if force_rescan else [p for p in all_paths if p not in self.cache]
        if not new:
            pass
        else:
            if progress: progress(0, desc=f"Scanning {len(new)} images...")
            self._ensure_sessions()           # warm up ONNX before spawning threads
            lock      = threading.Lock()
            completed = [0]

            def _run(p):
                result = self._analyze(p, effective_preset)
                with lock:
                    completed[0] += 1
                    done = completed[0]
                if progress:
                    progress(done / len(new), desc=f"Grading: {done}/{len(new)}")
                return p, result

            workers = min(os.cpu_count() or 2, 8)
            BATCH = 50
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for i, (p, result) in enumerate(ex.map(_run, new)):
                    with lock:
                        self.cache[p] = result
                    if i % BATCH == BATCH - 1:
                        gc.collect()

        batch_results = [{"path": p, **self.cache[p]} for p in all_paths]
        self._apply_batch_grades(batch_results)
        self._detect_similar_shots(batch_results)

        for g in batch_results:
            p = g.pop("path")
            self.cache[p].update(g)
        self._save_cache()
        self._build_niche_anchors()
        self._apply_niche_sweep(all_paths)
        
        # Detect median niche and apply to all images
        median_niche = self._detect_median_niche(batch_results)
        for p in all_paths:
            if p in self.cache:
                self.cache[p]["breakdown"]["Median_Niche"] = median_niche
        
        return [(p, self.cache[p]) for p in all_paths]

    def detect_subject_type(self, photos):
        """
        Infer the primary subject type from average breakdown scores.
        Returns one of: 'street', 'nature', 'portrait', 'architecture'

        Uses key-set lookups (same as _role_score) so preset-specific labels
        like 'Subject Isolation' and 'Decisive Moment' resolve correctly instead
        of always returning 0 for human and auth dimensions.
        """
        if not photos:
            return "street"

        def _dv(b, keys):
            return next((v for k, v in b.items() if k in keys), None)

        t_total = c_total = l_total = a_total = h_total = 0.0
        count = 0
        for p in photos:
            b = p.get("breakdown", {}) if isinstance(p, dict) else p[1].get("breakdown", {})
            tv = _dv(b, self._TECH_KEYS)
            cv = _dv(b, self._COMP_KEYS)
            lv = _dv(b, self._LIGHT_KEYS)
            av = _dv(b, self._AUTH_KEYS)
            hv = _dv(b, self._HUMAN_KEYS)
            if any(v is not None for v in (tv, cv, lv, av, hv)):
                t_total += tv or 0.0
                c_total += cv or 0.0
                l_total += lv or 0.0
                a_total += av or 0.0
                h_total += hv or 0.0
                count += 1

        if count == 0:
            return "street"

        tech  = t_total / count
        comp  = c_total / count
        light = l_total / count
        auth  = a_total / count
        human = h_total / count

        # Each archetype scores on its key signals and penalises contradicting ones.
        # Nature and architecture now lose points for strong human presence so a
        # street folder doesn't bleed into them just because comp/light/tech are decent.
        # Nature is light-dominant: penalise when comp overtakes light (looks like architecture).
        # Architecture requires both high comp AND high tech together (multiplicative) —
        # buildings produce geometric precision AND clinical sharpness simultaneously.
        # This separates them from landscapes where comp may be moderate but light is rich.
        scores = {
            "street":       human * 0.45 + auth  * 0.38 + comp  * 0.17,
            "nature":       light * 0.65 + comp  * 0.20 + tech  * 0.15
                            - max(0.0, comp  - light) * 0.80
                            - max(0.0, human - 0.20)  * 0.80,
            "portrait":     human * 0.65 + light * 0.20 + auth  * 0.15,
            "architecture": (comp * tech) * 0.70 + light * 0.10 + comp * 0.20
                            - max(0.0, human - 0.45) * 0.55,
        }

        _NICHE_TYPE = {
            "Street/Urban": "street",        "Travel/Documentary": "street",
            "Documentary/Low-Key": "street", "Street/Environmental": "street",
            "Portrait/People": "portrait",   "Portrait": "portrait",
            "Wedding/Event": "portrait",     "Boudoir/Intimate": "portrait",
            "Architecture": "architecture",  "Real Estate": "architecture",
            "Drone/Aerial": "architecture",
            "Nature/Landscape": "nature",    "Landscape/Nature": "nature",
            "Wildlife": "nature",
            "Night/Nocturnal": "nocturnal",  "Concert/Live Music": "nocturnal",
        }
        niche_votes: dict = {}
        n_with_niche = 0
        for p in photos:
            b = p.get("breakdown", {}) if isinstance(p, dict) else p[1].get("breakdown", {})
            mapped = _NICHE_TYPE.get(b.get("Detected_Niche", ""))
            if mapped:
                niche_votes[mapped] = niche_votes.get(mapped, 0) + 1
                n_with_niche += 1
        if niche_votes and n_with_niche > 0:
            top, top_count = max(niche_votes.items(), key=lambda kv: kv[1])
            if top_count / n_with_niche >= 0.40:
                return top

        return max(scores, key=scores.get)

    # Maps subject type → pacing preset name (fallback: "Classic Street")
    _PACING_MAP = {
        "street":       "Classic Street",
        "nature":       "Travel / Documentary",
        "portrait":     "Travel / Documentary",
        "architecture": "Minimalist / Art",
        "nocturnal":    "Cinematic/Editorial",
    }

    def _ensure_vlm(self):
        if self._vlm_instance is None:
            try:
                from vlm_niche_detector import VLMNicheDetector
                self._vlm_instance = VLMNicheDetector()
            except Exception:
                self._vlm_instance = None
        return self._vlm_instance

    def _apply_niche_sweep(self, paths):
        vlm = self._ensure_vlm()
        if vlm:
            vlm_cache = asyncio.run(vlm.classify_batch(paths))

            for p in paths:
                bd = self.cache.get(p, {}).get("breakdown", {})
                if not bd:
                    continue
                hit = vlm_cache.get(p, {})
                niche, source = classify_with_fallback(
                    hit.get("niche", ""), hit.get("confidence", 0.0), bd
                )
                bd["Detected_Niche"] = niche
                bd["Niche_Source"]   = source
        self._save_cache()

    def _detect_top_niches(self, results, top_n=5):
        """
        Score-based niche detector.

        Six signals per photo:
          c = comp   t = tech   h = human   l = light   a = auth/mood
          f = face count  (face cascade — 0 when no faces detected)

        Key discriminators:
          Portrait vs Street  → Portrait requires t >= a−0.12 (deliberate focus, not pure-candid)
          Architecture vs Nature → Architecture: c >= l;  Nature/Landscape: l >= c
          Street vs Travel    → Travel also requires l > 0.33 (environmental light matters)
          Thresholds are lower than before so fewer photos fall to General/Mixed.
        """
        def _dv(b, keys):
            return next((v for k, v in b.items() if k in keys), 0.0)

        # (name, score_fn(c,t,h,l,a,f) → float)
        niche_scorers = [
            # ── People-dominant ───────────────────────────────────────────────
            # Wedding: high human + event-quality light; f boosts when group detected
            ("Wedding/Event",
                lambda c,t,h,l,a,f: h*0.48 + l*0.28 + c*0.24
                    if h > 0.58 and l > 0.35 else 0.0),
            # Portrait: deliberate subject focus — tech must not be far below auth
            # (gap > 0.12 means the shot is candid, not posed/controlled → Street instead)
            ("Portrait",
                lambda c,t,h,l,a,f: h*0.48 + t*0.30 + l*0.22
                    if h > 0.48 and t > 0.38 and t >= a - 0.12 else 0.0),
            # Boudoir: human + warm soft light + intimate mood
            ("Boudoir/Intimate",
                lambda c,t,h,l,a,f: h*0.40 + l*0.36 + a*0.24
                    if h > 0.50 and l > 0.44 and a > 0.42 else 0.0),
            # Sports/Action: very high tech (motion freeze) + human + decisive
            ("Sports/Action",
                lambda c,t,h,l,a,f: t*0.50 + a*0.28 + h*0.22
                    if t > 0.62 and h > 0.36 else 0.0),
            # Concert: human + auth + low light (dark venue signature)
            ("Concert/Live Music",
                lambda c,t,h,l,a,f: h*0.44 + a*0.38 + (1.0 - l)*0.18
                    if h > 0.44 and l < 0.40 else 0.0),
            # ── Candid / street ───────────────────────────────────────────────
            # Street/Urban: auth-dominant decisive moment; auth must not lag tech by >0.08
            # (if t >> a the shot is posed/controlled → Portrait)
            ("Street/Urban",
                lambda c,t,h,l,a,f: a*0.46 + h*0.30 + c*0.24
                    if a > 0.36 and h > 0.25 and a >= t - 0.08 else 0.0),
            # Travel/Documentary: balanced human + auth + light (environment matters)
            ("Travel/Documentary",
                lambda c,t,h,l,a,f: h*0.34 + a*0.34 + l*0.32
                    if h > 0.32 and a > 0.36 and l > 0.33 else 0.0),
            # ── Architecture / structural ─────────────────────────────────────
            # Product/Commercial: extremely high tech+comp, virtually no human
            ("Product/Commercial",
                lambda c,t,h,l,a,f: t*0.52 + c*0.34 + l*0.14
                    if t > 0.70 and c > 0.58 and h < 0.18 else 0.0),
            # Automotive: high tech+comp, no human (vehicle as subject)
            ("Automotive",
                lambda c,t,h,l,a,f: t*0.46 + c*0.38 + l*0.16
                    if t > 0.62 and c > 0.54 and h < 0.22 else 0.0),
            # Architecture: comp-dominant AND comp >= light (not a landscape)
            ("Architecture",
                lambda c,t,h,l,a,f: c*0.56 + t*0.26 + l*0.18
                    if c > 0.55 and h < 0.28 and c >= l else 0.0),
            # Real Estate: comp + strong even light + very low human
            ("Real Estate",
                lambda c,t,h,l,a,f: c*0.38 + l*0.44 + t*0.18
                    if c > 0.48 and l > 0.48 and h < 0.22 else 0.0),
            # Drone/Aerial: comp + light + very low human (DINOv2 handles actual aerial pattern)
            ("Drone/Aerial",
                lambda c,t,h,l,a,f: c*0.46 + l*0.34 + t*0.20
                    if c > 0.54 and h < 0.22 and l > 0.42 else 0.0),
            # ── Nature / environment ──────────────────────────────────────────
            # Wildlife: high tech (telephoto/freeze) + low human
            ("Wildlife",
                lambda c,t,h,l,a,f: t*0.52 + a*0.28 + c*0.20
                    if t > 0.58 and h < 0.18 else 0.0),
            # Nature/Landscape: light-dominant AND light >= comp (not architecture)
            ("Nature/Landscape",
                lambda c,t,h,l,a,f: l*0.56 + c*0.26 + a*0.18
                    if l > 0.38 and h < 0.28 and l >= c else 0.0),
            # ── Low-light / mood ──────────────────────────────────────────────
            # Night: low luminance scene + strong auth/atmosphere
            ("Night/Nocturnal",
                lambda c,t,h,l,a,f: a*0.52 + c*0.28 + (1.0 - l)*0.20
                    if l < 0.38 and a > 0.46 else 0.0),
            # ── Detail / fine art ─────────────────────────────────────────────
            # Macro/Detail: very high comp + very high tech + no human
            ("Macro/Detail",
                lambda c,t,h,l,a,f: c*0.46 + t*0.42 + (1.0 - h)*0.12
                    if c > 0.60 and t > 0.56 and h < 0.18 else 0.0),
            # Food/Culinary: tech + warm light + comp + no human
            ("Food/Culinary",
                lambda c,t,h,l,a,f: t*0.38 + l*0.40 + c*0.22
                    if t > 0.50 and l > 0.48 and h < 0.22 else 0.0),
            # Abstract/Texture: very high comp + auth + no human
            ("Abstract/Texture",
                lambda c,t,h,l,a,f: c*0.48 + a*0.34 + (1.0 - h)*0.18
                    if c > 0.60 and h < 0.20 else 0.0),
            # Fine Art: auth + comp — conceptual/staged, distinct from candid Street
            ("Fine Art",
                lambda c,t,h,l,a,f: a*0.44 + c*0.36 + l*0.20
                    if a > 0.50 and c > 0.46 else 0.0),
            # ── Fallback ──────────────────────────────────────────────────────
            ("General/Mixed",
                lambda c,t,h,l,a,f: 0.28),
        ]

        # NicheClassifier visual-prototype name → _detect_top_niches display name
        _CLF_MAP = {
            "Portrait/People":    "Portrait",
            "Street/Urban":       "Street/Urban",
            "Travel/Tourism":     "Travel/Documentary",
            "Architecture":       "Architecture",
            "Real Estate":        "Real Estate",
            "Food/Culinary":      "Food/Culinary",
            "Product/Commercial": "Product/Commercial",
            "Night/Nocturnal":    "Night/Nocturnal",
            "Landscape/Nature":   "Nature/Landscape",
            "Wedding/Event":      "Wedding/Event",
            "Sports/Action":      "Sports/Action",
            "Macro/Detail":       "Macro/Detail",
            "General/Mixed":      "General/Mixed",
        }
        clf    = self._niche_clf
        clf_ok = clf is not None and bool(getattr(clf, "_anchors", {}))

        counts = {name: 0 for name, _ in niche_scorers}
        for path, data in results:
            b = data.get("breakdown", {})
            c = _dv(b, self._COMP_KEYS)
            t = _dv(b, self._TECH_KEYS)
            h = _dv(b, self._HUMAN_KEYS)
            l = _dv(b, self._LIGHT_KEYS)
            a = _dv(b, self._AUTH_KEYS)
            f = int(data.get("faces", 0))

            # Dimension-based assignment (baseline — always computed)
            dim_best, dim_score = "General/Mixed", 0.0
            for name, score_fn in niche_scorers:
                s = score_fn(c, t, h, l, a, f)
                if s > dim_score:
                    dim_score, dim_best = s, name

            best_name = dim_best
            # Visual-prototype override when classifier is confident (prob > 0.55)
            if clf_ok:
                emb = self.cache.get(path, {}).get("embedding")
                if emb:
                    try:
                        clf_niche, clf_prob = clf.top_niche(emb)
                        mapped = _CLF_MAP.get(clf_niche)
                        if mapped and clf_prob > 0.55 and mapped in counts:
                            best_name = mapped
                    except Exception:
                        pass

            counts[best_name] = counts.get(best_name, 0) + 1

        sorted_genres = sorted(
            [{"name": k, "count": v} for k, v in counts.items() if v > 0],
            key=lambda x: x["count"], reverse=True,
        )
        return [{"name": "Any", "count": len(results)}] + sorted_genres[:top_n]

    def _classify_genre(self, breakdown: dict) -> str:
        """
        Classify a single photo into a broad genre using key-set lookups so
        preset-specific labels (e.g. 'Subject Isolation', 'Framing') resolve
        correctly across all PRESET_RULES configurations.
        """
        def _dv(keys):
            return next((v for k, v in breakdown.items() if k in keys), 0.0)

        h = _dv(self._HUMAN_KEYS)
        c = _dv(self._COMP_KEYS)
        l = _dv(self._LIGHT_KEYS)
        t = _dv(self._TECH_KEYS)

        if h > 0.6 and t > 0.5:    return "Portrait/Human"
        if c > 0.7 and h < 0.3:    return "Architecture/Interior"
        if h > 0.4 and l > 0.5:    return "Street/Environment"
        if c > 0.6 and l < 0.4:    return "Detail/Texture"
        return "Mixed/General"

    def sequence_story(self, results, target=5, subject_type=None, avoid_paths=None,
                        seed=None, pacing_preset=None, cached_labels=None,
                        locked_slots=None):
        import random

        rng         = random.Random(seed if seed is not None else 42)
        avoid       = set(avoid_paths or [])
        # DEBUG LOG: Capture avoid list stats
        avoid_count = len(avoid)
        
        # subject_type=None means "Any" — auto-detect label but skip genre filtering.
        # subject_type=string means user explicitly chose a genre — apply its thresholds.
        stype_label = subject_type or "street"   # used for rationale labels and return value
        apply_filter = subject_type is not None  # only filter when user explicitly chose
        
        # Track image selection frequency to prevent top images from dominating
        # This is a simple in-memory approach - in a real application, this would be persisted
        if not hasattr(self, '_selection_frequency'):
            self._selection_frequency = {}

        # ── 1. Hard genre pre-filter ──────────────────────────────────────────
        # Thresholds are ONLY applied when the user explicitly picks a genre.
        # "Any" (subject_type=None) skips all genre thresholds so atmospheric /
        # low-key shots that don't hit the human/light floor still qualify.
        _GENRE_THRESH: dict = {
            "portrait":     {"human": 0.55, "tech":  0.35},
            "street":       {"human": 0.35, "light": 0.30},
            "architecture": {"comp":  0.60},
            "nature":       {"light": 0.45, "comp":  0.40},
            "nocturnal":    {"auth":  0.25},
        }

        def _dv(b, keys):
            return next((v for k, v in b.items() if k in keys), 0.0)

        def _passes(r):
            if r[0] in avoid:                                         return False
            if r[1]["grade"] == "Error \u274c":                       return False
            if r[1]["score"] <= 0.25:                                 return False
            if "\U0001f501" in r[1].get("sim_flag", ""):              return False
            if not apply_filter:
                return True
            thresh = _GENRE_THRESH.get(stype_label)
            if thresh is None:
                return True
            b   = r[1].get("breakdown", {})
            dim = {
                "human": _dv(b, self._HUMAN_KEYS),
                "light": _dv(b, self._LIGHT_KEYS),
                "comp":  _dv(b, self._COMP_KEYS),
                "tech":  _dv(b, self._TECH_KEYS),
                "auth":  _dv(b, self._AUTH_KEYS),
            }
            return all(dim[k] >= v for k, v in thresh.items())

        # DEBUG LOG: Count how many are filtered by avoid_paths
        before_avoid_filter = len(results)
        valid = [r for r in results if _passes(r)]
        after_avoid_filter = len(valid)
        filtered_by_avoid = before_avoid_filter - after_avoid_filter

        if len(valid) < target:
            genre_hint = f"'{stype_label}'" if apply_filter else "qualifying"
            msg = (f"\u274c Only {len(valid)} {genre_hint} image(s) pass quality "
                   f"thresholds (need {target}). "
                   + ("Try 'Any' or lower the genre filter." if apply_filter
                      else "Add more shots or re-grade the folder."))
            return [], [msg], stype_label

        # ── 1a. Per-niche classification of valid candidates ─────────────────
        # Classify every valid photo into one of the 4 sequencer niches so that
        # the candidate pool (§5 below) can guarantee each niche contributes a
        # minimum of MIN_PER_NICHE images. This prevents the "lopsiding" problem
        # where a folder dominated by one niche (e.g. 90% street) starves the
        # other niches and causes the sequencer to recycle the same handful of
        # photos on every regenerate.
        MIN_PER_NICHE = 10
        NICHE_TYPES   = ("street", "nature", "portrait", "architecture")

        # Bucket valid-array indices per niche. Using indices (not tuples) so we
        # can feed them straight into `pool` downstream without re-lookup.
        niche_buckets: dict[str, list[int]] = {n: [] for n in NICHE_TYPES}

        def _primary_niche_for(breakdown: dict) -> str:
            """Map a photo's breakdown to one of the 4 sequencer niches.

            Uses the same dimensional logic as `_detect_top_niches` but collapses
            the fine-grained niche names into the 4 coarse buckets the sequencer
            operates on. Every photo is assigned to exactly one niche so the
            buckets partition the valid pool (no double-counting).
            """
            h = _dv(breakdown, self._HUMAN_KEYS)
            c = _dv(breakdown, self._COMP_KEYS)
            l = _dv(breakdown, self._LIGHT_KEYS)
            t = _dv(breakdown, self._TECH_KEYS)
            a = _dv(breakdown, self._AUTH_KEYS)

            # Score each niche the same way detect_subject_type does, then pick
            # the winner. Keeps this classifier consistent with the rest of the
            # pipeline rather than introducing a third set of thresholds.
            niche_scores = {
                "portrait":     h * 0.65 + l * 0.20 + a * 0.15,
                "street":       h * 0.45 + a * 0.38 + c * 0.17,
                "architecture": (c * t) * 0.70 + l * 0.10 + c * 0.20
                                - max(0.0, h - 0.45) * 0.55,
                "nature":       l * 0.65 + c * 0.20 + t * 0.15
                                - max(0.0, c - l) * 0.80
                                - max(0.0, h - 0.20) * 0.80,
            }
            return max(niche_scores, key=niche_scores.get)

        for vi, r in enumerate(valid):
            b = r[1].get("breakdown", {})
            niche_buckets[_primary_niche_for(b)].append(vi)

        # ── 2. Embeddings ─────────────────────────────────────────────────────
        try:
            paths  = [r[0] for r in valid]
            scores = np.array([r[1]["score"] for r in valid], dtype=np.float32)
            embs   = np.array([r[1]["embedding"] for r in valid], dtype=np.float64)
            norms  = np.linalg.norm(embs, axis=1, keepdims=True)
            embs   = embs / (norms + 1e-9)
        except Exception:
            return [r[0] for r in valid[:target]], ["Embedding unavailable"], stype_label

        # ── 2b. Folder similarity calibration ─────────────────────────────────
        # Compute pairwise distribution so target_sim values auto-adapt to this
        # folder's embedding spread rather than using hardcoded absolute values.
        _n_cal = min(len(valid), 50)
        _cal_mat = _cosine_similarity(embs[:_n_cal], embs[:_n_cal]).astype(np.float32)
        np.fill_diagonal(_cal_mat, 0)
        _flat_sims = _cal_mat[_cal_mat > 0.05]
        if len(_flat_sims) >= 10:
            _sim_p20 = float(np.percentile(_flat_sims, 20))
            _sim_p75 = float(np.percentile(_flat_sims, 75))
            # Cap at 0.85: burst/event folders can have P88 > 0.90, which would
            # fail to catch obvious duplicates until similarity is almost identical.
            _dup_thr = float(min(0.85, np.percentile(_flat_sims, 88)))
        else:
            _sim_p20, _sim_p75, _dup_thr = 0.25, 0.65, 0.75

        def _calibrate(t):
            """Remap abstract target (0.25–0.55) to folder-specific similarity range.
            Hard ceiling at 0.70 so contrast/shift slots still produce real diversity
            even in highly-similar folders (burst series, same-location event shots).
            """
            if t is None:
                return None
            raw = float(_sim_p20 + (float(t) - 0.25) / 0.30 * (_sim_p75 - _sim_p20))
            return float(np.clip(raw, _sim_p20, 0.70))

        # Dimension weights for narrative distance blend: [comp, tech, human, light, narrative]
        _DIM_W = np.array([0.15, 0.10, 0.30, 0.25, 0.20], dtype=np.float32)

        # ── 3. Per-photo dimension cache ──────────────────────────────────────
        def _dims(idx):
            b = valid[idx][1].get("breakdown", {})
            return (
                _dv(b, self._COMP_KEYS),   # c
                _dv(b, self._TECH_KEYS),   # t
                _dv(b, self._HUMAN_KEYS),  # h
                _dv(b, self._LIGHT_KEYS),  # l
                _dv(b, self._AUTH_KEYS),   # a
            )

        dim_cache = [_dims(i) for i in range(len(valid))]

        # ── 4. Slot role configuration ────────────────────────────────────────
        # Each slot has a role fitness function + diversity weight.
        # role_weight = 1.0 - div_weight - 0.35 (quality is always 0.35).
        # Contrast/Shift maximises diversity from what came before (div_w=0.45).
        # Opening Frame picks on content alone (no prior sequence, div_w=0.10).
        # Slot tuples: (label, role_fn, div_w, target_sim)
        # target_sim = expected cosine similarity to the PREVIOUS shot.
        # None on slot 0 (no previous). Encodes the narrative transition type:
        #   ~0.50 = connected continuation  |  ~0.25 = deliberate contrast/shift
        _SLOT_CONFIGS: dict = {
            "nocturnal": [
                ("Opening Atmosphere", lambda c,t,h,l,a: l*0.50 + c*0.28 + a*0.22, 0.10, None),
                ("Focal Moment",       lambda c,t,h,l,a: h*0.40 + l*0.36 + a*0.24, 0.28, 0.50),
                ("Supporting Detail",  lambda c,t,h,l,a: t*0.42 + c*0.38 + l*0.20, 0.32, 0.40),
                ("Contrast/Shift",     lambda c,t,h,l,a: a*0.36 + l*0.34 + c*0.30, 0.45, 0.25),
                ("Closing Mood",       lambda c,t,h,l,a: l*0.52 + a*0.30 + c*0.18, 0.28, 0.50),
            ],
            "architecture": [
                ("Facade & Scale",     lambda c,t,h,l,a: c*0.50 + l*0.28 + t*0.22, 0.10, None),
                ("Geometric Detail",   lambda c,t,h,l,a: t*0.46 + c*0.38 + l*0.16, 0.28, 0.45),
                ("Light & Shadow",     lambda c,t,h,l,a: l*0.52 + c*0.30 + t*0.18, 0.32, 0.40),
                ("Human Scale",        lambda c,t,h,l,a: h*0.44 + c*0.32 + l*0.24, 0.40, 0.30),
                ("Abstract Close",     lambda c,t,h,l,a: c*0.50 + t*0.34 + l*0.16, 0.28, 0.45),
            ],
            "portrait": [
                ("Subject Intro",      lambda c,t,h,l,a: h*0.52 + l*0.30 + t*0.18, 0.10, None),
                ("Connection",         lambda c,t,h,l,a: h*0.50 + a*0.36 + l*0.14, 0.25, 0.55),
                ("Environmental",      lambda c,t,h,l,a: c*0.44 + h*0.36 + l*0.20, 0.35, 0.35),
                ("Unguarded",          lambda c,t,h,l,a: a*0.50 + h*0.34 + t*0.16, 0.42, 0.40),
                ("Defining Frame",     lambda c,t,h,l,a: l*0.44 + h*0.38 + c*0.18, 0.25, 0.50),
            ],
            "nature": [
                ("Scene Opener",       lambda c,t,h,l,a: l*0.46 + c*0.34 + t*0.20, 0.10, None),
                ("Landscape Anchor",   lambda c,t,h,l,a: l*0.54 + c*0.28 + t*0.18, 0.28, 0.55),
                ("Detail / Wildlife",  lambda c,t,h,l,a: t*0.46 + c*0.36 + l*0.18, 0.35, 0.30),
                ("Mood & Atmosphere",  lambda c,t,h,l,a: l*0.56 + a*0.26 + c*0.18, 0.40, 0.40),
                ("Quiet Close",        lambda c,t,h,l,a: l*0.44 + c*0.34 + t*0.22, 0.28, 0.50),
            ],
        }
        _SLOT_CONFIG = _SLOT_CONFIGS.get(stype_label, [
            ("Opening Frame",
                lambda c,t,h,l,a: c*0.42 + l*0.28 + t*0.18 + a*0.12, 0.10, None),
            ("Focal Subject",
                lambda c,t,h,l,a: h*0.46 + a*0.36 + t*0.18,          0.28, 0.50),
            ("Supporting Detail",
                lambda c,t,h,l,a: t*0.42 + c*0.38 + l*0.20,          0.32, 0.40),
            ("Contrast/Shift",
                lambda c,t,h,l,a: a*0.36 + l*0.34 + c*0.30,          0.45, 0.25),
            ("Closing Mood",
                lambda c,t,h,l,a: l*0.46 + a*0.36 + c*0.18,          0.28, 0.45),
        ])

        # ── 5. Candidate pool: adaptive score floor with diversity enhancement ───────────────────────────
        # Floor = folder's own 25th-percentile score, minimum 0.25, to include more diverse candidates.
        # Fallback minimum is 5× target (25 for 5 slots) to ensure each slot has ample choice for diversity.
        score_floor = max(0.20, float(np.percentile(scores, 15)))  # Lower floor for more diversity
        pool = [i for i in np.argsort(-scores).tolist() if scores[i] >= score_floor]
        if len(pool) < target * 5:  # Increased from 4× to 5× for better diversity
            # Include a mix of top and diverse candidates
            top_count = min(len(scores), target * 3)
            top_candidates = np.argsort(-scores)[:top_count].tolist()  # Top scoring candidates
            # Add some randomly selected candidates to increase diversity
            remaining_indices = [i for i in range(len(scores)) if i not in top_candidates]
            if remaining_indices and len(remaining_indices) > target * 2:
                # Randomly sample from remaining candidates to add diversity
                rng.shuffle(remaining_indices)
                diverse_count = min(len(remaining_indices), target * 2)
                diverse_candidates = remaining_indices[:diverse_count]
            else:
                diverse_candidates = remaining_indices if remaining_indices else []
            pool = list(set(top_candidates + diverse_candidates))  # Combine and deduplicate
            # Sort by score to maintain some quality preference
            pool.sort(key=lambda i: scores[i], reverse=True)
            # Limit to reasonable size
            pool = pool[:min(len(scores), target * 5)]
        # Ensure we have enough candidates by adding top scorers if needed
        if len(pool) < target:
            pool = np.argsort(-scores)[:min(len(scores), target)].tolist()
        

        # ── 5a. Niche balancing: guarantee MIN_PER_NICHE per niche ────────────
        # Regeneration diversity collapses when one niche dominates the pool.
        # Force each of the 4 sequencer niches (street/nature/portrait/architecture)
        # to contribute at least MIN_PER_NICHE=10 of its top-scoring images.
        # If a niche has fewer than 10 valid photos, we include all of them and
        # attach a soft warning rather than aborting — under-represented niches
        # simply get less weight but still appear.
        pool_set = set(pool)
        niche_warnings: list[str] = []
        for niche in NICHE_TYPES:
            bucket = niche_buckets.get(niche, [])
            # Sort this niche's members by score (best first) so padding uses quality order.
            bucket_sorted = sorted(bucket, key=lambda i: scores[i], reverse=True)
            top_in_niche  = bucket_sorted[:MIN_PER_NICHE]

            # Count how many of this niche's top picks are already in the pool
            already_in_pool = [i for i in top_in_niche if i in pool_set]
            missing         = MIN_PER_NICHE - len(already_in_pool)
            if missing > 0:
                # Pad with the best-scoring images from this niche that aren't yet in pool
                additions = [i for i in bucket_sorted if i not in pool_set][:missing]
                pool.extend(additions)
                pool_set.update(additions)

            # If this niche has fewer than MIN_PER_NICHE total photos, log a warning.
            # The sequencer still runs; the UI can surface the imbalance if desired.
            if len(bucket) < MIN_PER_NICHE:
                niche_warnings.append(
                    f"\u26a0\ufe0f Niche '{niche}' has only {len(bucket)} qualifying "
                    f"image(s) (<{MIN_PER_NICHE}). Sequencer may over-sample this niche."
                )

        # Re-sort the (possibly enlarged) pool so downstream slot scoring keeps
        # operating on a quality-ordered list. Niche padding preserves each
        # niche's top-10 without disturbing the original score ordering above.
        pool.sort(key=lambda i: scores[i], reverse=True)

        # Role requirements lookup keyed by slot index for this subject type.
        _req_list = self._ROLE_REQUIREMENTS.get(stype_label,
                        self._ROLE_REQUIREMENTS["street"])

        # ── 6. Slot-by-slot greedy assignment ─────────────────────────────────
        # Pre-populate locked slots so regeneration leaves them untouched.
        # locked_slots = {slot_index: path_string} from the frontend.
        _locked: dict[int, int] = {}   # slot_idx → valid-array index
        if locked_slots:
            _path_to_idx = {r[0]: i for i, r in enumerate(valid)}
            for slot_str, path in locked_slots.items():
                slot_i = int(slot_str)
                vi = _path_to_idx.get(path)
                if vi is not None and 0 <= slot_i < target:
                    _locked[slot_i] = vi

        seq: list[int | None] = [_locked.get(i) for i in range(target)]
        used: set[int] = set(_locked.values())

        for slot_idx in range(min(target, len(_SLOT_CONFIG))):
            if seq[slot_idx] is not None:   # locked — keep as-is
                continue
            lbl, role_fn, div_w, target_sim = _SLOT_CONFIG[slot_idx]
            # Quality weight scales down as diversity weight rises so div slots
            # aren't overridden by the constant 0.35 quality contribution.
            qual_w = max(0.15, 0.35 - div_w * 0.40)
            role_w = 1.0 - div_w - qual_w
            cands  = [i for i in pool if i not in used]
            if not cands:
                break

            # Role fitness from breakdown dimensions
            role_arr = np.array(
                [role_fn(*dim_cache[i]) for i in cands], dtype=np.float32
            )

            # Apply _ROLE_REQUIREMENTS trait penalties so each slot enforces its
            # visual contract (e.g. Establishing Shot must have comp >= 0.40).
            # Penalties are subtracted from role_arr; photos meeting the trait are unaffected.
            if slot_idx < len(_req_list):
                req = _req_list[slot_idx]
                trait_fn, penalty = req["trait"], req["penalty"]
                if trait_fn is not None:
                    for ci, idx in enumerate(cands):
                        c_, t_, h_, l_, a_ = dim_cache[idx]
                        if not trait_fn(h_, l_, c_, t_, a_):
                            role_arr[ci] = max(0.0, role_arr[ci] - penalty)

            # Quality — normalised within the pool so relative differences matter
            # Apply frequency penalty to prevent top images from dominating
            qual_raw  = scores[np.array(cands)]
            # Apply a small penalty based on selection frequency (max 15% reduction)
            freq_penalties = np.array([
                min(0.15, self._selection_frequency.get(paths[cand_idx], 0) * 0.05)
                for cand_idx in cands
            ], dtype=np.float32)
            qual_raw = qual_raw * (1.0 - freq_penalties)
            
            q_min, q_max = float(qual_raw.min()), float(qual_raw.max())
            qual_arr  = (qual_raw - q_min) / (q_max - q_min + 1e-9)

            # Transition coherence: reward composite sim close to calibrated target.
            # Composite = 0.60×cosine + 0.40×dimension_delta so narrative distance
            # (human/light/mood dims) contributes alongside raw embedding distance.
            # Dup threshold uses folder-calibrated P88 instead of hardcoded 0.75.
            _filled = [s for s in seq[:slot_idx] if s is not None]
            if _filled:
                cand_embs  = embs[np.array(cands)]
                cal_target = _calibrate(target_sim)
                if cal_target is not None:
                    prev_emb  = embs[_filled[-1]].reshape(1, -1)
                    cos_sim   = _cosine_similarity(cand_embs, prev_emb).flatten().astype(np.float32)

                    # Dimension delta: weighted L1 across [comp,tech,human,light,narrative]
                    prev_dims  = np.array(dim_cache[_filled[-1]], dtype=np.float32)
                    cand_dims  = np.array([dim_cache[i] for i in cands], dtype=np.float32)
                    dim_delta  = np.abs(cand_dims - prev_dims)            # (N, 5)
                    dim_dist   = (dim_delta * _DIM_W).sum(axis=1)         # (N,)
                    dim_sim    = np.clip(1.0 - dim_dist, 0.0, 1.0).astype(np.float32)

                    composite  = (0.60 * cos_sim + 0.40 * dim_sim).astype(np.float32)
                    band       = float(max(cal_target, 1.0 - cal_target))
                    coherence_arr = np.clip(
                        1.0 - np.abs(composite - cal_target) / band, 0.0, 1.0
                    )
                else:
                    coherence_arr = np.ones(len(cands), dtype=np.float32)
                sel_embs    = embs[np.array(_filled)]
                max_sim_any = _cosine_similarity(cand_embs, sel_embs).max(axis=1).astype(np.float32)
                dup_penalty = np.clip((max_sim_any - _dup_thr) * 4.0, 0.0, 1.0)
                div_arr     = np.clip(coherence_arr - dup_penalty, 0.0, 1.0)
            else:
                div_arr = np.ones(len(cands), dtype=np.float32)

            # Subject thread: opening had human presence → bias slots 1-2 toward it
            h_thread = np.zeros(len(cands), dtype=np.float32)
            _first_filled = next((s for s in seq if s is not None), None)
            if slot_idx in (1, 2) and _first_filled is not None:
                h0 = dim_cache[_first_filled][2]
                if h0 > 0.35:
                    h_thread = np.array(
                        [dim_cache[i][2] for i in cands], dtype=np.float32
                    ) * 0.06

            jitter = np.array(
                [rng.uniform(-0.12, 0.12) for _ in cands], dtype=np.float32
            )

            combined = role_arr * role_w + qual_arr * qual_w + div_arr * div_w + h_thread + jitter
            pick     = cands[int(np.argmax(combined))]
            seq[slot_idx] = pick
            used.add(pick)

        # ── 7. Rationales: slot-aware, dimension-honest ───────────────────────
        # Each caption names the 2 dimensions most relevant to THIS slot's role.
        _SLOT_DIM_PRIORITY_MAP: dict = {
            "nocturnal": [
                ["Light",       "Narrative",   "Composition", "Technical", "Human"],
                ["Light",       "Human",       "Narrative",   "Composition","Technical"],
                ["Technical",   "Composition", "Light",       "Narrative",  "Human"],
                ["Narrative",   "Light",       "Composition", "Technical",  "Human"],
                ["Light",       "Narrative",   "Composition", "Technical",  "Human"],
            ],
            "architecture": [
                ["Composition", "Light",       "Technical",   "Narrative",  "Human"],
                ["Technical",   "Composition", "Light",       "Narrative",  "Human"],
                ["Light",       "Composition", "Technical",   "Narrative",  "Human"],
                ["Human",       "Composition", "Light",       "Technical",  "Narrative"],
                ["Composition", "Technical",   "Light",       "Narrative",  "Human"],
            ],
            "portrait": [
                ["Human",       "Light",       "Technical",   "Narrative",  "Composition"],
                ["Human",       "Narrative",   "Light",       "Technical",  "Composition"],
                ["Composition", "Human",       "Light",       "Narrative",  "Technical"],
                ["Narrative",   "Human",       "Technical",   "Light",      "Composition"],
                ["Light",       "Human",       "Composition", "Narrative",  "Technical"],
            ],
            "nature": [
                ["Light",       "Composition", "Technical",   "Narrative",  "Human"],
                ["Light",       "Composition", "Technical",   "Narrative",  "Human"],
                ["Technical",   "Composition", "Light",       "Narrative",  "Human"],
                ["Light",       "Narrative",   "Composition", "Technical",  "Human"],
                ["Light",       "Composition", "Technical",   "Narrative",  "Human"],
            ],
        }
        _SLOT_DIM_PRIORITY = _SLOT_DIM_PRIORITY_MAP.get(stype_label, [
            ["Composition", "Light",     "Technical", "Narrative", "Human"],   # Opening
            ["Human",       "Narrative", "Technical", "Light",     "Composition"],  # Focal
            ["Technical",   "Composition","Light",    "Narrative", "Human"],   # Detail
            ["Light",       "Narrative", "Composition","Technical","Human"],   # Contrast
            ["Light",       "Narrative", "Composition","Technical","Human"],   # Closing
        ])
        _DIM_NAMES = ["Composition", "Technical", "Human", "Light", "Narrative"]

        # Drop any None entries (pool ran dry for an unlocked slot)
        seq_clean = [i for i in seq if i is not None]

        rationale = []
        for i, idx in enumerate(seq_clean):
            lbl   = _SLOT_CONFIG[i][0] if i < len(_SLOT_CONFIG) else f"Frame {i+1}"
            score = valid[idx][1]["score"]
            c, t, h, l, a = dim_cache[idx]
            dim_vals = dict(zip(_DIM_NAMES, [c, t, h, l, a]))
            priority = _SLOT_DIM_PRIORITY[i] if i < len(_SLOT_DIM_PRIORITY) else _DIM_NAMES
            top2 = sorted(priority[:3], key=lambda n: dim_vals.get(n, 0), reverse=True)[:2]
            metrics = " · ".join(f"{n}: {dim_vals[n]:.2f}" for n in top2)
            locked_tag = " 🔒" if i in _locked else ""
            rationale.append(f"**{lbl}**{locked_tag} — Score {score:.2f} · {metrics}")

        # Update selection frequency for chosen images to prevent domination
        selected_paths = [paths[i] for i in seq_clean]
        for path in selected_paths:
            self._selection_frequency[path] = self._selection_frequency.get(path, 0) + 1

        return selected_paths, rationale[:target], stype_label

