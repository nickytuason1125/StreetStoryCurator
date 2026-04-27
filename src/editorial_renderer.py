import os, io, zipfile
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List

CANVAS_W, CANVAS_H = 1080, 1350


def _parse_fill(fill):
    """Convert CSS rgba() string to a PIL-compatible RGB tuple."""
    if isinstance(fill, str) and fill.startswith("rgba("):
        parts = fill[5:-1].split(",")
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        return (r, g, b)
    return fill


def draw_text_fallback(draw, text, position, font_size, fill="white"):
    """Try multiple font paths for cross-platform compatibility."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "Arial.ttf",  # fallback in PATH
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue
    else:
        font = ImageFont.load_default()
    draw.text(position, text, fill=_parse_fill(fill), font=font)


def _apply_gradient(canvas: Image.Image, y_range: int, max_alpha: int) -> Image.Image:
    """Composite a bottom-up black gradient onto canvas via RGBA blending."""
    overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for y in range(y_range):
        alpha = int(max_alpha * (y / y_range) ** 2)
        od.rectangle((0, CANVAS_H - y, CANVAS_W, CANVAS_H - y), fill=(0, 0, 0, alpha))
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def generate_magazine_carousel(images_data, output_path="output/magazine_carousel.zip"):
    Path("output").mkdir(exist_ok=True)
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, item in enumerate(images_data):
            canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (10, 10, 12))

            try:
                img = Image.open(item["path"]).convert("RGB")
                target_ratio = CANVAS_H / CANVAS_W
                img_ratio = img.height / img.width
                if img_ratio > target_ratio:
                    new_h = int(img.width * target_ratio)
                    top = (img.height - new_h) // 2
                    box = (0, top, img.width, top + new_h)
                else:
                    new_w = int(img.height / target_ratio)
                    left = (img.width - new_w) // 2
                    box = (left, 0, left + new_w, img.height)
                canvas = img.crop(box).resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
            except Exception:
                pass

            if i == 0:
                canvas = _apply_gradient(canvas, 600, 255)
                draw = ImageDraw.Draw(canvas)
                draw_text_fallback(draw, "STREET STORY", (80, CANVAS_H - 220), 80, "white")
                draw_text_fallback(draw, "CURATED COLLECTION", (80, CANVAS_H - 140), 30, "rgba(255,255,255,0.7)")
                draw_text_fallback(draw, "SSC", (80, 80), 40, "white")
            else:
                canvas = _apply_gradient(canvas, 180, 210)
                draw = ImageDraw.Draw(canvas)
                draw_text_fallback(draw, f"{i+1}/5", (CANVAS_W - 80, 60), 26, "rgba(255,255,255,0.4)")
                caption = item.get("rationale", "")
                if caption:
                    draw_text_fallback(draw, caption, (60, CANVAS_H - 150), 32, "rgba(255,255,255,0.9)")

            img_byte_arr = io.BytesIO()
            canvas.save(img_byte_arr, format="JPEG", quality=95, optimize=True)
            img_byte_arr.seek(0)
            zf.writestr(f"slide_{i+1}.jpg", img_byte_arr.getvalue())

    final_path = os.path.join("output", "magazine_carousel.zip")
    with open(final_path, "wb") as f:
        f.write(zip_buffer.getvalue())
    return final_path


# ---------------------------------------------------------------------------
# render_editorial_carousel — used by /api/editorial slot-selection endpoint
# ---------------------------------------------------------------------------

FORMAT_SIZES = {
    "portrait":  (1080, 1350),
    "square":    (1080, 1080),
    "landscape": (1080,  566),
}


def _crop_and_resize(img: Image.Image, tw: int, th: int) -> Image.Image:
    img = img.convert("RGB")
    sw, sh = img.size
    if (sw / sh) > (tw / th):
        nw = int(sh * tw / th)
        img = img.crop(((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh))
    else:
        nh = int(sw * th / tw)
        img = img.crop((0, (sh - nh) // 2, sw, (sh - nh) // 2 + nh))
    return img.resize((tw, th), Image.Resampling.LANCZOS)


def render_editorial_carousel(
    selected: List[dict],
    out_dir: Path,
    fmt: str = "portrait",
) -> tuple:
    """Pure full-bleed crop — no overlays. Returns (out_paths, zip_path)."""
    tw, th = FORMAT_SIZES.get(fmt, FORMAT_SIZES["portrait"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[str] = []
    for i, item in enumerate(selected):
        try:
            with Image.open(item["path"]) as src:
                slide = _crop_and_resize(src, tw, th)
        except Exception:
            slide = Image.new("RGB", (tw, th), (0, 0, 0))

        out_path = out_dir / f"editorial_{i+1:02d}_{Path(item['path']).stem}.jpg"
        slide.save(str(out_path), "JPEG", quality=92, optimize=True)
        out_paths.append(str(out_path))

    zip_path = str(out_dir / f"editorial_{out_dir.name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_paths:
            zf.write(p, Path(p).name)

    return out_paths, zip_path
