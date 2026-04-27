import os
from datetime import datetime
from pathlib import Path
from fpdf import FPDF
from PIL import Image


def _strip_emoji(text: str) -> str:
    """Drop characters outside Latin-1 so Helvetica renders them cleanly."""
    return text.encode("latin-1", errors="ignore").decode("latin-1")


class ScorecardPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    def add_cover(self, preset_name, total_images):
        self.add_page()
        self.ln(40)
        self.set_font("Helvetica", "B", 24)
        self.cell(0, 15, "Street Photography Jury Scorecard", ln=True, align="C")
        self.set_font("Helvetica", "", 14)
        self.cell(0, 10, f"Rubric: {preset_name}  |  Images Analyzed: {total_images}", ln=True, align="C")
        self.ln(10)
        self.set_draw_color(200, 200, 200)
        self.line(40, self.get_y(), 170, self.get_y())

    def add_sequence_page(self, seq_paths, scores, critiques, rationale):
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Recommended 5-Frame Sequence", ln=True)
        self.ln(5)

        # Snapshot y before placing thumbnails — image() is absolute so cursor
        # doesn't advance; we restore position manually afterwards.
        thumb_y = self.get_y()
        x_start = 10
        for i, path in enumerate(seq_paths[:5]):
            try:
                with Image.open(path) as img:
                    img.thumbnail((34, 46))
                    temp = f"output/thumb_{i}.jpg"
                    img.save(temp)
                    self.image(temp, x=x_start + i * 38, y=thumb_y, w=34, h=46)
            except Exception:
                self.set_xy(x_start + i * 38, thumb_y)
                self.multi_cell(34, 10, "IMG\nERROR", align="C")

        self.set_y(thumb_y + 50)

        # Rationale
        self.set_font("Helvetica", "I", 10)
        self.multi_cell(0, 8, "\n".join(rationale))
        self.ln(5)

        # Score table header
        self.set_font("Helvetica", "B", 10)
        self.cell(40,  8, "Image",           border=1)
        self.cell(20,  8, "Score",           border=1)
        self.cell(25,  8, "Grade",           border=1)
        self.cell(105, 8, "Judge's Critique", border=1, ln=True)

        # Score table rows
        self.set_font("Helvetica", "", 8)
        for i, path in enumerate(seq_paths[:5]):
            s = scores.get(path, {})
            grade_text   = _strip_emoji(s.get("grade",   "-"))
            critique_text = s.get("critique", "-")[:90]   # prevent cell overflow
            self.cell(40,  12, f"Frame {i + 1}: {Path(path).name[:16]}", border=1)
            self.cell(20,  12, str(s.get("score", "-")),                  border=1)
            self.cell(25,  12, grade_text,                                border=1)
            self.cell(105, 12, critique_text,                             border=1, ln=True)


def generate_pdf(seq_paths, all_scores, preset_name, output_dir="output"):
    Path(output_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"scorecard_{ts}.pdf")

    pdf = ScorecardPDF()
    pdf.add_cover(preset_name, len(all_scores))

    seq_scores = {p: all_scores.get(p, {}) for p in seq_paths}
    rationale_stub = [
        "1. Establishing: Context & Geometry",
        "2. Moment: Human Anchor",
        "3. Detail: Visual Rhythm",
        "4. Contrast: Shift/Tension",
        "5. Atmosphere: Resolution",
    ]
    pdf.add_sequence_page(seq_paths, seq_scores, seq_scores, rationale_stub)

    pdf.output(out_path)
    return out_path
