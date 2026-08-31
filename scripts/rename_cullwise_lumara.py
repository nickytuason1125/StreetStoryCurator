"""One-shot brand migration: Lumara → Lumara.

Second migration in the chain (FrameGrade → Lumara → Lumara). Same
discipline as scripts/deprecate_framegrade.py: every live source file —
code, comments, docs, configs, env vars, Tauri/Cargo metadata. Skipped:
build output (dist/, target/), vendor dirs, caches, generated reports.

Ordered case-sensitive replacements:
    LUMARA  → LUMARA   (env vars, report headers)
    Lumara  → Lumara   (camel variant)
    Lumara  → Lumara   (brand, docs, UI strings)
    lumara  → lumara   (package names, binary names, domains)

After this script: rename Lumara.spec → Lumara.spec and
"Install Lumara.bat" → "Install Lumara.bat" on disk.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    "dist", "venv", "node_modules", "cdp-shots", ".git", ".claude",
    ".superpowers", ".pytest_cache", "build_pyi", "reports", "__pycache__",
    "target", "logs", "dataset_images", "models", "output", "screenshots",
    "frontend\\node_modules", ".vscode", ".github",
}
EXTS = {".py", ".tsx", ".ts", ".html", ".json", ".md", ".css", ".mjs",
        ".spec", ".js", ".bat", ".ps1", ".sh", ".toml", ".rs", ".conf",
        ".vbs", ".cfg"}

REPLACEMENTS = [
    ("LUMARA", "LUMARA"),
    ("Lumara", "Lumara"),
    ("Lumara", "Lumara"),
    ("lumara", "lumara"),
]


def main() -> int:
    changed = 0
    total_hits = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if "lumara" not in text.lower():
            continue
        new_text = text
        hits = 0
        for old, new in REPLACEMENTS:
            hits += new_text.count(old)
            new_text = new_text.replace(old, new)
        if hits and new_text != text:
            p.write_text(new_text, encoding="utf-8")
            changed += 1
            total_hits += hits
            print(f"  {hits:>3}  {rel}")
    print(f"\n{changed} files updated, {total_hits} replacements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())