"""One-shot brand migration: Cullwise → Cullwise.

Deprecates the old name across every live source file — code, comments,
docs, configs, env vars, Tauri/Cargo metadata. Deliberately skipped:
build output (dist/, target/), vendor dirs, caches, generated reports.

Ordered case-sensitive replacements:
    CULLWISE  → CULLWISE   (env vars, report headers)
    Cullwise  → Cullwise   (brand, docs, UI strings)
    Cullwise → Cullwise   (spaced variant)
    Cullwise  → Cullwise   (safety)
    cullwise  → cullwise   (package names, binary names, domains)

After this script: rename Cullwise.spec → Cullwise.spec and
"Install Cullwise.bat" → "Install Cullwise.bat" on disk.
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
    ("CULLWISE", "CULLWISE"),
    ("Cullwise", "Cullwise"),
    ("Cullwise", "Cullwise"),
    ("Cullwise", "Cullwise"),
    ("cullwise", "cullwise"),
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
        if "cullwise" not in text.lower().replace(" ", "").replace("-", "") \
           and "cullwise" not in text.lower():
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
