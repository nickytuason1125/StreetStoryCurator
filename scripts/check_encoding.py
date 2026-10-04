"""Encoding integrity guard -- catches the bug classes from the 2026-09 incident:

  * double-encoded mojibake (UTF-8 read as cp1252 and re-saved): '--', '--',
    the box-drawing banner corruption in CreativeDirector.tsx
  * UTF-8 BOM in .py files: creative_director-style tests that do
    ast.parse(read_text(encoding="utf-8")) die with SyntaxError U+FEFF
    (this bit local_llm.py and disabled an entire test until stripped)
  * NUL bytes / UTF-16 leakage from binary-safe editors writing text files

Usage:
    venv\\Scripts\\python.exe scripts\\check_encoding.py          # whole tree
    venv\\Scripts\\python.exe scripts\\check_encoding.py a.py b.tsx  # files only

Exit 0 = clean, 1 = findings. Designed for the pre-commit hook and CI.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Distinct mojibake signatures (all are invalid in sane source):
#   -- + continuation-range char        -- UTF-8 decoded as cp1252, once
#   -- followed by --/--/--/--/--/--/--/"/"/"/"/"-range punctuation -- same, once
#   the replacement character          -- irreversible corruption
#   double-encoded forms: --, -- etc. are covered by the -- prefix rule
import re

MOJIBAKE = re.compile(
    r"[\u00c3][\u0080-\u00bf\u2018\u2019\u201c\u201d\u2020-\u203a\u20ac]"
    r"|[\u00c2][\u0080-\u00bf]"
    r"|\u00e2\u0080[\u0094\u0098\u0099\u009c\u009d\u009e\u00a0\u00a6\u00b9\u00ba]"
    r"|\u00e2\u0082\u00ac"
    r"|\ufffd"
)

CODE_DIRS = ("src", "routers", "scripts", "tests")
CODE_EXTS = {".py", ".ts", ".tsx", ".css"}


def findings_for(path: Path) -> list[str]:
    out = []
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        return [f"not valid UTF-8: {e}"]
    if "\x00" in text:
        out.append("contains NUL bytes (UTF-16 leakage?)")
    if path.suffix == ".py" and data[:3] == b"\xef\xbb\xbf":
        out.append("UTF-8 BOM -- breaks ast.parse(read_text(encoding='utf-8'))")
    for m in MOJIBAKE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        out.append(f"mojibake at line {line}: {m.group(0)!r}")
        if len(out) > 5:
            out.append("-- (further occurrences suppressed)")
            break
    return out


def main() -> int:
    if len(sys.argv) > 1:
        paths = [Path(a) for a in sys.argv[1:]]
    else:
        paths = [
            p for base in CODE_DIRS
            for p in (ROOT / base).rglob("*")
            if p.suffix in CODE_EXTS and p.is_file()
        ]
        frontend = ROOT / "frontend" / "src"
        if frontend.exists():
            paths += [p for p in frontend.rglob("*")
                      if p.suffix in CODE_EXTS and p.is_file()]

    bad = 0
    for p in paths:
        if not p.is_file():
            continue
        try:
            issues = findings_for(p)
        except OSError:
            continue
        for issue in issues:
            print(f"{p.relative_to(ROOT)}: {issue}")
            bad += 1
    if bad:
        print(f"\nENCODING GUARD: {bad} finding(s). "
              "Fix the file (restore from git, or repair as UTF-8); "
              "never 're-save' with a codepage editor.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
