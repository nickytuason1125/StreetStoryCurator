"""
xmp_sidecar.py — write ratings to XMP sidecars so Lightroom can read them.

Why
---
Grades and stars currently live only in this app's own database. That makes the
cull a dead end: you can see the results here, but nothing carries into the
editing workflow. Sidecars are how the rest of the industry solves this — a
plain `photo.xmp` next to `photo.RW2` that Lightroom, Bridge, Capture One and
exiftool all read natively.

Design rules, in order of importance:

  1. NEVER modify the original image. Sidecars only. A culling tool that writes
     into a photographer's RAW files is unacceptable regardless of how careful
     the write is.
  2. NEVER clobber an existing sidecar. Lightroom stores develop settings,
     crops and keywords in the same file. We update the rating/label fields and
     leave every other property byte-for-byte alone.
  3. Atomic writes. A half-written sidecar is worse than none — Lightroom may
     read it as corrupt and discard the photo's metadata.

RAW vs JPEG: Lightroom reads sidecars for RAW formats. For JPEG/TIFF it prefers
metadata embedded in the file itself, which we will not do (rule 1), so a JPEG
sidecar is written for other tools (Bridge, exiftool) but Lightroom may ignore
it. That limitation is inherent to not touching originals.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

# Grade -> XMP label. Lightroom shows these as colour labels when the label
# text matches its configured set; otherwise it displays the raw string.
_GRADE_LABEL = {"Strong": "Green", "Mid": "Yellow", "Weak": "Red"}

_TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="FrameGrade">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/">
   <xmp:Rating>{rating}</xmp:Rating>
   <xmp:Label>{label}</xmp:Label>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def sidecar_path(image_path: str) -> Path:
    """photo.RW2 -> photo.xmp (the convention Lightroom expects)."""
    p = Path(image_path)
    return p.with_suffix(".xmp")


def _grade_word(grade: str) -> str:
    for w in ("Strong", "Mid", "Weak"):
        if w in (grade or ""):
            return w
    return ""


def _upsert(xml: str, tag: str, value: str) -> str:
    """Set one xmp property, whether it exists as an element or an attribute.

    Existing sidecars written by Lightroom may express xmp:Rating either way, so
    both forms are handled. Everything not matched is left untouched — that is
    what makes this safe to run over a catalogue someone has already edited.
    """
    el = re.compile(rf"<xmp:{tag}>.*?</xmp:{tag}>", re.DOTALL)
    if el.search(xml):
        return el.sub(f"<xmp:{tag}>{value}</xmp:{tag}>", xml, count=1)
    attr = re.compile(rf'xmp:{tag}\s*=\s*"[^"]*"')
    if attr.search(xml):
        return attr.sub(f'xmp:{tag}="{value}"', xml, count=1)
    # Not present: insert into the first rdf:Description, declaring the xmp
    # namespace if the existing file did not.
    m = re.search(r"<rdf:Description\b[^>]*>", xml)
    if m:
        head = m.group(0)
        if "xmlns:xmp=" not in head:
            head_new = head[:-1] + ' xmlns:xmp="http://ns.adobe.com/xap/1.0/">'
            xml = xml.replace(head, head_new, 1)
            head = head_new
        return xml.replace(head, head + f"\n   <xmp:{tag}>{value}</xmp:{tag}>", 1)
    return xml


def write_sidecar(image_path: str, stars: Optional[int] = None,
                  grade: str = "", dry_run: bool = False) -> Optional[Path]:
    """Write/update the sidecar for one photo. Returns the path, or None.

    `stars` maps straight to xmp:Rating (0-5). `grade` becomes xmp:Label.
    Nothing is written when there is nothing to say (no stars, no grade).
    """
    rating = None
    if stars is not None:
        try:
            rating = max(0, min(5, int(stars)))
        except (TypeError, ValueError):
            rating = None
    word = _grade_word(grade)
    if rating is None and not word:
        return None

    sc = sidecar_path(image_path)
    label = _GRADE_LABEL.get(word, word)
    try:
        if sc.exists():
            xml = sc.read_text(encoding="utf-8", errors="replace")
            if rating is not None:
                xml = _upsert(xml, "Rating", str(rating))
            if label:
                xml = _upsert(xml, "Label", label)
        else:
            xml = _TEMPLATE.format(rating=rating if rating is not None else 0,
                                   label=label)
        if dry_run:
            return sc
        tmp = sc.with_suffix(".xmp.tmp")
        tmp.write_text(xml, encoding="utf-8")
        os.replace(tmp, sc)          # atomic: never leave a half-written sidecar
        return sc
    except Exception as exc:
        print(f"[xmp] sidecar write failed for {Path(image_path).name}: {exc}")
        return None


def write_for_gallery(photos, only_rated: bool = False) -> int:
    """Write sidecars for a gallery (list of dicts with path/stars/grade)."""
    n = 0
    for ph in photos:
        p = ph.get("path")
        if not p:
            continue
        stars = ph.get("stars") or None
        if only_rated and not stars:
            continue
        if write_sidecar(p, stars=stars, grade=str(ph.get("grade", ""))):
            n += 1
    print(f"[xmp] wrote {n} sidecars")
    return n


if __name__ == "__main__":
    import json, sys
    cat = Path(__file__).resolve().parent.parent / "cache" / "catalog.json"
    only = "--rated-only" in sys.argv
    photos = json.loads(cat.read_text(encoding="utf-8")).get("photos", [])
    if "--dry-run" in sys.argv:
        n = sum(1 for ph in photos if ph.get("path") and
                write_sidecar(ph["path"], ph.get("stars") or None,
                              str(ph.get("grade", "")), dry_run=True))
        print(f"[xmp] would write {n} sidecars (dry run)")
    else:
        write_for_gallery(photos, only_rated=only)
