"""
Simulation test for LightweightStreetScorer.
Generates synthetic images in a temp folder and exercises the full pipeline.

Run from the project root:
    venv/Scripts/python tests/test_scorer.py
"""

import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from lightweight_analyzer import LightweightStreetScorer

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    return condition

def make_image(path, kind="normal", w=640, h=480):
    """Synthesise an image designed to exercise a specific code path."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if kind == "normal":
        # Mid-grey with some edges and brightness variation
        img[:] = 100
        cv2.rectangle(img, (w//4, h//4), (3*w//4, 3*h//4), (200, 200, 200), -1)
        cv2.line(img, (0, 0), (w, h), (50, 50, 50), 3)
        cv2.line(img, (w, 0), (0, h), (50, 50, 50), 3)
    elif kind == "bright":
        img[:] = 240   # over-exposed
    elif kind == "dark":
        img[:] = 10    # under-exposed
    elif kind == "sharp":
        # High-frequency grid → high Laplacian variance
        for i in range(0, w, 4):
            img[:, i] = 255 if (i // 4) % 2 == 0 else 0
    elif kind == "tiny":
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[:] = 128
    elif kind == "corrupt":
        # Write garbage bytes — cv2.imread should return None
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02\x03" * 100)
        return   # skip normal save
    cv2.imwrite(path, img)

def run_all():
    tmpdir = tempfile.mkdtemp(prefix="ssc_test_")
    cache  = os.path.join(tmpdir, "test_cache.json")
    print(f"\nTest folder: {tmpdir}\n")

    try:
        # ── Generate synthetic images ────────────────────────────────────────
        kinds = ["normal", "bright", "dark", "sharp", "tiny", "corrupt"]
        for i, kind in enumerate(kinds):
            p = os.path.join(tmpdir, f"{i:02d}_{kind}.jpg")
            make_image(p, kind=kind)

        scorer = LightweightStreetScorer(cache_path=cache)

        # ── analyze_folder ───────────────────────────────────────────────────
        print("analyze_folder:")
        results = scorer.analyze_folder(tmpdir)
        check("returns a list",              isinstance(results, list))
        check("processes all valid images",  len(results) == len(kinds),
              f"got {len(results)}, expected {len(kinds)}")

        # ── per-result structure ─────────────────────────────────────────────
        print("\nResult structure:")
        required_keys = {"score", "grade", "human_perception", "critique", "dims", "faces", "embedding"}
        all_ok = True
        for path, r in results:
            missing = required_keys - set(r.keys())
            if missing:
                check(f"  {os.path.basename(path)} keys", False, f"missing: {missing}")
                all_ok = False
        check("all results have required keys", all_ok)

        scores = [r["score"] for _, r in results]
        check("all scores in [0, 1]",
              all(0.0 <= s <= 1.0 for s in scores),
              str(scores))

        emb_lens = [len(r["embedding"]) for _, r in results]
        check("all embeddings length 256",
              all(l == 256 for l in emb_lens),
              str(emb_lens))

        # ── corrupt image ────────────────────────────────────────────────────
        print("\nCorrupt image handling:")
        corrupt_results = [(p, r) for p, r in results if "corrupt" in p]
        if corrupt_results:
            _, cr = corrupt_results[0]
            check("corrupt returns score 0.0",  cr["score"] == 0.0)
            check("corrupt returns Error grade", "Error" in cr["grade"])
            check("corrupt has valid embedding", len(cr["embedding"]) == 256)

        # ── tiny image (boundary slicing) ────────────────────────────────────
        print("\nTiny image (20×20) boundary safety:")
        tiny_results = [(p, r) for p, r in results if "tiny" in p]
        if tiny_results:
            _, tr = tiny_results[0]
            check("tiny: no exception (score present)", "score" in tr)
            check("tiny: score in range", 0.0 <= tr["score"] <= 1.0)

        # ── preset switching ─────────────────────────────────────────────────
        print("\nPreset switching:")
        normal_path = os.path.join(tmpdir, "00_normal.jpg")
        r_magnum = scorer._analyze(normal_path, preset="Magnum Editor")
        r_lspf   = scorer._analyze(normal_path, preset="LSPF (London Street)")
        r_spi    = scorer._analyze(normal_path, preset="SPI (International)")
        r_bad    = scorer._analyze(normal_path, preset="Nonexistent Preset")
        check("Magnum returns valid score",   0.0 <= r_magnum["score"] <= 1.0)
        check("LSPF returns valid score",     0.0 <= r_lspf["score"]   <= 1.0)
        check("SPI returns valid score",      0.0 <= r_spi["score"]    <= 1.0)
        check("Unknown preset falls back OK", 0.0 <= r_bad["score"]    <= 1.0)

        # ── caching ──────────────────────────────────────────────────────────
        print("\nDisk cache:")
        import json
        from pathlib import Path
        check("cache file written", Path(cache).exists())
        cached = json.loads(Path(cache).read_text(encoding="utf-8"))
        check("cache contains entries", len(cached) > 0, f"{len(cached)} entries")

        scorer2 = LightweightStreetScorer(cache_path=cache)
        check("cache loaded on re-init", len(scorer2.cache) == len(cached))

        # ── sequence_story ───────────────────────────────────────────────────
        print("\nsequence_story:")
        valid_results = [(p, r) for p, r in results if r["score"] > 0]
        seq_paths, rationale, stype_label = scorer.sequence_story(valid_results, target=5)
        check("returns 5 paths (or fewer if not enough)",
              len(seq_paths) <= 5 and len(seq_paths) > 0,
              f"{len(seq_paths)} paths")
        check("no duplicate paths",  len(seq_paths) == len(set(seq_paths)))
        check("rationale non-empty", any(r for r in rationale))
        check("subject type label returned", isinstance(stype_label, str))

        # ── too few images for sequence ──────────────────────────────────────
        short_results = valid_results[:2]
        sp2, _, _ = scorer.sequence_story(short_results, target=5)
        check("graceful fallback with < target images",
              isinstance(sp2, list) and len(sp2) <= 5)

        # ── selection frequency tracking ─────────────────────────────────────────
        print("\nSelection frequency tracking:")
        # Run sequence_story multiple times to test frequency tracking
        seq1_paths, _, _ = scorer.sequence_story(valid_results, target=3)
        seq2_paths, _, _ = scorer.sequence_story(valid_results, target=3)
        seq3_paths, _, _ = scorer.sequence_story(valid_results, target=3)
        
        # Check that selection frequency tracking is working
        check("selection frequency tracking initialized",
              hasattr(scorer, '_selection_frequency') and isinstance(scorer._selection_frequency, dict))
        
        # Check that some images were selected
        all_selected = set(seq1_paths + seq2_paths + seq3_paths)
        check("images were selected in sequences", len(all_selected) > 0)
        
        # Check that frequency tracking recorded selections
        if all_selected:
            freq_sum = sum(scorer._selection_frequency.get(path, 0) for path in all_selected)
            check("selection frequency tracking recorded selections", freq_sum > 0)

        # ── empty folder ─────────────────────────────────────────────────────
        print("\nEdge cases:")
        empty_dir = os.path.join(tmpdir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        empty_results = scorer.analyze_folder(empty_dir)
        check("empty folder returns []", empty_results == [])

        bad_results = scorer.analyze_folder("/nonexistent/path/xyz")
        check("nonexistent folder returns []", bad_results == [])

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print()

if __name__ == "__main__":
    run_all()
