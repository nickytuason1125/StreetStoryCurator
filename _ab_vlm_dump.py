"""Grade the 25 test photos with whichever VLM QWEN_VLM_DIR selects and dump
path -> {score, breakdown, critique} plus timing to the JSON file in argv[1].

Usage:
    python -u _ab_vlm_dump.py out_a.json                       (default 2.5-VL-3B)
    QWEN_VLM_DIR=models/qwen3_vl python -u _ab_vlm_dump.py out_b.json
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

out_path = sys.argv[1]
FOLDER   = r"C:\Users\Nicky Tuason\Desktop\test_batch_25"
MODE     = "classic_street"

from pathlib import Path
exts  = {".jpg", ".jpeg", ".png"}
paths = sorted(str(p) for p in Path(FOLDER).iterdir() if p.suffix.lower() in exts)[:25]

from qwen_vlm_grader import QwenVLMGrader, MODEL_CACHE_DIR
from pdf_rag import load_concepts

print(f"model dir: {MODEL_CACHE_DIR}")
t0 = time.time()
grader = QwenVLMGrader()
t_load = time.time() - t0
print(f"load: {t_load:.1f}s")

t0 = time.time()
results = grader.grade_images_scored(paths, mode=MODE, rag_phrases=load_concepts())
t_grade = time.time() - t0
grader.unload()

out = {
    "model_dir": str(MODEL_CACHE_DIR),
    "t_load":    round(t_load, 1),
    "t_grade":   round(t_grade, 1),
    "s_per_img": round(t_grade / len(paths), 2),
    "results": {
        os.path.basename(r.path): {
            "score":     round(r.score, 3),
            "breakdown": {k: round(float(v), 3) for k, v in r.breakdown.items()
                          if isinstance(v, (int, float))},
            "critique":  r.critique,
        }
        for r in results
    },
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
n_blank = sum(1 for r in results if not r.breakdown)
print(f"done: {len(results)} graded, {n_blank} blank breakdowns, "
      f"{out['s_per_img']}s/img -> {out_path}")
