"""Five-model VLM tournament: grade the 25 test photos with each model via
_ab_vlm_dump.py (QWEN_VLM_DIR env switch), waiting for downloads to complete,
then print a comparison table.
"""
import json, os, subprocess, sys, time

ROOT   = os.path.dirname(os.path.abspath(__file__))
PYEXE  = os.path.join(ROOT, "venv", "Scripts", "python.exe")
MODELS = [
    ("baseline_qwen25_3b", "models/qwen_vlm"),
    ("qwen3_2b",           "models/qwen3_vl_2b"),
    ("internvl35_2b",      "models/internvl35_2b"),
    ("smolvlm2_22b",       "models/smolvlm2"),
    ("qwen3_4b",           "models/qwen3_vl"),
]
WAIT_EACH_S = 45 * 60   # max wait per model for its download to finish


def weights_complete(d: str) -> bool:
    """True when config.json exists and every shard in the index is on disk."""
    d = os.path.join(ROOT, d)
    if not os.path.exists(os.path.join(d, "config.json")):
        # HF-cache layout fallback: any snapshot with config + weights
        return False
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        try:
            shards = set(json.load(open(idx, encoding="utf-8"))["weight_map"].values())
        except Exception:
            return False
        return all(
            os.path.exists(os.path.join(d, s)) and os.path.getsize(os.path.join(d, s)) > 1e6
            for s in shards
        )
    return any(f.endswith(".safetensors") for f in os.listdir(d))


results = {}
for name, mdir in MODELS:
    print(f"\n=== {name} ({mdir}) ===", flush=True)
    t_wait0 = time.time()
    while not weights_complete(mdir):
        if time.time() - t_wait0 > WAIT_EACH_S:
            print(f"SKIP {name}: weights not complete after {WAIT_EACH_S/60:.0f} min", flush=True)
            break
        print(f"  waiting for weights… ({int(time.time() - t_wait0)}s)", flush=True)
        time.sleep(60)
    if not weights_complete(mdir):
        results[name] = {"error": "weights incomplete"}
        continue

    out_json = os.path.join(ROOT, f"_ab_{name}.json")
    env = dict(os.environ)
    env["QWEN_VLM_DIR"]     = mdir
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [PYEXE, "-u", os.path.join(ROOT, "_ab_vlm_dump.py"), out_json],
            env=env, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=45 * 60,
        )
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-6:])
        print(tail, flush=True)
        if proc.returncode != 0:
            err_tail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
            print(f"FAILED rc={proc.returncode}\n{err_tail}", flush=True)
            results[name] = {"error": f"rc={proc.returncode}: {err_tail[-300:]}"}
            continue
        results[name] = json.load(open(out_json, encoding="utf-8"))
    except subprocess.TimeoutExpired:
        print(f"FAILED: timeout", flush=True)
        results[name] = {"error": "timeout 45min"}
    except Exception as e:
        results[name] = {"error": str(e)}

# ── Comparison table ─────────────────────────────────────────────────────────
import numpy as np

print("\n\n================ TOURNAMENT RESULTS ================", flush=True)
base = results.get("baseline_qwen25_3b", {})
base_scores = {k: v["score"] for k, v in base.get("results", {}).items()} if "results" in base else {}

print(f"{'model':<22} {'s/img':>6} {'load_s':>7} {'mean':>6} {'std':>6} {'blank':>6} {'r_vs_base':>10}")
for name, r in results.items():
    if "error" in r:
        print(f"{name:<22} ERROR: {r['error'][:90]}")
        continue
    scores = {k: v["score"] for k, v in r["results"].items()}
    arr = np.array(list(scores.values()))
    blank = sum(1 for v in r["results"].values() if not v["breakdown"])
    corr = ""
    if base_scores and name != "baseline_qwen25_3b":
        common = [k for k in scores if k in base_scores]
        if len(common) >= 5:
            corr = f"{np.corrcoef([scores[k] for k in common], [base_scores[k] for k in common])[0,1]:.3f}"
    print(f"{name:<22} {r['s_per_img']:>6} {r['t_load']:>7} {arr.mean():>6.3f} {arr.std():>6.3f} {blank:>6} {corr:>10}")

with open(os.path.join(ROOT, "_ab_tournament.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nfull results: _ab_tournament.json", flush=True)
