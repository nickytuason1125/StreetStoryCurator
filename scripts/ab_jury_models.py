"""
A/B the jury LLM: is a smaller model as good as deepseek-r1-8b-q5?

R1-8B-Q5 is 5.4 GB, the largest live weight in the project. A 3-4B at Q5 is
~2-2.5 GB, so a swap reclaims more than the entire disk-audit exercise did -
but only if quality holds.

WHAT MAKES THIS MEASURABLE
--------------------------
Jury verdicts are prose, which normally means no automatic metric. This one has
two, because the existing pipeline already validates itself:

  grounding   every verdict may cite {slot, aspect, value}, and
              signal_validator.validate_claims checks that citation against the
              REAL per-image data. A model that invents evidence fails.
  spread      three personas score the same sequence independently; a large
              spread means the model is not reading the data, it is guessing.

Grounding alone is gameable: a model that never cites anything passes
vacuously. So the headline number is GROUNDED CITATION RATE - citations that
validate, over all verdicts - which punishes hallucinating and refusing to cite
equally. `cite_rate` and `hallucination_rate` are reported separately so a tie
can be broken by looking at which way a model failed.

FAIRNESS
--------
Only model_path varies. Same llama.cpp, same GBNF grammar, same personas, same
prompts, same sequences, same n_ctx, same max_tokens, same seed. Candidates come
from Ollama's blob store, so nothing is downloaded and the Ollama server - which
was rejected for grading as CPU-bound - is never involved.

Usage:
    python scripts/ab_jury_models.py --sequences 8
    python scripts/ab_jury_models.py --sequences 8 --models llama3.2:latest gemma3:4b
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

OLLAMA = Path.home() / ".ollama" / "models"
ASPECTS = ["Composition", "Lighting", "Narrative", "Human/Culture", "Technical"]
ROLES = ["opener", "subject", "contrast", "detail", "closer"]


# ── candidates ────────────────────────────────────────────────────────────────

def ollama_ggufs() -> dict[str, Path]:
    """Map ollama model name -> GGUF blob on disk."""
    out: dict[str, Path] = {}
    lib = OLLAMA / "manifests" / "registry.ollama.ai" / "library"
    if not lib.exists():
        return out
    for man in lib.rglob("*"):
        if not man.is_file():
            continue
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
        except Exception:
            continue
        for layer in m.get("layers", []):
            if "model" in layer.get("mediaType", ""):
                blob = OLLAMA / "blobs" / layer["digest"].replace(":", "-")
                if blob.exists():
                    out[f"{man.parent.name}:{man.name}"] = blob
    return out


# ── fixed evaluation set ──────────────────────────────────────────────────────

def build_sequences(n_seq: int, slots: int = 5, seed: int = 1234) -> list[dict]:
    """Sequences of REAL graded photos, with their real aspect values.

    Real data matters: the grounding metric checks a model's cited value against
    the actual number, so synthetic aspects would measure nothing.
    """
    cat = json.loads((_ROOT / "cache" / "catalog.json").read_text(encoding="utf-8"))
    rows = cat if isinstance(cat, list) else cat.get("photos", cat)

    usable = []
    for r in rows:
        bd = r.get("breakdown") or {}
        if isinstance(bd, str):
            try:
                bd = json.loads(bd)
            except Exception:
                continue
        vals = {a: bd[a] for a in ASPECTS if isinstance(bd.get(a), (int, float))}
        if len(vals) == len(ASPECTS):
            usable.append({"filename": Path(r.get("path", "")).name,
                           "score": float(r.get("score", 0.5)), **vals})
    rng = random.Random(seed)
    rng.shuffle(usable)

    seqs = []
    for i in range(n_seq):
        chunk = usable[i * slots:(i + 1) * slots]
        if len(chunk) < slots:
            break
        seqs.append({
            "images": chunk,
            "roles": ROLES[:slots],
            "scores": [c["score"] for c in chunk],
            "aspects_by_slot": [{a: c[a] for a in ASPECTS} for c in chunk],
        })
    return seqs


# ── running one model ─────────────────────────────────────────────────────────

def run_model(name: str, gguf: Path, seqs: list[dict], n_ctx: int = 1024) -> dict:
    import jury_engine as je
    from signal_validator import Claim, validate_claims
    from llama_cpp import Llama

    try:
        from tier_select import has_gpu          # subprocess probe, never in-process
        n_gpu = -1 if has_gpu() else 0
    except Exception:
        n_gpu = 0

    t0 = time.monotonic()
    llm = None
    for rung in ([-1, 24, 16, 8, 0] if n_gpu else [0]):
        try:
            llm = Llama(model_path=str(gguf), n_ctx=n_ctx, n_gpu_layers=rung,
                        flash_attn=(rung != 0), verbose=False, seed=1234)
            break
        except Exception:
            continue
    if llm is None:
        return {"model": name, "error": "failed to load at every offload level"}
    load_s = time.monotonic() - t0

    grammar = je._load_grammar()
    n_verdicts = n_parsed = n_cited = n_valid = 0
    spreads: list[float] = []
    lat: list[float] = []
    # Judgment, as distinct from data-reading. Grounding only asks whether a
    # cited number is real; a model could score 100% on it while emitting an
    # identical verdict score for every sequence. These two catch that:
    #   discrimination  spread of a model's own scores ACROSS sequences. Near
    #                   zero means it is not judging, it is pattern-matching.
    #   calibration     correlation between its score and the mean grade of the
    #                   photos in the sequence. Near zero means it is not
    #                   reading the evidence it was handed.
    seq_mean_scores: list[float] = []
    seq_jury_scores: list[float] = []

    for seq in seqs:
        slot_summary = je._build_slot_summary(seq["images"], seq["roles"], seq["scores"])
        scores_this_seq = []
        for persona in je._PERSONAS:
            n_verdicts += 1
            prompt = (
                f"{persona['system']}\n"
                f"ROLE: {persona['name']} juror delivering a verdict on a curated street-photo sequence.\n"
                f"Style brief: 'candid street work, natural light'. Theme: street. Color: color.\n"
                f"Sequence (slot:role:file:score:aspects): {slot_summary}\n"
                "Give a one-sentence verdict, a 0.00-1.00 score, and cite the specific slot/aspect/value "
                "driving your score (or \"none\"/null if purely qualitative).\n"
                'Output ONLY JSON: {"verdict":"...","score":0.00,"cited_aspect":'
                '"Composition|Lighting|Narrative|Human/Culture|Technical|none","cited_slot":<int or null>,'
                '"cited_value":<float or null>}'
            ) + je._THINK_SKIP

            # Mirror production exactly, including the unconstrained retry. An
            # earlier version of this harness only tried the grammar and
            # swallowed failures, which reported 0% parse for EVERY model -
            # including the one in production. A harness that scores the
            # current model at zero is measuring itself, not the models.
            t1 = time.monotonic()
            kw = dict(max_tokens=200, temperature=persona["temp"])
            raw = None
            if grammar is not None:
                try:
                    raw = llm(prompt, grammar=grammar, **kw)["choices"][0]["text"]
                except Exception:
                    raw = None
            if raw is None:
                try:
                    raw = llm(prompt, **kw)["choices"][0]["text"]
                except Exception:
                    lat.append(time.monotonic() - t1)
                    continue
            lat.append(time.monotonic() - t1)

            v = je._parse_verdict(raw, persona["name"])
            if not v:
                continue
            n_parsed += 1
            scores_this_seq.append(v["score"])

            if v["cited_aspect"] and v["cited_value"] is not None:
                n_cited += 1
                res = validate_claims(
                    [Claim(text=v["verdict"], cited_aspect=v["cited_aspect"],
                           cited_value=v["cited_value"], cited_slot=v["cited_slot"])],
                    seq["aspects_by_slot"])
                if res.passed:
                    n_valid += 1

        if len(scores_this_seq) >= 2:
            spreads.append(max(scores_this_seq) - min(scores_this_seq))
        if scores_this_seq:
            seq_jury_scores.append(sum(scores_this_seq) / len(scores_this_seq))
            seq_mean_scores.append(sum(seq["scores"]) / len(seq["scores"]))

    del llm
    import gc
    gc.collect()

    def _stdev(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    def _pearson(xs: list[float], ys: list[float]) -> float:
        if len(xs) < 3:
            return float("nan")
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx = sum((a - mx) ** 2 for a in xs) ** 0.5
        dy = sum((b - my) ** 2 for b in ys) ** 0.5
        return num / (dx * dy) if dx and dy else float("nan")

    return {
        "model": name,
        "size_gb": gguf.stat().st_size / 1e9,
        "load_s": load_s,
        "verdicts": n_verdicts,
        "parse_rate": n_parsed / n_verdicts if n_verdicts else 0.0,
        "cite_rate": n_cited / n_verdicts if n_verdicts else 0.0,
        # judgment metrics
        "discrimination": _stdev(seq_jury_scores),
        "calibration": _pearson(seq_jury_scores, seq_mean_scores),
        # headline: citations that validate, over ALL verdicts. Punishes
        # hallucinating and refusing to cite equally.
        "grounded_cite_rate": n_valid / n_verdicts if n_verdicts else 0.0,
        "hallucination_rate": (n_cited - n_valid) / n_cited if n_cited else 0.0,
        "mean_spread": sum(spreads) / len(spreads) if spreads else float("nan"),
        "s_per_verdict": sum(lat) / len(lat) if lat else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", type=int, default=8)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cands = ollama_ggufs()
    baseline = _ROOT / "models" / "deepseek-r1-8b-q5.gguf"
    if baseline.exists():
        cands["deepseek-r1:8b-q5 (CURRENT)"] = baseline
    if args.models:
        cands = {k: v for k, v in cands.items()
                 if any(m in k for m in args.models) or "CURRENT" in k}

    seqs = build_sequences(args.sequences)
    print(f"[ab] {len(seqs)} sequences x {len(cands)} models x 3 personas "
          f"= {len(seqs) * len(cands) * 3} verdicts\n")

    # Write after EVERY model, not at the end. A run this long (CPU inference
    # across several GB-scale models) must not lose everything to a crash on
    # the last candidate, and a partial file is how progress is observed while
    # it runs.
    results = []
    out_path = Path(args.out) if args.out else None
    for i, (name, gguf) in enumerate(cands.items(), 1):
        print(f"[ab] ({i}/{len(cands)}) {name} ({gguf.stat().st_size/1e9:.2f} GB) …",
              flush=True)
        try:
            r = run_model(name, gguf, seqs)
        except Exception as e:
            r = {"model": name, "error": str(e)}
        results.append(r)
        print(f"     {r}\n", flush=True)
        if out_path:
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'MODEL':<28} {'GB':>5} {'PARSE':>6} {'CITE':>6} {'GROUND':>7} "
          f"{'HALLUC':>7} {'DISCRIM':>8} {'CALIB':>7} {'SPREAD':>7} {'S/VERD':>7}")
    for r in sorted(results, key=lambda x: -x.get("grounded_cite_rate", -1)):
        if "error" in r:
            print(f"  {r['model']:<26} ERROR: {r['error'][:60]}")
            continue
        print(f"  {r['model']:<26} {r['size_gb']:5.2f} {r['parse_rate']:6.0%} "
              f"{r['cite_rate']:6.0%} {r['grounded_cite_rate']:7.0%} "
              f"{r['hallucination_rate']:7.0%} {r['discrimination']:8.3f} "
              f"{r['calibration']:7.2f} {r['mean_spread']:7.3f} "
              f"{r['s_per_verdict']:7.2f}")
    print("\n  DISCRIM = stdev of the model's own scores across sequences "
          "(~0 means it is not judging)")
    print("  CALIB   = correlation of its score with the sequences' real mean "
          "photo grade (~0 means it is not reading)")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
