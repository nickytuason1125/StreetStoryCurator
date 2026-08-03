"""
Verdict PROSE quality, for the finalists of ab_jury_models.py.

The metrics pass answers "is it reliable" - valid JSON, real citations, scores
that move with the evidence. It cannot answer "is the writing any good", and a
model can pass every numeric check while producing boilerplate:

    "This sequence is technically proficient but lacks emotional depth."
    "This sequence is technically proficient but lacks emotional depth."
    "This sequence is technically proficient but lacks emotional depth."

Perfect grounding, perfect schema, useless jury.

Half of that IS automatable. A verdict that says nothing specific says the same
nothing every time, so DISTINCTNESS - how different a model's verdicts are from
EACH OTHER across different sequences - catches boilerplate without a human.
The other half (is the observation insightful, is the prose any good) is not
automatable and should not be faked with a proxy metric, so this script dumps
the verdicts BLIND, labelled Model A/B/C, for a human read.

Usage:
    python scripts/ab_jury_prose.py --models qwen3.5:4b phi4-mini --sequences 4
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import ab_jury_models as ab          # noqa: E402


def distinctness(verdicts: list[str]) -> float:
    """1.0 = every verdict unlike the others; 0.0 = the same sentence each time.

    Mean pairwise difflib ratio, inverted. Crude, but it separates "wrote
    something about this sequence" from "emitted its house style again", which
    is the distinction that matters.
    """
    texts = [v.strip().lower() for v in verdicts if v and v.strip()]
    if len(texts) < 2:
        return float("nan")
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(difflib.SequenceMatcher(None, texts[i], texts[j]).ratio())
    return 1.0 - (sum(sims) / len(sims))


def collect(name: str, gguf: Path, seqs: list[dict]) -> list[dict]:
    """One verdict per sequence, Purist persona only (fixed, low temperature),
    so differences are the model's, not sampling noise across personas."""
    import jury_engine as je
    from llama_cpp import Llama
    try:
        from tier_select import has_gpu
        rungs = [-1, 24, 16, 8, 0] if has_gpu() else [0]
    except Exception:
        rungs = [0]

    llm = None
    for rung in rungs:
        try:
            llm = Llama(model_path=str(gguf), n_ctx=1024, n_gpu_layers=rung,
                        flash_attn=(rung != 0), verbose=False, seed=1234)
            break
        except Exception:
            continue
    if llm is None:
        return []

    grammar = je._load_grammar()
    persona = je._PERSONAS[0]
    out = []
    for seq in seqs:
        ss = je._build_slot_summary(seq["images"], seq["roles"], seq["scores"])
        prompt = (
            f"{persona['system']}\n"
            f"ROLE: {persona['name']} juror delivering a verdict on a curated street-photo sequence.\n"
            f"Style brief: 'candid street work, natural light'. Theme: street. Color: color.\n"
            f"Sequence (slot:role:file:score:aspects): {ss}\n"
            "Give a one-sentence verdict, a 0.00-1.00 score, and cite the specific slot/aspect/value "
            "driving your score (or \"none\"/null if purely qualitative).\n"
            'Output ONLY JSON: {"verdict":"...","score":0.00,"cited_aspect":'
            '"Composition|Lighting|Narrative|Human/Culture|Technical|none","cited_slot":<int or null>,'
            '"cited_value":<float or null>}'
        ) + je._THINK_SKIP
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
                continue
        v = je._parse_verdict(raw, persona["name"])
        if v:
            out.append({"seq_mean": sum(seq["scores"]) / len(seq["scores"]), **v})
    del llm
    import gc
    gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    all_ggufs = ab.ollama_ggufs()
    baseline = _ROOT / "models" / "deepseek-r1-8b-q5.gguf"
    if baseline.exists():
        all_ggufs["deepseek-r1:8b-q5 (CURRENT)"] = baseline

    chosen = {k: v for k, v in all_ggufs.items()
              if any(m in k for m in args.models) or "CURRENT" in k}
    seqs = ab.build_sequences(args.sequences)

    results = {}
    for name, gguf in chosen.items():
        print(f"[prose] {name} …", flush=True)
        results[name] = collect(name, gguf, seqs)

    print("\n=== DISTINCTNESS (1.0 = every verdict different, 0.0 = boilerplate) ===")
    for name, vs in results.items():
        d = distinctness([v["verdict"] for v in vs])
        print(f"  {name:<30} {d:5.2f}   ({len(vs)} verdicts)")

    # Blind dump. Labels are assigned by shuffling, and the key is printed LAST
    # so it can be scrolled past - reading the verdicts with the model names
    # visible is not a blind comparison.
    names = list(results)
    rng = random.Random(99)
    rng.shuffle(names)
    labels = {n: chr(ord("A") + i) for i, n in enumerate(names)}

    print("\n=== BLIND VERDICTS ===")
    for si in range(args.sequences):
        print(f"\n--- sequence {si} (real mean grade "
              f"{results[names[0]][si]['seq_mean']:.2f}) ---"
              if si < len(results[names[0]]) else f"\n--- sequence {si} ---")
        for n in sorted(names, key=lambda x: labels[x]):
            vs = results[n]
            if si < len(vs):
                v = vs[si]
                print(f"  [{labels[n]}] {v['score']:.2f}  {v['verdict']}")

    print("\n=== KEY ===")
    for n in sorted(names, key=lambda x: labels[x]):
        print(f"  {labels[n]} = {n}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"results": results, "labels": labels}, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
