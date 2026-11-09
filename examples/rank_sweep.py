"""Rank sweep: how does drift-point move as the replay rank grows?

Measures drift on a few prompts at ranks 4/8/16/32 and prints a compact table:
rank | avg tokens agreeing with the original | per-prompt drift points.

Run:  python examples/rank_sweep.py --steps 8
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer

from drift_point import Replay, build_bases, collect_deltas, measure_drift

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "Tokyo is the capital of",
    "The sun rises in the",
    "Water boils at",
]
RANKS = [4, 8, 16, 32]


def main():
    ap = argparse.ArgumentParser(prog="rank-sweep")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--output", default="rank_sweep.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype="float32")
    model.eval()

    store = collect_deltas(model, tok, PROMPTS)
    report = []
    for rank in RANKS:
        bases = build_bases(store, rank)
        replay = Replay(model, bases)
        results = [measure_drift(model, tok, p, replay, steps=a.steps) for p in PROMPTS]
        replay.remove()
        agree = sum(r["tokens_agree"] for r in results) / len(results)
        pts = [r["drift_point"] for r in results]
        report.append({"rank": rank, "avg_tokens_agree": agree, "drift_points": pts})
        print("rank " + str(rank).rjust(3) + " | avg agree " + str(round(agree, 2)) + " | drift points " + str(pts))

    with open(a.output, "w") as f:
        json.dump({"model": a.model, "steps": a.steps, "ranks": report}, f, indent=2)
    print("saved: " + a.output)


if __name__ == "__main__":
    main()
