"""Qwen2.5-0.5B rank-16 stress test for the drift-point metric.

Measures 10-token greedy generation with and without replay on classic prompts
and saves drift_report.json.

Run:  python examples/qwen_stress_test.py
"""

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


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="float32")
    model.eval()

    store = collect_deltas(model, tok, PROMPTS)
    bases = build_bases(store, rank=16)
    replay = Replay(model, bases)
    results = [measure_drift(model, tok, p, replay, steps=10) for p in PROMPTS]
    replay.remove()

    print("model:", MODEL, "rank: 16, steps: 10")
    for r in results:
        print("-" * 60)
        print("prompt :", r["prompt"])
        print("original:", r["original"])
        print("replay  :", r["replay"])
        print("drift_point:", r["drift_point"], "| tokens_agree:", r["tokens_agree"], "/", r["steps"])

    with open("drift_report.json", "w") as f:
        json.dump({"model": MODEL, "rank": 16, "results": results}, f, indent=2)
    print("-" * 60)
    print("saved: drift_report.json")


if __name__ == "__main__":
    main()
