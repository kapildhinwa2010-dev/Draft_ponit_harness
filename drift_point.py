"""drift-point: measure WHERE a compressed replay first diverges from the original model.

A rank-k linear "delta replay" of an LM can capture ~99% of activation variance
yet still collapse mid-generation. Drift-point is the index of the first
generated token where the replay and the original model disagree.

Small number, big story: instead of "99% variance preserved", you get
"drift started at token 2".

CPU-only. No training. Bounded evaluation.
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layers(model):
    """Return the decoder layer list for Llama/Qwen/GPT2-style models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("unsupported architecture: no decoder layer list found")


@torch.no_grad()
def collect_deltas(model, tok, prompts, max_len=64):
    """Record (output - input) deltas for every decoder layer on cal prompts."""
    layers = get_layers(model)
    store = {i: [] for i in range(len(layers))}
    handles = []

    def make_hook(idx):
        def hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            y = out[0] if isinstance(out, tuple) else out
            d = y - x
            store[idx].append(d.reshape(-1, d.shape[-1]).float().cpu())
        return hook

    for i, lyr in enumerate(layers):
        handles.append(lyr.register_forward_hook(make_hook(i)))
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids[:, :max_len]
        model(ids)
    for h in handles:
        h.remove()
    return store


def build_bases(store, rank):
    """Per-layer SVD basis of the delta space (mean + top-rank directions)."""
    bases = {}
    for i, chunks in store.items():
        D = torch.cat(chunks, dim=0)
        mu = D.mean(dim=0)
        A = D - mu
        U, S, V = torch.linalg.svd(A, full_matrices=False)
        bases[i] = {"mu": mu, "V": V[:rank].T.contiguous()}
    return bases


class Replay:
    """Toggleable per-layer delta replay: reconstruct each layer delta via its basis."""

    def __init__(self, model, bases):
        self.on = False
        self.handles = []

        def make_hook(b):
            def hook(mod, inp, out):
                if not self.on:
                    return out
                x = inp[0] if isinstance(inp, tuple) else inp
                y = out[0] if isinstance(out, tuple) else out
                d = y - x
                flat = d.reshape(-1, d.shape[-1]).float()
                rec = (flat - b["mu"]) @ b["V"] @ b["V"].T + b["mu"]
                new = x + rec.reshape(x.shape).to(x.dtype)
                if isinstance(out, tuple):
                    return (new,) + out[1:]
                return new
            return hook

        for i, lyr in enumerate(get_layers(model)):
            self.handles.append(lyr.register_forward_hook(make_hook(bases[i])))

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def greedy(model, tok, prompt, steps):
    """Greedy step-by-step generation; returns list of generated token ids."""
    ids = tok(prompt, return_tensors="pt").input_ids
    gen = []
    for _ in range(steps):
        logits = model(ids).logits[:, -1, :]
        nxt = int(logits.argmax(dim=-1).item())
        gen.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return gen


def measure_drift(model, tok, prompt, replay, steps=10):
    """Generate with replay OFF (reference) and ON (replay); find first mismatch."""
    replay.on = False
    ref = greedy(model, tok, prompt, steps)
    replay.on = True
    rep = greedy(model, tok, prompt, steps)
    replay.on = False

    drift = None
    match = []
    for i in range(len(ref)):
        same = ref[i] == rep[i]
        match.append(bool(same))
        if not same and drift is None:
            drift = i

    return {
        "prompt": prompt,
        "original": tok.decode(ref),
        "replay": tok.decode(rep),
        "drift_point": drift,
        "steps": steps,
        "match": match,
        "tokens_agree": int(sum(match)),
    }


def main():
    ap = argparse.ArgumentParser(
        prog="drift-point",
        description="Measure the first token where a rank-k delta replay diverges from the original model.",
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--prompt", action="append", help="prompt to stress; repeatable")
    ap.add_argument("--cal-prompt", action="append", help="calibration prompt; repeatable")
    ap.add_argument("--output", help="optional path for the JSON report")
    a = ap.parse_args()

    prompts = a.prompt or ["Tokyo is the capital of"]
    cal = a.cal_prompt or prompts

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()

    store = collect_deltas(model, tok, cal)
    bases = build_bases(store, a.rank)
    replay = Replay(model, bases)
    results = [measure_drift(model, tok, p, replay, a.steps) for p in prompts]
    replay.remove()

    out = {"model": a.model, "rank": a.rank, "results": results}
    text = json.dumps(out, indent=2)
    print(text)
    if a.output:
        with open(a.output, "w") as f:
            f.write(text + chr(10))


if __name__ == "__main__":
    main()
