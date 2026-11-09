# drift-point

[![CI](https://github.com/kapildhinwa2010-dev/Draft_ponit_harness/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Draft_ponit_harness/actions/workflows/ci.yml)

**Measure WHERE a compressed replay first diverges from the original model.**

Compression papers report "99% of activation variance preserved". This harness
answers the question that actually matters: at which generated token does the
compressed model stop behaving like the original?

**Drift-point** = the index of the first generated token where a rank-k linear
"delta replay" disagrees with the original model. Small number, big story.

- CPU-only, no training, bounded evaluation
- Works with any Llama/Qwen/GPT2-style transformers model
- Per-token match table + JSON report
- Bounded CI tests with self-diagnosing logs

## Why this is different

Everyone measures how much variance a compression keeps. Nobody measures
*when* the behavior breaks. Drift-point turns "99% variance preserved" into
"drift started at token 2".

## Install

    pip install --index-url https://download.pytorch.org/whl/cpu torch
    pip install -r requirements.txt

## Usage

CLI (defaults to Qwen2.5-0.5B, rank 16, 10 steps):

    python drift_point.py --prompt "Tokyo is the capital of"

Custom model and rank, JSON report:

    python drift_point.py --model Qwen/Qwen2.5-0.5B-Instruct --rank 16 --steps 10 --output report.json

Library:

    from drift_point import collect_deltas, build_bases, Replay, measure_drift

## Tests

    pytest -q

CI uses a tiny random GPT-2 (seconds, bounded); the example script runs the
full Qwen2.5-0.5B replication locally on CPU.

## What the experiments showed (Reverse.Dean)

Rank-16 delta replay on Qwen2.5-0.5B, greedy generation:

- First 3 tokens can all agree (replay looks perfect)
- Then a single wrong token cascades: original continues a fact, replay loops
  ("Tokyo is the capital of Japan ... located in Tokyo")
- Drift starts at token 2 on the classic Tokyo prompt; lookahead and
  repetition brakes can delay it, but not remove it

Honest research note: drift-point is a *diagnostic*, not a fix. It does not
make compressions better; it tells you exactly when they stop being faithful.
See `examples/reverse_dean_replication.py` to reproduce.

## License

MIT
