# 🧬 drift-point

[![CI](https://github.com/kapildhinwa2010-dev/Draft_ponit_harness/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Draft_ponit_harness/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Draft_ponit_harness/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Draft_ponit_harness/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

## 🎯 Measure WHERE a compressed replay first diverges from the original model

Compression papers say **"99% of activation variance preserved"**.
This harness answers the question that actually matters:

> 🦋 **At which generated token does the compressed model stop behaving like the original?**

**Drift-point** = index of the first generated token where a rank-k linear delta replay disagrees with the original model. **Small number, big story.**

| 🧪 Property | ✅ Value |
|---|---|
| 🖥️ Execution | CPU by default; the CLI does not automatically select CUDA |
| 🚫 No training | Pure measurement, no gradient steps |
| 📦 Bounded | CI tests finish in seconds |
| 🤗 Any model | Llama / Qwen / GPT2-style transformers |

## 🔍 Why this is different

Everyone measures **how much variance** a compression keeps.
Nobody measures **when the behavior breaks**.

drift-point turns *"99% variance preserved"* into *"drift started at token 2"*.

## ⚡ Install

Install dependencies (the current CLI runs on CPU):

    pip install torch
    pip install -r requirements.txt

<details>
<summary>🖥️ CPU-only PyTorch variant (laptops / servers bina GPU)</summary>

    pip install --index-url https://download.pytorch.org/whl/cpu torch
    pip install -r requirements.txt

</details>

The current CLI loads the model and inputs on CPU; installing a CUDA-enabled PyTorch build alone does not enable GPU execution.

## 💻 Usage

Single prompt (defaults: Qwen2.5-0.5B, rank 16, 10 steps):

    python drift_point.py --prompt "Tokyo is the capital of"

<details>
<summary>📦 Interactive batch mode — ek ek line me prompts pao</summary>

    python drift_point.py --interactive --model Qwen/Qwen2.5-0.5B-Instruct --rank 16 --steps 10

    # ya file se:
    python drift_point.py --interactive < prompts.txt

Har line = ek prompt. Sab par drift-point measure hoga, ek hi JSON report me.

</details>

<details>
<summary>🎛️ Full CLI options</summary>

    --model       🤗 HF model id (default: Qwen/Qwen2.5-0.5B-Instruct)
    --rank        🎚️ replay rank (default: 16)
    --steps       📏 generated tokens to compare (default: 10)
    --tokenizer-fallback  🤗 explicit tokenizer name if model-tokenizer loading fails
    --prompt      ❓ prompt, repeatable
    --cal-prompt  📚 calibration prompt, repeatable
    --interactive 📦 batch mode: read prompts from stdin
    --output      💾 path for the JSON report

</details>

Tokenizer loading errors are surfaced by default. Use `--tokenizer-fallback gpt2` only when a different tokenizer is an intentional substitute; the CLI prints a warning when it uses that fallback. Rank must be non-negative and steps must be positive.

Library:

    from drift_point import collect_deltas, build_bases, Replay, measure_drift

## 🧪 Tests

    pytest -q

## 📊 Examples

| 🚀 Script | Kya karta hai |
|---|---|
| `examples/qwen_stress_test.py` | Qwen2.5-0.5B, rank-16, 3 prompts, 10-token stress — drift_report.json |
| `examples/rank_sweep.py` | Rank 4 → 32 sweep: kya rank badhane se drift late hota hai? |

## 📈 What the experiments showed

Rank-16 delta replay, greedy generation (Qwen2.5-0.5B):

- ✅ First 3 tokens all agree — replay perfect lagta hai
- ❌ Then one wrong token cascades — original fact continue karta hai, replay loop me fas jata hai
- 🦋 Drift starts at token 2 on the classic Tokyo prompt
- 🛑 Lookahead + repetition brakes can **delay** drift, not remove it

<details>
<summary>🔬 Honest research note</summary>

drift-point is a **diagnostic, not a fix**. It does not make compressions better; it tells you exactly **when** they stop being faithful. Low drift-point on one prompt does not certify a compression — run the rank sweep and multiple prompts before concluding anything.

</details>

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — ground rule: every claim needs a test or a reproducible example. Honest results only, negative results welcome. 🙏

## 📖 Cite

See [CITATION.cff](CITATION.cff).

## ⚖️ License

MIT — see [LICENSE](LICENSE).
