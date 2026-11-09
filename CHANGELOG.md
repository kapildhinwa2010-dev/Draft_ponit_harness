# 📒 Changelog

## [0.1.0] - 2026-11-09

### ✨ Added
- Core drift-point metric: first generated token where a rank-k delta replay diverges from the original model.
- Toggleable per-layer delta replay hooks (SVD basis, mean-centered).
- CLI with model / rank / steps / prompt options, JSON report output, and 📦 interactive batch mode (stdin).
- Bounded CI smoke tests on a tiny random GPT-2.
- CI pytest logs are retained as downloadable workflow artifacts.
- 🚀 Examples: Qwen2.5-0.5B stress test (rank 16) and rank sweep (drift vs rank curve).
- 🛡️ Production infra: pyproject, CHANGELOG, CITATION.cff, CodeQL, dependabot, issue/PR templates.
