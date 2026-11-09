# Changelog

## [0.1.0] - 2026-11-09

### Added
- Core drift-point metric: first generated token where a rank-k delta replay diverges from the original model.
- Toggleable per-layer delta replay hooks (SVD basis, mean-centered).
- CLI with model / rank / steps / prompt options and JSON report output.
- Bounded CI smoke tests on a tiny random GPT-2.
- Self-diagnosing CI logs (ci_logs/pytest.log committed by CI).
- Examples: Reverse.Dean Spiral-DNA replication (Qwen2.5-0.5B, rank 16) and rank sweep (drift-point vs rank curve).
