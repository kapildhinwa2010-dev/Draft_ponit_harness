# Contributing

Thanks for your interest in drift-point.

## Ground rules

- Keep the project CPU-only and bounded: no training jobs, no giant downloads in CI.
- Every claim needs a test or a reproducible example. No "production-ready" labels without proof.
- Honest results only: negative results are welcome, marketing is not.

## How to contribute

1. Fork the repo and create a branch.
2. Make your change, add or update a test under tests/.
3. Run: pytest -q
4. Open a pull request using the template.

## Reporting problems

Open an issue with: model id, rank, prompt, drift-point report JSON, and the pytest log if a test failed.
