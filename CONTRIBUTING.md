# Contributing

Thanks for contributing to RNX.

## Development Flow

1. Create a feature branch from `master`.
2. Make focused, reviewable commits.
3. Keep notebook and mirrored `.py` script versions aligned when changing pipeline logic.
4. Open a pull request with:
- a short summary,
- why the change helps quality/runtime,
- any Kaggle/runtime tradeoffs.

## Coding Expectations

- Prefer clear and reproducible behavior over micro-optimizations.
- Keep RNG behavior deterministic where feasible.
- Preserve Kaggle-safe defaults unless explicitly changing competition strategy.

## Validation Checklist

- Notebook runs without syntax/runtime regressions in a Kaggle-like environment.
- Submission schema remains unchanged.
- Long-target behavior remains bounded for runtime safety.
