# Changelog

## 0.1.0 (unreleased)

- TrueSkill 1 on a hardened EP engine (stable truncation math, subtraction-free messages,
  scale-free convergence, damping fallback). See `docs/ts1.md`.
- API: `sg.rate` / `sg.TS1.rate` (defaults) and `sg.TS1(...).rate` (custom); `rate_1vs1`,
  `rate_matches` (returns `BatchResult`), `quality`, `win_probabilities`, `outcome_probability`.
- Per-player `partial_play` and per-match `match_weight`.
- Requires Python >= 3.10.
