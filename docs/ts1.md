# TrueSkill 1: fixes, guarantees, limitations

## Fixes relative to existing implementations

| Problem | Reference | Here |
|---|---|---|
| Extreme upsets crash | sublee/trueskill #5 | Stable at 10^12-sd upsets |
| Textbook `1 - w` cancels: variance errors of 100% (big upsets) to 3,000,000x (small draw probability) | common implementations | Regime-specific formulas, <= 2e-10 relative vs 80-digit mpmath |
| Wrong 3x3+ matrix inverse | #47 | NumPy; `quality` uses log-determinants (no overflow with many teams) |
| Odd ratings in large FFAs | #23 | Reproduced setup: rank corr. 0.999 with true skill; matches dense EP to 1e-12 |
| Convergence check not scale-free | original design | Mean shift in sds / relative sd change |
| Cavities by subtracting large precisions (negative variances) | original design | Products of the other messages; no Gaussian division anywhere |
| EP oscillation on contradictory evidence | any EP | Damping fallback; converges up to 1e5-sd gaps, else valid ratings + `ConvergenceWarning` |
| Results drift between versions | #33 | Golden-file test (300 cases, inputs + outputs) |
| Player on both teams | #57 | `rate_matches` rejects duplicate ids |
| Win probability for 2 teams only | #24, #32, #35, #52 | `win_probabilities` (any n), `outcome_probability` (incl. draws) |
| No per-match weight | #59 | `match_weight` |
| No batch API / slow | #26, #39 | `rate_matches`; closed form for 2 teams (~79k 1v1/s) |

## Accuracy (enforced by tests)

- 2 teams: exact closed form; matches the factor graph to 1e-9 and an all-mpmath closed form to 1e-10 (output-sigma units).
- 3+ teams: EP; matches an independent dense EP to 1e-8; default `min_delta` within 1e-3 sigma of converged;
  ~0.05 sigma from the exact posterior (Monte Carlo).
- `win_probabilities`: ~1e-15 up to 25 teams; seeded Monte Carlo above (SE <= 0.0016).
- `outcome_probability`: 2 teams exact and log-stable; 3 teams ~1e-15 and 4-12 teams ~1e-7 *absolute*
  (rarer outcomes can return 0); > 12 teams unsupported.
- Exact symmetries: identical-player draws keep means; mirrored matches mirror; deterministic output.

## Model limitations (not bugs)

- Slightly overconfident sigmas: ~60% of true skills within 1 sigma (ideal 68%); 2-sigma coverage ~95%.
- Results depend on match order (one match at a time, by design).
- With 3+ teams, ties among more than two teams depend on input order (only consecutive finishers are compared).
- Under the default `draw_probability=0.10` a draw is surprising; set it to your observed draw rate.
- Predictions use the match-time predictive (sigma^2 + tau^2 + beta^2 per player); `quality` keeps the classic
  definition without tau.

Throughput (one core): ~79k 1v1/s, ~33k 5v5/s, ~850 8-player FFAs/s, ~36 100-player FFAs/s.
