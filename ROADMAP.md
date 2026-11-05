# Roadmap

- [x] EP engine and TrueSkill 1, hardened against independent oracles (`docs/ts1.md`).
- [ ] TrueSkill 2 (`sg.TS2`): time-based skill drift, experience, individual stats,
      squads, quitting, cross-mode skill; each with an ablation test.
- [ ] Offline parameter fitting (`sg.TS2.fit(history)`), maximizing predictive log-likelihood.
- [ ] Evaluation harness: accuracy, log-loss, calibration vs TS1 / OpenSkill / Elo.
      Needs log-stable outcome probabilities for 3+ teams (EP evidence).
- [ ] Datasets: Dota 2 (OpenDota) for validation, tennis / EPL showcases. PyPI release.

## References

- TrueSkill 1: Herbrich, Minka & Graepel, NIPS 2006.
- TrueSkill 2: Minka, Cleven & Zaykov, MSR-TR-2018-8.
  https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/trueskill2.pdf
- Infer.NET (MIT) can generate reference outputs for TS2 tests.

## Legal (not legal advice)

A quick search found no TS2-specific patent, but that is not a clearance search.
Microsoft holds TS1-era patents (e.g. US8538910, priority 2006); some may still be
in force. Before commercial use, run a proper search (assignee Microsoft;
inventors Minka, Zaykov, Herbrich, Graepel) or ask a patent attorney.
Don't use "TrueSkill" in the package name; keep the non-affiliation notice.
