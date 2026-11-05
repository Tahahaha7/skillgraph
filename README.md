# skillgraph

Bayesian skill ratings in Python, built on an expectation-propagation engine.
TrueSkill 1 is complete; TrueSkill 2 is in progress (see `ROADMAP.md`).

> TrueSkill is a trademark of Microsoft. This project is independent and not
> affiliated with or endorsed by Microsoft.

```python
import skillgraph as sg
from skillgraph import Match, Rating

alice, bob = sg.rate_1vs1(Rating(), Rating())          # alice won
# Rating(mu=29.396, sigma=7.171), Rating(mu=20.604, sigma=7.171)

env = sg.TS1(beta=4.0, draw_probability=0.25)          # custom settings
red, blue = [Rating(), Rating()], [Rating(), Rating()]
red, blue = env.rate([red, blue], ranks=[1, 0], partial_play=[[1, 0.5], [1, 1]], match_weight=0.8)

sg.quality([red, blue])                        # 0..1, higher = more balanced
sg.win_probabilities([red, blue, [alice]])     # P(each team finishes first)
sg.outcome_probability([red, blue], [0, 0])    # P(this exact result), e.g. for log-loss
alice.conservative()                           # mu - 3*sigma, for leaderboards

result = sg.rate_matches(
    [Match([["ann", "ben"], ["cat", "dan"]], ranks=[0, 1]), Match([["ann"], ["cat"]], ranks=[0, 0])],
    with_probabilities=True,
)
result.ratings["ann"], result.probabilities
```

Top-level functions use TrueSkill 1 with default settings, same as `sg.TS1.rate(...)`.
Accuracy guarantees, fixes relative to other implementations, and known
limitations: `docs/ts1.md`.

## Development

```bash
pip install -e ".[dev]"
pytest                   # full suite (~1 min)
pytest -m "not slow"     # quick run (~10 s)
SKILLGRAPH_STRESS=1 pytest   # ~10x more random cases
ruff check . && mypy
```
