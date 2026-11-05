"""Numerical settings and limits, in one place.

Changing any value here can change results; tests/test_regression.py will flag it.
"""

# Truncated-Gaussian moments (_engine/truncation.py)
SERIES_BELOW_X = -10.0  # win case: asymptotic series below this standardized difference
SERIES_TERMS = 24  # terms in that series; truncation error < 1e-18 at the threshold
DRAW_QUAD_NODES = 96  # Gauss-Legendre nodes for draws with small tilt/curvature
DRAW_QUAD_LIMIT = 50.0  # largest tilt/curvature handled by that quadrature

# Expectation propagation (_engine/factorgraph.py)
DAMPING_SCHEDULE = (1.0, 0.5, 0.2, 0.05)  # tried in order until EP converges

# Two-team closed form (ts1.py)
EXACT_LEAVE_ONE_OUT_MAX = 256  # up to this many players, sum "everyone but i" exactly

# Predictions (_model.py, _engine/probability.py)
EXACT_WIN_PROB_MAX_TEAMS = 25  # above this, win_probabilities uses Monte Carlo
OUTCOME_PROB_MAX_TEAMS = 12  # outcome_probability raises above this
WIN_PROB_SPAN = 12.0  # integrate +-12 sd of the candidate team's performance
WIN_PROB_STEP_OFFSETS = (-16.0, -4.0, -1.0, 0.0, 1.0, 4.0, 16.0)  # breakpoints, in step widths
NEGLIGIBLE_Z = -40.0  # a team this many sd behind everywhere has probability 0
MC_SAMPLES = 100_000  # default Monte Carlo samples (standard error <= 0.0016)
MC_CHUNK_ELEMENTS = 4_000_000  # samples x teams per chunk (~32 MB)
MVN_ABSEPS = 1e-8  # multivariate normal box probability (4+ teams) ...
MVN_RELEPS = 1e-6
MVN_MAXPTS_PER_DIM = 50_000
