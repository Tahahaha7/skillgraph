"""Independent reference implementations used only by the tests.

None of these share code with the library's inference path:
- mpmath (arbitrary precision) truncated-normal moments;
- a closed-form two-team update (Herbrich et al. 2006, eqs. for one truncation);
- a dense, matrix-based EP over team differences with rank-one updates;
- Monte Carlo rejection sampling of the exact posterior.
"""

import math

import mpmath as mp
import numpy as np
from scipy.special import ndtr


# ----------------------------------------------------------- mpmath moments
def mp_win(x, dps=80):
    """(v, 1-w) for N(0,1) conditioned on z > -x."""
    with mp.workdps(dps):
        x = mp.mpf(x)
        V = mp.npdf(x) / mp.ncdf(x)
        return V, 1 - V * (V + x)


def mp_draw(t, eps, dps=60):
    """(mean, var) of z ~ N(0,1) truncated to [-eps - t, eps - t], by quadrature."""
    with mp.workdps(dps):
        t, eps = mp.mpf(t), mp.mpf(eps)
        lo, hi = -eps - t, eps - t
        x0 = mp.mpf(0) if lo <= 0 <= hi else (hi if hi < 0 else lo)
        scale = 1 / max(abs(x0), mp.mpf(1))
        pts = {lo, hi}
        for k in (0, 1, 3, 10, 30, 100, 300):
            for p in (x0 - k * scale, x0 + k * scale):
                if lo < p < hi:
                    pts.add(p)
        pts = sorted(pts)

        def g(x):
            return mp.exp(-(x * x - x0 * x0) / 2)

        z = mp.quad(g, pts)
        m = mp.quad(lambda x: x * g(x), pts) / z
        var = mp.quad(lambda x: (x - m) ** 2 * g(x), pts) / z
        return m, var


class OutOfRange(Exception):
    """The case is too extreme for the simple oracle; use the mpmath oracle."""


# ------------------------------------------------------ truncated moments
def _mp_moments(mean, var, lo, hi):
    sd = math.sqrt(var)
    if math.isinf(hi):
        v, omw = mp_win((mean - lo) / sd, dps=60)
        return mean + sd * float(v), var * float(omw)
    c = (lo + hi) / 2
    m_, v_ = mp_draw((mean - c) / sd, (hi - lo) / (2 * sd), dps=40)
    return mean + sd * float(m_), var * float(v_)


def _trunc_moments(mean, var, lo, hi, allow_mp=True, force_mp=False):
    """Classic textbook truncated-normal moments; mpmath fallback in the tails."""
    if force_mp:
        return _mp_moments(mean, var, lo, hi)
    sd = math.sqrt(var)
    a, b = (lo - mean) / sd, (hi - mean) / sd
    pa = 0.0 if math.isinf(a) else math.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    pb = 0.0 if math.isinf(b) else math.exp(-0.5 * b * b) / math.sqrt(2 * math.pi)
    z = float(ndtr(b) - ndtr(a)) if a < 0 else float(ndtr(-a) - ndtr(-b))
    one_sided_ok = math.isinf(b) and a < 30  # textbook error ~ a^4 * 1e-16 <= 1e-10 here
    if z < 1e-6 and not one_sided_ok:  # textbook formulas lose accuracy: fall back to mpmath
        if not allow_mp:
            raise OutOfRange("too far in the tail for the fast oracle")
        if math.isinf(hi):
            v, omw = mp_win((mean - lo) / sd, dps=50)
            return mean + sd * float(v), var * float(omw)
        c = (lo + hi) / 2
        m_, v_ = mp_draw((mean - c) / sd, (hi - lo) / (2 * sd), dps=40)
        return mean + sd * float(m_), var * float(v_)
    m = (pa - pb) / z
    ta = 0.0 if math.isinf(a) else a * pa
    tb = 0.0 if math.isinf(b) else b * pb
    v = 1.0 + (ta - tb) / z - m * m
    return mean + sd * m, var * v


def _flatten(teams, partial_play):
    mus = np.array([r.mu for t in teams for r in t], dtype=float)
    sig2 = np.array([r.sigma**2 for t in teams for r in t], dtype=float)
    w = np.array([x for ws in partial_play for x in ws], dtype=float)
    owner = np.array([k for k, t in enumerate(teams) for _ in t])
    return mus, sig2, w, owner


def _full_play(teams, partial_play):
    return [[1.0] * len(t) for t in teams] if partial_play is None else partial_play


# ------------------------------------------------------- closed-form, 2 teams
def closed_form_two_teams(env, teams, ranks, partial_play=None, force_mp=False):
    """Exact one-truncation update for two teams (winner first by rank).

    The textbook moments are good to ~1e-9 relative far in the tails;
    ``force_mp=True`` uses mpmath moments throughout (~1e-15).
    """
    partial_play = _full_play(teams, partial_play)
    order = sorted(range(2), key=lambda i: ranks[i])
    A, B = order
    drawn = ranks[0] == ranks[1]
    mus, sig2, w, owner = _flatten(teams, partial_play)
    s2 = sig2 + env.tau**2
    sign = np.where(owner == A, 1.0, -1.0)
    c2 = float(np.sum(w**2 * (s2 + env.beta**2)))
    mean_d = float(np.sum(sign * w * mus))
    margin = env.draw_margin(len(teams[0]) + len(teams[1]))
    lo, hi = (-margin, margin) if drawn else (margin, math.inf)
    m_new, v_new = _trunc_moments(mean_d, c2, lo, hi, force_mp=force_mp)
    gain = sign * w * s2 / c2
    new_mu = mus + gain * (m_new - mean_d)
    new_var = s2 - (w * s2) ** 2 / c2 * (1 - v_new / c2)
    out, i = [], 0
    for t in teams:
        out.append([(new_mu[i + j], math.sqrt(new_var[i + j])) for j in range(len(t))])
        i += len(t)
    return out


# ------------------------------------------------------------- dense EP
def dense_ep(env, teams, ranks, partial_play=None, tol=1e-11, max_sweeps=3000, allow_mp=False):
    """EP on the joint Gaussian of consecutive team differences.

    Same model as the factor graph but a completely different algorithm
    (matrix form, rank-one site updates). Both must reach the same fixed point.
    """
    partial_play = _full_play(teams, partial_play)
    n = len(teams)
    order = sorted(range(n), key=lambda i: ranks[i])
    pos = {team: p for p, team in enumerate(order)}
    mus, sig2, w, owner = _flatten(teams, partial_play)
    P = len(mus)
    Wm = np.zeros((n, P))
    for i in range(P):
        Wm[pos[owner[i]], i] = w[i]
    D = np.zeros((n - 1, n))
    for k in range(n - 1):
        D[k, k], D[k, k + 1] = 1.0, -1.0
    A = D @ Wm
    S = np.diag(sig2 + env.tau**2)
    m0 = A @ mus
    C = A @ S @ A.T + env.beta**2 * (A @ A.T)
    margins, drawn = [], []
    for k in range(n - 1):
        ta, tb = order[k], order[k + 1]
        margins.append(env.draw_margin(len(teams[ta]) + len(teams[tb])))
        drawn.append(ranks[ta] == ranks[tb])
    site_tau = np.zeros(n - 1)
    site_nu = np.zeros(n - 1)
    Cinv = np.linalg.inv(C)
    for _ in range(max_sweeps):
        Sd = np.linalg.inv(Cinv + np.diag(site_tau))
        md = Sd @ (Cinv @ m0 + site_nu)
        change = 0.0
        for k in range(n - 1):
            V, m = Sd[k, k], md[k]
            cav_var = 1.0 / (1.0 / V - site_tau[k])
            cav_mean = cav_var * (m / V - site_nu[k])
            lo, hi = (-margins[k], margins[k]) if drawn[k] else (margins[k], math.inf)
            new_mean, new_var = _trunc_moments(cav_mean, cav_var, lo, hi, allow_mp=allow_mp)
            nt = 1.0 / new_var - 1.0 / cav_var
            nn = new_mean / new_var - cav_mean / cav_var
            dt, dn = nt - site_tau[k], nn - site_nu[k]
            change = max(change, abs(dt), abs(dn))
            site_tau[k], site_nu[k] = nt, nn
            s = Sd[:, k].copy()
            denom = 1.0 + dt * Sd[k, k]
            md = md + s * (dn - dt * md[k]) / denom
            Sd = Sd - np.outer(s, s) * (dt / denom)
        if change < tol:
            break
    Sd = np.linalg.inv(Cinv + np.diag(site_tau))
    md = Sd @ (Cinv @ m0 + site_nu)
    K = S @ A.T @ Cinv
    mu_s = mus + K @ (md - m0)
    var_s = np.diag(S - K @ (C - Sd) @ K.T)
    out, i = [], 0
    for t in teams:
        out.append([(mu_s[i + j], math.sqrt(var_s[i + j])) for j in range(len(t))])
        i += len(t)
    return out


# ------------------------------------------------------------ Monte Carlo
def monte_carlo(env, teams, ranks, partial_play=None, n=2_000_000, seed=0):
    """Exact posterior moments by rejection sampling."""
    partial_play = _full_play(teams, partial_play)
    rng = np.random.default_rng(seed)
    mus, sig2, w, owner = _flatten(teams, partial_play)
    skills = rng.normal(mus, np.sqrt(sig2 + env.tau**2), size=(n, len(mus)))
    perf = (skills + rng.normal(0, env.beta, size=skills.shape)) * w
    team_perf = [perf[:, owner == k].sum(axis=1) for k in range(len(teams))]
    order = sorted(range(len(teams)), key=lambda i: ranks[i])
    keep = np.ones(n, dtype=bool)
    for a, b in zip(order, order[1:]):
        d = team_perf[a] - team_perf[b]
        eps = env.draw_margin(len(teams[a]) + len(teams[b]))
        keep &= (np.abs(d) <= eps) if ranks[a] == ranks[b] else (d > eps)
    s = skills[keep]
    return s.mean(axis=0), s.std(axis=0), int(keep.sum())
