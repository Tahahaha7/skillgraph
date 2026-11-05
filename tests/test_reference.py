"""Published reference values (the `trueskill` package / Moserware implementations)."""

from cases import max_err
from skillgraph import TS1, Rating


def check(out, expected, tol=1e-3):
    for team, exp_team in zip(out, expected):
        for r, (mu, sigma) in zip(team, exp_team):
            assert abs(r.mu - mu) < tol and abs(r.sigma - sigma) < tol, (r, mu, sigma)


def test_1vs1_win():
    check(TS1().rate([[Rating()], [Rating()]]), [[(29.396, 7.171)], [(20.604, 7.171)]])


def test_1vs1_draw():
    check(TS1().rate([[Rating()], [Rating()]], [0, 0]), [[(25.000, 6.458)], [(25.000, 6.458)]])


def test_1vs2():
    check(TS1().rate([[Rating()], [Rating(), Rating()]]), [[(33.731, 7.317)], [(16.269, 7.317), (16.269, 7.317)]])


def test_free_for_all_3():
    check(TS1().rate([[Rating()], [Rating()], [Rating()]]), [[(31.675, 6.656)], [(25.000, 6.208)], [(18.325, 6.656)]])


def test_quality_1vs1():
    assert abs(TS1().quality([[Rating()], [Rating()]]) - 0.447) < 1e-3


def test_rate_1vs1_matches_rate():
    env = TS1()
    a, b = Rating(27, 5), Rating(22, 7)
    assert max_err([list(env.rate_1vs1(a, b))], [[r[0] for r in env.rate([[a], [b]])]]) == 0
    assert max_err([list(env.rate_1vs1(a, b, drawn=True))], [[r[0] for r in env.rate([[a], [b]], [0, 0])]]) == 0
