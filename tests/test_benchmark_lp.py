"""Tests of benchmark_lp.py (CPU): the planner, the exact best response, and the
units check against the environment.

    JAX_PLATFORMS=cpu python3 tests/test_benchmark_lp.py
"""
import os
import sys

os.environ.update(KETS_DETERMINISTIC='1', KETS_PENALTY='fixed', KETS_NO_MAJOR='1', KETS_CAP_F_MULT='2.5',
                  KETS_FLOOR_MIN='0', KETS_FLOOR_MAX='0', KETS_ALLOC_MIN='0.5', KETS_ALLOC_MAX='1.2',
                  KETS_KAPPA_MIN='10', KETS_KAPPA_MAX='80', KETS_MM_MIN='0', KETS_MM_MAX='0.5',
                  KETS_CTX_REF='0.805 60 0 0.2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax.numpy as jnp
from benchmark_lp import (market_data, planner, planner_lp, best_response_lp, path_reward,
                          value_star, p_days, merit_order)
from majmin_eval_common import make_env

T = 100
env = make_env(T)
# init_alloc * T full fossil days fit under the cap: an integer number makes Phi flat
# between two shutdown prices (I* an interval, Corollary 2); a half-integer gives a unique a*
SCEN = [(0.805, 60.0, 0.0, 0.2),   # interior, unique
        (0.8, 60.0, 0.0, 0.2),     # interior, I* an interval
        (0.805, 20.0, 0.0, 0.2),   # the fine binds
        (1.1, 60.0, 0.0, 0.0),     # the cap does not bind
        (0.505, 80.0, 0.0, 0.5),
        (0.955, 52.0, 0.0, 0.1)]


def test_1_planner_matches_highs():
    for g in SCEN:
        m = market_data(env, g)
        pl, lp = planner(m), planner_lp(m)
        assert abs(pl['phi'] - lp['value']) <= 1e-6 * max(1.0, abs(lp['value'])), (g, pl['phi'], lp['value'])
        lo, hi = pl['I_star']
        assert lo - 1e-6 <= lp['a_dual'] <= hi + 1e-6, (g, pl['I_star'], lp['a_dual'])
        print(f"  {g}: a* = {pl['a_star']:.4f} ({pl['regime']}, unique {pl['unique']}), "
              f"HiGHS dual {lp['a_dual']:.4f}, Phi {pl['phi']:.6g} = LP {lp['value']:.6g}")


def test_2_regimes_and_emissions():
    """Theorem 1 (ii)(d) at the merit order: E = Theta interior, E <= Theta at 0, E >= Theta at kappa."""
    for g in SCEN:
        m = market_data(env, g)
        pl = planner(m)
        a = pl['a_star']
        _, e_right = merit_order(m, a + 1e-6)
        _, e_left = merit_order(m, max(a - 1e-6, 0.0))
        Tc = m['Theta_c']
        tol = 1e-5 * Tc                         # theta comes from the float32 environment
        if pl['regime'] == 'interior':
            assert e_right - tol <= Tc <= e_left + tol, (g, e_left, Tc, e_right)
            days = g[0] * T                     # the cap allows init_alloc * T full fossil days
            assert pl['unique'] == (abs(days - round(days)) > 1e-3), (g, pl['I_star'])
        elif pl['regime'] == 'cap_slack':
            assert e_right <= Tc + tol, (g, e_right, Tc)
        else:
            assert e_left >= Tc - tol, (g, e_left, Tc)


def test_3_best_response_at_a_star_is_V_star():
    """Lemma 3: at A_t = a, rho = 1, the best response value is V*_j(a)."""
    for g in SCEN:
        m = market_data(env, g)
        a = planner(m)['a_star']
        one = np.ones(T)
        for d in m['types']:
            v = best_response_lp(m, d, a * one, one, one)
            vs = value_star(d, p_days(m), a, T)
            assert abs(v - vs) <= 1e-6 * max(1.0, abs(vs)), (g, d['j'], v, vs)


def test_4_lp_path_reward_equals_environment_reward():
    """The LP's optimal path, replayed with the environment's reward and dynamics, gives
    the LP value (units, day indexing, penalty): at a* and along a random market."""
    rng = np.random.default_rng(0)
    for g in SCEN[:3]:
        m = market_data(env, g)
        a = planner(m)['a_star']
        paths = [(a * np.ones(T), np.ones(T), np.ones(T)),
                 (a + 5 * rng.standard_normal(T).cumsum() / np.sqrt(T), rng.uniform(0.3, 1, T), rng.uniform(0.3, 1, T))]
        for A, rs, rb in paths:
            A = np.maximum(A, 0.0)
            for d in m['types']:
                v, path = best_response_lp(m, d, A, rs, rb, return_path=True)
                A_traj = jnp.asarray(np.concatenate([[0.0], A]), jnp.float32)
                r = path_reward(env, g, d['j'], path['q'], path['fh'], A_traj)
                # float32 replay: the error scales with the gross cash flows, not with the value
                # (a financial institution trades ~ A C^f a day for a value near 0)
                gross = float(np.sum(np.abs(A * path['fh'])) + np.sum(np.abs(path['q'])) * 100.0)
                assert abs(r - v) <= 1e-6 * gross + 1e-4 * max(1.0, abs(v)), (g, d['j'], r, v, gross)


if __name__ == '__main__':
    for nm, fn in sorted(globals().items()):
        if nm.startswith('test_'):
            fn()
            print(f"{nm}: PASSED", flush=True)
