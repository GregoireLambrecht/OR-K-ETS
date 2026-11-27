"""Job 3 (the FP run) and job 4 (the LP) must be the same problem (RUNS_SETTING.md §4-5,
benchmark_lp.tex Assumption 1). CPU, plain asserts. Run with job 3's environment
(run_jobs.sh env_job3), e.g. through MODE=check of run_jobs.sh.

Checks, at scenarios across job 3's box: the good-price path, the private technologies
and capacity, the allocation per unit of mass (= the LP's cap), the penalty (the fixed
fine kappa1 (E - theta)^+), no noise, floor 0 < ceiling, Assumption 1(v) on C^f, and the
LP's optimal production replayed through the environment's own reward = the LP value.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import linprog

from envs.base_params import *
from envs.params_v2 import A_CEIL_V2
from majmin_eval_common import make_env

assert LAYOUT == 'v5' and NO_MAJOR and DETERMINISTIC and PENALTY_MODE == 'fixed', 'run with job 3 environment'
T = 100
env = make_env(T)
cs = [C_MIN, 0.645, C_MAX]
G = [(c, 0.0, FLOOR_MIN, mm, k1) for c in cs for mm in (MM_MIN, 0.25, MM_MAX) for k1 in (KAPPA_MIN, KAPPA_MAX)]
_, ef, eg, cf, cg, capf, capxi, _, tech = PRIVATE_GENERATOR
e, cost = np.array([ef, eg]), np.array([cf, cg])
P_lp = np.asarray(generate_prices_ou(jax.random.PRNGKey(0), T), np.float64)        # lp_job4.py's path


def lp_profit(c, k1):
    obj = np.concatenate([-(P_lp[:T, None] - cost[None, :]).ravel(), [k1]])
    A = [np.concatenate([np.tile(e, T), [-1.0]])]; b = [c * T * capxi * e.max()]
    for t in range(T):
        r = np.zeros(2 * T + 1); r[2 * t:2 * t + 2] = 1.0; A.append(r); b.append(capxi)
    res = linprog(obj, A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=[(0, None)] * (2 * T + 1), method='highs')
    assert res.status == 0
    return -res.fun, res.x[:2 * T].reshape(T, 2)


def test_1_prices_technologies_noise_corridor():
    for k in range(3):
        P = np.asarray(env.generate_P(jax.random.PRNGKey(k), T))
        assert np.allclose(P, P_lp, atol=1e-4), 'good-price path differs from the LP'
        assert float(jnp.abs(env.generate_eps0(jax.random.PRNGKey(k), T)).max()) == 0.0
        for j in (0, 2):
            assert float(jnp.abs(env.generate_idiosyncratic_noise(jax.random.PRNGKey(k), env.agent_params[j, 7], T)).max()) == 0.0
    ap = np.asarray(env.agent_params)
    assert np.allclose(ap[0, 1:5], [ef, eg, cf, cg]) and ap[0, 6] == capxi and ap[0, 8] < 0, ap[0]
    assert ap[2, 6] == 0.0 and ap[2, 7] == 0.0, 'market makers must not produce or emit'
    assert FLOOR_MIN == FLOOR_MAX == 0.0 and A_CEIL_V2 > KAPPA_MAX, (FLOOR_MIN, FLOOR_MAX, A_CEIL_V2)


def test_2_allocation_is_the_lp_cap():
    for g in G:
        eb = env.bind(jnp.asarray(g, jnp.float32))
        th = float(eb.type_initial_state(0)[1])
        cap = g[0] * T * capxi * e.max()
        assert abs(th - cap) <= 1e-4 * cap, (g, th, cap)
        assert float(eb.type_initial_state(2)[1]) == 0.0
        assert float(eb.major_initial_state()[1]) == 0.0 and float(eb.context_weights()[1]) == 0.0


def test_3_penalty_is_the_fixed_fine():
    idx = env.dic_indexes
    for g in G[:4]:
        eb = env.bind(jnp.asarray(g, jnp.float32))
        x = eb.type_initial_state(0)
        for th, xi in ((100.0, (40.0, 10.0)), (5000.0, (10.0, 0.0)), (0.0, (0.0, 0.0))):
            x2 = x.at[idx['theta']].set(th).at[idx['xi']].set(jnp.array(xi)).at[idx['A_avg']].set(123.0)
            E = xi[0] * ef + xi[1] * eg
            want = -g[4] * max(E - th, 0.0)
            assert abs(float(eb.terminal_reward(x2)) - want) <= 1e-3 * max(1.0, abs(want)), (g, th, xi)


def test_4_assumption_v_trading_limits():
    """C^f > wbar * Cbar with wbar = 1/T (no exogenous supply): T C^f > max(theta, Theta + T C^xi e_max)."""
    for g in G:
        eb = env.bind(jnp.asarray(g, jnp.float32))
        w = np.asarray(eb.context_weights())
        th = np.array([float(eb.type_initial_state(j)[1]) for j in range(3)])
        Theta = float((w * th).sum())
        for j in (0, 2):
            cbar = max(th[j], Theta + T * float(env.agent_params[j, 6]) * e.max())
            assert T * float(env.agent_params[j, 5]) > cbar, (g, j, T * float(env.agent_params[j, 5]), cbar)


def test_5_lp_production_replayed_in_the_environment():
    """The LP's optimal production, no trade (holdings = cap = emissions): the environment's
    reward of a private producer equals the LP value per unit of mass."""
    idx = env.dic_indexes
    for g in [G[2], G[8], G[14]]:
        val, q = lp_profit(g[0], g[4])
        eb = env.bind(jnp.asarray(g, jnp.float32))
        P = eb.generate_P(jax.random.PRNGKey(0), T)
        x = eb.type_initial_state(0)
        tot = 0.0
        for t in range(T):
            xi = float(q[t].sum())
            a = jnp.array([0.0, xi, float(q[t, 0]) / xi if xi > 0 else 0.0], jnp.float32)
            tot += float(eb.running_reward(x, a, jnp.float32(50.0)))
            x = eb.single_step_dynamics(x, a, 0.0, jnp.float32(50.0), P[t + 1])
        tot += float(eb.terminal_reward(x))
        assert abs(tot - val) <= 1e-4 * abs(val), (g, tot, val)


if __name__ == '__main__':
    for nm, fn in sorted(globals().items()):
        if nm.startswith('test_'):
            fn()
            print(f"{nm}: PASSED", flush=True)
