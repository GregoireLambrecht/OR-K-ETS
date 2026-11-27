"""Tests of the v5 scenario layout (RUNS_SETTING.md §1-2, §4). CPU, plain asserts.

    JAX_PLATFORMS=cpu KETS_LAYOUT=v5 python3 tests/test_layout_v5.py
    JAX_PLATFORMS=cpu KETS_LAYOUT=v5 KETS_NO_MAJOR=1 python3 tests/test_layout_v5.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp

from envs.base_params import *
from envs.environment_majmin import MajorMinorEnv, MINORS, MAJOR
from utils import norm_gamma, CTX_REF

assert LAYOUT == 'v5', 'run with KETS_LAYOUT=v5'
T = 100
env = MajorMinorEnv(kappa=3.0, T=T, generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
                    market_impact_func=market_impact_base, generate_eps0_func=supply_noise_base,
                    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
                    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=20).set_T(T)
SCEN = [CTX_REF, (0.3, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 15.0, 0.5, 6.0), (0.6, 0.5, 3.0, 0.25, 2.0)]
ef = max(PRIVATE_GENERATOR[1], PRIVATE_GENERATOR[2])
efb = max(BIG_PUBLIC_GENERATOR[1], BIG_PUBLIC_GENERATOR[2])


def test_1_masses():
    for g in SCEN:
        w = np.asarray(env.bind(jnp.asarray(g, jnp.float32)).context_weights())
        m0 = 0.0 if NO_MAJOR else MAJOR_MASS_V5
        assert np.allclose(w, [0.8 - g[3], m0, g[3]], atol=1e-6), (g, w)


def test_2_allocation():
    """sum_l n_l theta_l = Theta_tot = c T (C0 m0 e0 + 100 n_priv e_priv); major s Theta_tot; mm 0."""
    for g in SCEN:
        eb = env.bind(jnp.asarray(g, jnp.float32))
        w = np.asarray(eb.context_weights())
        th = np.array([float(eb.type_initial_state(j)[1]) for j in range(3)])
        tot = g[0] * T * (w[1] * BIG_PUBLIC_GENERATOR[6] * efb + w[0] * PRIVATE_GENERATOR[6] * ef)
        assert abs(float(eb.total_cap_v5()) - tot) <= 1e-4 * tot, (g, float(eb.total_cap_v5()), tot)
        assert abs((w * th).sum() - tot) <= 1e-4 * tot, (g, (w * th).sum(), tot)
        assert th[2] == 0.0, th
        maj = float(eb.major_initial_state()[1])              # aggregate units: theta_major x m0
        want = 0.0 if NO_MAJOR else g[1] * tot
        assert abs(maj - want) <= 1e-4 * max(tot, 1.0), (g, maj, want)
        if NO_MAJOR:                                          # job 3: all of it to the private producers
            assert abs(th[0] - g[0] * T * PRIVATE_GENERATOR[6] * ef) <= 1e-3 * th[0], (g, th[0])


def test_3_market_makers_never_emit():
    assert MARKET_MAKER[6] == 0.0 and MARKET_MAKER[7] == 0.0, MARKET_MAKER
    x = jnp.zeros((5, T))
    eps = env.generate_idiosyncratic_noise(jax.random.PRNGKey(0), env.agent_params[2, 7], T)
    assert float(jnp.abs(eps).max()) == 0.0


def test_4_s_cap_and_reference():
    if NO_MAJOR:
        return
    assert abs(s_cap_v5(0.0) - 140 / (140 + 80)) < 1e-9 and abs(s_cap_v5(0.5) - 140 / (140 + 30)) < 1e-9
    assert abs(CTX_REF_V5[1] - s_cap_v5(0.4)) < 1e-12 and len(CTX_REF) == 5


def test_5_supply_fixed_at_reference():
    vals = [float(env.bind(jnp.asarray(g, jnp.float32)).sigma_S()) for g in SCEN]
    assert np.allclose(vals, vals[0], rtol=1e-6), vals


def test_6_context_sampling_and_norm():
    g = env.generate_context(jax.random.PRNGKey(0), 4096)
    lo = np.array([C_MIN, S_MIN, FLOOR_MIN, MM_MIN, KAPPA_MIN]); hi = np.array([C_MAX, S_MAX, FLOOR_MAX, MM_MAX, KAPPA_MAX])
    g = np.asarray(g)
    assert g.shape == (4096, 5) and np.all(g >= lo - 1e-6) and np.all(g <= hi + 1e-6)
    z = np.asarray(jax.vmap(norm_gamma)(jnp.asarray(g)))
    assert np.all(np.abs(z) <= 1 + 1e-5)


if __name__ == '__main__':
    for nm, fn in sorted(globals().items()):
        if nm.startswith('test_'):
            fn()
            print(f"{nm}: PASSED (NO_MAJOR={NO_MAJOR})", flush=True)
