"""Market-clearing tests (MARKET_CLEARING.md §3). CPU, plain asserts:

    JAX_PLATFORMS=cpu python3 tests/test_clearing.py      (or pytest)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import TypePolicyNN, stack_policies
from utils import (make_price_fn_batched, make_br_stage_fn_r2, draw_common_noise,
                   draw_idio)

T = 10
CTX = (0.9, 3.0, 6.0, 0.4)


def make_env():
    env = ExogenousMarketEnvJAX(
        kappa=3.0, T=T, generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
        market_impact_func=market_impact_base, generate_eps0_func=supply_noise_base,
        generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
        A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=20)
    return env.set_context(jnp.asarray(CTX, jnp.float32))


def const_outs(env, Xs, alpha, beta):
    return [jnp.broadcast_to(jnp.array([alpha, beta, 0.5, 0.5], jnp.float32),
                             X.shape[:-1] + (4,)) for X in Xs]


def cap_total(env):
    w = np.asarray(env.context_weights())
    return float(np.sum(w * np.asarray(env.agent_params[:, 5])))


def test_single_schedule():
    env = make_env()
    Xs = env.population_initial_states(2)
    w = env.context_weights()
    alpha, beta = 0.5, 1.0
    D, _ = env.excess_demand(Xs, const_outs(env, Xs, alpha, beta), 0.0, w)
    A_pre = float(Xs[0][0, 0, 5])
    A_t, _, _ = env.clear_price(D, A_pre)
    R = A_pre + float(env.A_scale) * alpha / beta
    assert abs(float(A_t) - R) < 1e-4, (float(A_t), R)


def test_floor():
    env = make_env()
    Xs = env.population_initial_states(2)
    w = env.context_weights()
    D, _ = env.excess_demand(Xs, const_outs(env, Xs, -50.0, 1.0), 0.0, w)
    A_t, _, _ = env.clear_price(D, float(Xs[0][0, 0, 5]))
    assert abs(float(A_t) - float(env.Afloor)) < 1e-6
    assert -float(D(A_t)) > 0                     # government buys


def test_ceiling():
    env = make_env()
    Xs = env.population_initial_states(2)
    w = env.context_weights()
    D, _ = env.excess_demand(Xs, const_outs(env, Xs, 50.0, 1.0), 0.0, w)
    A_t, _, _ = env.clear_price(D, float(Xs[0][0, 0, 5]))
    assert abs(float(A_t) - A_CEIL) < 1e-4
    assert -float(D(A_t)) < 0                     # government sells


def test_flat_region():
    env = make_env()
    Xs = env.population_initial_states(2)
    w = env.context_weights()
    outs = const_outs(env, Xs, 50.0, 1.0)          # saturated at the buy cap
    D0, _ = env.excess_demand(Xs, outs, 0.0, w)
    S = D0(50.0)                                   # supply = total demand
    D, _ = env.excess_demand(Xs, outs, S, w)
    A_pre = float(Xs[0][0, 0, 5])
    A_t, am, ap = env.clear_price(D, A_pre)
    assert float(am) < float(ap)
    assert abs(float(A_t) - A_pre) < 1e-6


def test_monotone():
    env = make_env()
    Xs = env.population_initial_states(3)
    w = env.context_weights()
    k = jax.random.PRNGKey(0)
    outs = []
    for X in Xs:
        k, k1, k2 = jax.random.split(k, 3)
        sh = X.shape[:-1]
        outs.append(jnp.stack([jax.random.normal(k1, sh) * 2,
                               BETA_MIN + jax.random.exponential(k2, sh),
                               jnp.full(sh, .5), jnp.full(sh, .5)], -1))
    D, _ = env.excess_demand(Xs, outs, 3.0, w)
    grid = jnp.linspace(float(env.Afloor), A_CEIL, 1000)
    d = np.asarray(jax.vmap(D)(grid))
    assert np.all(np.diff(d) <= 1e-4), np.diff(d).max()


def _random_mixture(K, seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), K * N_TYPES)
    hist = [tuple(TypePolicyNN(key=ks[k * N_TYPES + j]) for j in range(N_TYPES))
            for k in range(K)]
    stacked = tuple(stack_policies([h[j] for h in hist])[0] for j in range(N_TYPES))
    statics = tuple(stack_policies([hist[0][j]])[1] for j in range(N_TYPES))
    return hist, stacked, statics


def test_conservation_and_shapes():
    env = make_env()
    _, stacked, statics = _random_mixture(2)
    P, eps0 = draw_common_noise(env, T, 1, jax.random.PRNGKey(1))
    idio = draw_idio(env, T, env.type_counts, jax.random.PRNGKey(2))
    A, diag = env.ensemble_price_path(stacked, statics, P[0], eps0[0], idio,
                                      return_diag=True)
    assert A.shape == (T + 1,)
    assert abs(float(A[0]) - A_REF) < 1e-6
    cap = cap_total(env)
    assert float(jnp.max(jnp.abs(diag['cons_err']))) / cap < 1e-5
    assert float(jnp.max(diag['resid'])) / cap < 1e-5
    A2 = env.ensemble_price_path(stacked, statics, P[0], eps0[0], idio)
    assert np.allclose(np.asarray(A), np.asarray(A2))


def test_jit_vmap():
    env = make_env()
    hist, stacked, statics = _random_mixture(2)
    P, eps0 = draw_common_noise(env, T, 3, jax.random.PRNGKey(1))
    idio = draw_idio(env, T, env.type_counts, jax.random.PRNGKey(2))
    A = make_price_fn_batched(env, statics)(stacked, P, eps0, idio)
    assert A.shape == (3, T + 1)
    _, static = eqx.partition(hist[0][0], eqx.is_array)
    opt = optax.adam(1e-3)
    stage = make_br_stage_fn_r2(env, statics, opt, 2, 4, static)
    models = [TypePolicyNN(key=jax.random.PRNGKey(10 + j)) for j in range(N_TYPES)]
    parts = [eqx.partition(m, eqx.is_array)[0] for m in models]
    p_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *parts)
    p_f, _, losses = stage(p_stack, opt.init(p_stack), jax.random.PRNGKey(3), stacked)
    assert np.all(np.isfinite(np.asarray(losses)))


def test_deviator_consistency():
    """A deviator holding the parameters of a one-component mixture, facing the
    price that mixture clears and the same idiosyncratic draws, earns exactly
    what the population's type-j particles earn."""
    env = make_env()
    hist, stacked, statics = _random_mixture(1, seed=5)
    P, eps0 = draw_common_noise(env, T, 1, jax.random.PRNGKey(1))
    idio = draw_idio(env, T, env.type_counts, jax.random.PRNGKey(2))
    diagn = env.ensemble_diagnostics(stacked, statics, P[0], eps0[0], idio)
    for j in range(N_TYPES):
        _, R = env.rollout_agent_batch(hist[0][j], j, diagn['A_traj'], P[0], idio[j])
        a, b = float(jnp.mean(R)), float(diagn['per_type_reward'][j])
        assert abs(a - b) <= 1e-4 * max(1.0, abs(b)), (j, a, b)


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"PASS {name}", flush=True)
            except Exception as e:                       # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    print(f"{fails} failure(s)")
    sys.exit(1 if fails else 0)
