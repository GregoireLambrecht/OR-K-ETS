"""Market clearing v2 tests (MARKET_CLEARING.md rev. 7be619c, §3). CPU, plain asserts.

    JAX_PLATFORMS=cpu KETS_FLOOR_MODE=block   python3 tests/test_clearing_v2.py
    JAX_PLATFORMS=cpu KETS_FLOOR_MODE=support python3 tests/test_clearing_v2.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp

from envs.base_params import *
from envs.params_v2 import A_REF, A_CEIL_V2, SIGMA_S, FLOOR_MODE, K_MAX
from envs.environment_v2 import MarketClearingEnvV2, stratified_components, N_TYPES
from envs.models_v2 import TypePolicyNNV2, make_buffer, buffer_set

T = 10
env = MarketClearingEnvV2(
    kappa=3.0, T=T, generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
    market_impact_func=market_impact_base, generate_eps0_func=supply_noise_base,
    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=20)
env = env.set_context(jnp.array([0.9, 3.0, 6.0, 0.4])).set_T(T)
w = env.context_weights()
cap = float(jnp.sum(w * env.agent_params[:, 5]))
FL = float(env.Afloor)
SQ = float(SIGMA_S * jnp.sqrt(env.dt))      # S_t = SQ * eps_t


def schedule_orders(alpha, beta, A_pre=A_REF, cf=None, lo=None):
    """Orders of one representative block: (buy, sell) at price a."""
    cf = cap if cf is None else cf
    lo = -cf if lo is None else lo

    def fn(a):
        f = jnp.clip(cf * jnp.tanh(alpha - beta * (a - A_pre) / env.A_scale), lo, cf)
        return jnp.maximum(f, 0.0), jnp.maximum(-f, 0.0)
    return fn


def test_1_single_schedule():
    for alpha in (0.5, 1.5, 3.0):
        A_pre = 20.0
        A_t, rs, rb, G, cd = env.clear(schedule_orders(alpha, 1.0, A_pre), 0.0, A_pre)
        R = A_pre + float(env.A_scale) * alpha / 1.0
        assert abs(float(A_t) - R) < 1e-4 * R, (alpha, float(A_t), R)
        assert float(rs) == 1.0 and float(rb) == 1.0 and float(G) == 0.0


def test_2_monotone():
    key = jax.random.PRNGKey(0)
    for i in range(20):
        k1, k2, key = jax.random.split(key, 3)
        al = jax.random.normal(k1, (64,)) * 3
        be = 0.1 + jax.nn.softplus(jax.random.normal(k2, (64,)))
        f = lambda a: cap * jnp.tanh(al - be * (a - A_REF) / env.A_scale)
        grid = jnp.linspace(FL, A_CEIL_V2, 1000)
        D = jax.vmap(lambda a: jnp.mean(f(a)))(grid)
        assert (np.diff(np.asarray(D)) <= 1e-4).all()


def test_3_flat_region():
    # net orders exactly zero on [50, 80] (S = 0): buy below 50, sell above 80
    fn = lambda a: (jnp.maximum(50.0 - a, 0.0), jnp.maximum(a - 80.0, 0.0))
    for A_pre, expect in ((60.0, 60.0), (15.0, 50.0), (120.0, 80.0)):
        A_t, rs, rb, G, cd = env.clear(fn, 0.0, A_pre)
        assert abs(float(cd['a_minus']) - 50.0) < 1e-3 and abs(float(cd['a_plus']) - 80.0) < 1e-3
        assert abs(float(A_t) - expect) < 1e-3, (A_pre, float(A_t), expect)
        assert float(rs) == 1.0 and float(rb) == 1.0 and float(G) == 0.0


def test_4_5_floor_binds():
    fn = schedule_orders(-3.0, 1.0)            # everyone sells hard
    A_t, rs, rb, G, cd = env.clear(fn, 0.0, A_REF)
    assert abs(float(A_t) - FL) < 1e-6 and bool(cd['on_floor'])
    B, O = float(cd['B']), float(cd['O'])
    assert O > B and abs(float(G) - (O - B)) < 1e-4 * O and float(G) > 0
    assert float(rb) == 1.0
    if FLOOR_MODE == 'block':
        assert float(rs) < 1.0 and abs(float(rs) - B / O) < 1e-6
        assert abs(B - float(rs) * O) < 1e-4 * O     # executed buys == executed sells
    else:
        assert float(rs) == 1.0


def test_6_interior():
    fn = schedule_orders(1.0, 1.0, A_pre=30.0)
    A_t, rs, rb, G, cd = env.clear(fn, 0.3, 30.0)
    assert not bool(cd['on_floor']) and not bool(cd['on_ceil'])
    assert float(rs) == 1.0 and float(rb) == 1.0 and float(G) == 0.0
    assert abs(float(cd['D_At'])) < 1e-4 * cap, float(cd['D_At'])


def test_6b_ceiling_binds():
    fn = schedule_orders(30.0, 0.1, A_pre=150.0)  # everyone buys at the cap up to A_CEIL
    A_t, rs, rb, G, cd = env.clear(fn, 0.0, 150.0)
    assert abs(float(A_t) - A_CEIL_V2) < 1e-4 and bool(cd['on_ceil'])
    assert float(G) < 0 and float(rs) == 1.0
    if FLOOR_MODE == 'block':
        assert float(rb) < 1.0 and abs(float(rb) - float(cd['O']) / float(cd['B'])) < 1e-6
    else:
        assert float(rb) == 1.0


def _mixture(k, seed=0):
    pols = [tuple(TypePolicyNNV2(key=jax.random.PRNGKey(100 * seed + 10 * i + j))
                  for j in range(N_TYPES)) for i in range(k)]
    bufs, statics = [], []
    for j in range(N_TYPES):
        b, st = make_buffer(pols[0][j], K_MAX)
        for i in range(k):
            b = buffer_set(b, i, pols[i][j])
        bufs.append(b); statics.append(st)
    return tuple(bufs), tuple(statics)


def test_7_8_path_conservation_jit():
    n, k = 12, 3
    bufs, statics = _mixture(k)
    kP, kE, kI, kC = jax.random.split(jax.random.PRNGKey(1), 4)
    S = 4
    P = jax.vmap(env.generate_P, in_axes=(0, None))(jax.random.split(kP, S), T)
    eps0 = jax.vmap(env.generate_eps0, in_axes=(0, None))(jax.random.split(kE, S), T) * 3
    idio = tuple(jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
        jax.random.split(jax.random.fold_in(kI, j), n), env.agent_params[j, 7], T)
        for j in range(N_TYPES))

    @jax.jit
    def fn(bufs, k, ck, P, eps0):
        return jax.vmap(lambda c, p, e: env.ensemble_price_path_v2(
            bufs, statics, k, c, p, e, idio, return_diag=True))(ck, P, eps0)
    A, rho, diag = fn(bufs, jnp.int32(k), jax.random.split(kC, S), P, eps0)
    assert A.shape == (S, T + 1) and rho.shape == (S, T + 1, 2)
    assert np.allclose(np.asarray(A[:, 0]), A_REF) and np.allclose(np.asarray(rho[:, 0]), 1.0)
    assert np.isfinite(np.asarray(A)).all() and np.isfinite(np.asarray(rho)).all()
    a = np.asarray(A[:, 1:])
    assert (a >= FL - 1e-5).all() and (a <= A_CEIL_V2 + 1e-5).all()
    ce = float(np.abs(np.asarray(diag['cons_err'])).max())
    assert ce < 1e-4 * cap, (ce, cap)
    rs = float(np.asarray(diag['resid']).max())
    assert rs < 1e-4 * cap, (rs, cap)
    fn(bufs, jnp.int32(2), jax.random.split(kC, S), P, eps0)   # other k, same compile
    pol = TypePolicyNNV2(key=jax.random.PRNGKey(7))
    Xf, R = env.rollout_agent_batch_v2(pol, 0, A[0], rho[0], P[0], idio[0][:5])
    assert R.shape == (5,) and np.isfinite(np.asarray(R)).all()


def test_10_stratification():
    for n, k in ((50, 30), (128, 7), (20, 20), (10, 30)):
        if n >= k:
            for s in range(5):
                c = np.asarray(stratified_components(jax.random.PRNGKey(s), k, n))
                cnt = np.bincount(c, minlength=k)
                assert cnt.min() >= n // k and cnt.max() <= -(-n // k), (n, k, cnt)
        freq = np.zeros(k)
        for s in range(3000):
            c = np.asarray(stratified_components(jax.random.PRNGKey(s), k, n))
            freq += np.bincount(c, minlength=k)
        freq /= freq.sum()
        assert np.abs(freq - 1 / k).max() < 0.25 / k, (n, k, freq)


def test_11_block_limit():
    k, N = 4, 5
    c = np.asarray(stratified_components(None, k, k * N))
    assert (c == np.repeat(np.arange(k), N)).all(), c


if __name__ == '__main__':
    tests = [v for kk, v in sorted(globals().items()) if kk.startswith('test_')]
    for t in tests:
        t()
        print(f"ok  {t.__name__}  [{FLOOR_MODE}]", flush=True)
    print(f"all {len(tests)} v2 tests passed  [{FLOOR_MODE}]")
