"""Major–minor clearing tests (MAJOR_MINOR_CLEARING.md §10). CPU, plain asserts.

    JAX_PLATFORMS=cpu KETS_FLOOR_MODE=block   python3 tests/test_majmin.py
    JAX_PLATFORMS=cpu KETS_FLOOR_MODE=support python3 tests/test_majmin.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)

from envs.base_params import *
from envs.params_v2 import A_CEIL_V2, FLOOR_MODE, K_MAX
from envs.environment_majmin import (MajorMinorEnv, clear_price, _bisect, _D, _order,
                                     MINORS, MAJOR)
from envs.models_majmin import minor_policy, major_policy, make_buffer, buffer_set
from utils import norm_gamma

T = 10
env = MajorMinorEnv(kappa=3.0, T=T, generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
                    market_impact_func=market_impact_base, generate_eps0_func=supply_noise_base,
                    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
                    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=20)
G = jnp.array([0.9, 3.0, 6.0, 0.4], jnp.float32)
env = env.bind(G).set_T(T)
SC = float(env.A_scale)
FL, CE = jnp.float32(float(env.Afloor)), jnp.float32(A_CEIL_V2)
w = env.context_weights()
WM = (w[0], w[2])
EPS_D = env.eps_d()


def minors(alpha, beta, apre, n=8, cap=50.0, seed=0):
    k = jax.random.PRNGKey(seed)
    al = alpha + 0.3 * jax.random.normal(k, (n,))
    one = jnp.ones(n)
    p = (al, beta * one, -cap * one, cap * one, cap * one, apre * one)
    return (p, p)


def major(alpha, beta, apre, cap=200.0):
    return (jnp.float32(alpha), jnp.float32(beta), jnp.float32(-cap), jnp.float32(cap),
            jnp.float32(cap), jnp.float32(apre))


def test_1_flat_major_matches_grid():
    apre = 40.0
    mins = minors(0.8, 1.0, apre)
    maj = major(0.5, 0.0, apre, cap=30.0)            # beta0 = 0: a quantity order (interior root)
    A = float(clear_price(maj, mins, WM, 0.0, apre, FL, CE, SC, EPS_D))
    grid = jnp.linspace(float(FL), float(CE), 200001)
    D = np.asarray(jax.vmap(lambda a: _D(a, maj, mins, WM, 0.0, SC))(grid))
    root = float(grid[np.argmin(np.abs(D))])
    assert abs(A - root) < 2e-3 * max(1.0, root), (A, root)


def test_2_ift_gradient_vs_fd():
    apre = 40.0
    mins = minors(0.8, 1.0, apre)

    def A_of(al0, be0):
        return clear_price(major(al0, be0, apre), mins, WM, jnp.float32(3.0), apre, FL, CE, SC, EPS_D)
    for al0, be0 in ((0.3, 0.8), (-0.4, 1.5)):
        ga, gb = jax.grad(A_of, argnums=(0, 1))(jnp.float32(al0), jnp.float32(be0))
        h = 1e-2
        fa = (A_of(al0 + h, be0) - A_of(al0 - h, be0)) / (2 * h)
        fb = (A_of(al0, be0 + h) - A_of(al0, be0 - h)) / (2 * h)
        assert abs(float(ga) - float(fa)) < 0.03 * abs(float(fa)) + 1e-2, (float(ga), float(fa))
        assert abs(float(gb) - float(fb)) < 0.03 * abs(float(fb)) + 1e-2, (float(gb), float(fb))
        assert float(ga) > 0                       # buying more raises the price


def test_3_rationing_gradient_at_floor():
    apre = float(FL) + 1.0
    mins = minors(-2.0, 1.0, apre)                  # minors sell hard: the floor binds

    def fhat0(al0):
        maj = major(al0, 1.0, apre)
        A = clear_price(maj, mins, WM, jnp.float32(0.0), apre, FL, CE, SC, EPS_D)
        f0 = _order(A, maj, SC)
        fm = [_order(A, p, SC) for p in mins]
        B = jnp.maximum(f0, 0) + sum(w_ * jnp.mean(jnp.maximum(f, 0)) for w_, f in zip(WM, fm))
        O = jnp.maximum(-f0, 0) + sum(w_ * jnp.mean(jnp.maximum(-f, 0)) for w_, f in zip(WM, fm))
        rho = B / O
        return jnp.where(f0 >= 0, f0, rho * f0), A
    _, A = fhat0(jnp.float32(-1.0))
    assert abs(float(A) - float(FL)) < 1e-5
    g = jax.grad(lambda a: fhat0(a)[0])(jnp.float32(-1.0))
    h = 1e-2
    fd = (fhat0(jnp.float32(-1.0 + h))[0] - fhat0(jnp.float32(-1.0 - h))[0]) / (2 * h)
    assert abs(float(g) - float(fd)) < 0.03 * abs(float(fd)) + 1e-3, (float(g), float(fd))


def _mix(k, n_types=3):
    pols = [[major_policy(jax.random.PRNGKey(10 * i + j)) if j == MAJOR
             else minor_policy(jax.random.PRNGKey(10 * i + j)) for j in range(n_types)] for i in range(k)]
    bufs, stats = [], []
    for j in range(n_types):
        b, st = make_buffer(pols[0][j], K_MAX)
        for i in range(k):
            b = buffer_set(b, i, pols[i][j])
        bufs.append(b); stats.append(st)
    return bufs, stats, pols


def test_4_conservation_and_shapes():
    n, k = 6, 3
    bufs, stats, pols = _mix(k)
    mb, ms = tuple(bufs[j] for j in MINORS), tuple(stats[j] for j in MINORS)
    kP, kE, kI, kC = jax.random.split(jax.random.PRNGKey(3), 4)
    P = env.generate_P(kP, T)
    eps0 = env.generate_eps0(kE, T) * 3
    imin = tuple(jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
        jax.random.split(jax.random.fold_in(kI, j), n), env.agent_params[j, 7], T) for j in MINORS)
    imaj = env.generate_idiosyncratic_noise(kI, env.agent_params[MAJOR, 7], T) * env.major_mass()
    maj = pols[1][MAJOR]
    A, rho, H, r0, ys, _ = env.simulate(lambda x, c: maj(x, c), mb, ms, jnp.int32(k), kC,
                                        norm_gamma(G), P, eps0, imin, imaj)
    assert A.shape == (T + 1,) and rho.shape == (T + 1, 2) and H.shape == (T, 3)
    assert np.isfinite(np.asarray(A)).all() and np.isfinite(float(r0))
    cap = float(env.cap_total())
    ce = float(np.abs(np.asarray(ys['cons_err'])).max())
    assert ce < 1e-4 * cap, (ce, cap)
    # gradient of the major's reward through the clearing (myopic), finite
    p, st = jax.tree_util.tree_map(lambda x: x, (maj, None))
    import equinox as eqx
    prm, stc = eqx.partition(maj, eqx.is_array)

    def r_of(q):
        net = eqx.combine(q, stc)
        return env.simulate(lambda x, c: net(x, c), mb, ms, jnp.int32(k), kC, norm_gamma(G),
                            P, eps0, imin, imaj, myopic=True, checkpoint=True)[3]
    g = jax.grad(r_of)(prm)
    leaves = jax.tree_util.tree_leaves(g)
    assert all(np.isfinite(np.asarray(l)).all() for l in leaves)
    assert sum(float(jnp.abs(l).sum()) for l in leaves) > 0


def test_5_minor_rollout():
    k = 2
    bufs, stats, pols = _mix(k)
    A = jnp.linspace(6.05, 60.0, T + 1)
    rho = jnp.ones((T + 1, 2))
    H = jnp.zeros((T, 3))
    P = env.generate_P(jax.random.PRNGKey(5), T)
    idio = jnp.zeros((4, T))
    R = env.rollout_minor(pols[0][0], 0, A, rho, H, P, idio, norm_gamma(G))
    assert R.shape == (4,) and np.isfinite(np.asarray(R)).all()


def test_6_major_rollout_reproduces_cleared_reward():
    """POLICY_SURFACES.md §4.1: the major's price-taking rollout of component u on the
    paths of a market cleared with component u returns that market's r0."""
    n, k = 6, 3
    bufs, stats, pols = _mix(k)
    mb, ms = tuple(bufs[j] for j in MINORS), tuple(stats[j] for j in MINORS)
    kP, kE, kI, kC = jax.random.split(jax.random.PRNGKey(11), 4)
    P = env.generate_P(kP, T)
    eps0 = env.generate_eps0(kE, T) * 3
    imin = tuple(jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
        jax.random.split(jax.random.fold_in(kI, j), n), env.agent_params[j, 7], T) for j in MINORS)
    imaj = env.generate_idiosyncratic_noise(jax.random.fold_in(kI, 99), env.agent_params[MAJOR, 7], T) \
        * env.major_mass()
    for u in range(k):
        maj = pols[u][MAJOR]
        A, rho, H, r0, ys, _ = env.simulate(lambda x, c: maj(x, c), mb, ms, jnp.int32(k), kC,
                                            norm_gamma(G), P, eps0, imin, imaj)
        R = env.rollout_major(maj, A, rho, H, P, imaj, norm_gamma(G))
        assert abs(float(R) - float(r0)) <= 1e-4 * max(1.0, abs(float(r0))), (u, float(R), float(r0))


if __name__ == '__main__':
    tests = [v for kk, v in sorted(globals().items()) if kk.startswith('test_')]
    for t in tests:
        t()
        print(f"ok  {t.__name__}  [{FLOOR_MODE}]", flush=True)
    print(f"all {len(tests)} major-minor tests passed  [{FLOOR_MODE}]")
