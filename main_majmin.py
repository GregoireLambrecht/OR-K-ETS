"""Major–minor market clearing (MAJOR_MINOR_CLEARING.md), fictitious play (utils_majmin).

    rung 2 (fixed scenario current_kets):
      python3 main_majmin.py --rung 2 [--fresh-br] --model-dir results_majmin/r2_T30_K30_n30
    rung 3 (scenario box, FiLM):
      KETS_FLOOR_MAX=30 python3 main_majmin.py --rung 3 --model-dir results_majmin/r3_T30_K30_n30
"""
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

p = argparse.ArgumentParser()
p.add_argument('--rung', type=int, choices=[2, 3], required=True)
p.add_argument('--model-dir', required=True)
p.add_argument('--nb-policy', type=int, default=30)
p.add_argument('--t-grid', type=int, default=30)
p.add_argument('--n-sim', type=int, default=30, help='simulated agents per MINOR type')
p.add_argument('--iters', type=int, default=8000, help='minors BR steps')
p.add_argument('--iters-major', type=int, default=4000, help='major BR steps')
p.add_argument('--batch', type=int, default=256)
p.add_argument('--batch-major', type=int, default=64)
p.add_argument('--verify-samples', type=int, default=200)
p.add_argument('--fixed-samples', type=int, default=100)
p.add_argument('--eval-samples', type=int, default=32)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--lr-min', type=float, default=1e-4)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--floor-mode', default='block', choices=['block', 'support'])
p.add_argument('--agents', type=int, default=20, help='only the rounding of the type weights, as in v1')
p.add_argument('--mm-steps', type=int, default=6000)
p.add_argument('--iters-warm', type=int, default=2000)
p.add_argument('--iters-major-warm', type=int, default=1000)
p.add_argument('--lr-warm', type=float, default=1e-4, help='constant lr of warm-started fine-tuning')
p.add_argument('--fresh-br', action='store_true', help='every best response trains from scratch (no warm start)')
p.add_argument('--bank-chunk', type=int, default=8192, help='markets cleared per bank-building call')
p.add_argument('--bank', type=int, default=65536, help='markets in the minors price bank')
p.add_argument('--clear-iters', type=int, default=40, help='bisection halvings per side')
p.add_argument('--major-price-taker', action='store_true',
               help='job 2 (RUNS_SETTING.md): the major best-responds as a price taker on frozen paths')
args = p.parse_args()
os.environ['KETS_FLOOR_MODE'] = args.floor_mode
os.environ['KETS_N_SIM'] = str(args.n_sim)
os.environ['KETS_CLEAR_ITERS'] = str(args.clear_iters)

import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from envs.base_params import *
from envs.environment_majmin import MajorMinorEnv
from envs.models_majmin import minor_policy, IN_DIM
from utils import CTX_REF
from utils_majmin import fictitious_play_majmin

env = MajorMinorEnv(kappa=CTX_REF[1], T=args.t_grid, generate_P_func=generate_prices_ou,
                    A0=A0_BASE, P0=P0, market_impact_func=market_impact_base,
                    generate_eps0_func=supply_noise_base,
                    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
                    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=args.agents)


# ── financial-institution warm start: the buy-then-sell regression of v1/v2,
#    on the FiLM schedule network (state 8 + public history 3, random gamma) ──
def _target(t_n, th_n, buy_frac=0.2):
    sell = -jnp.minimum(th_n / jnp.maximum(1.0 - t_n, 1e-3), 1.0)
    return jnp.where(t_n <= buy_frac, jnp.ones_like(sell), sell)


def _sample(key, B, theta_max=0.45):
    k = jax.random.split(key, 9)
    t_n = jax.random.uniform(k[0], (B,))
    th_n = jax.random.uniform(k[1], (B,), minval=0.0, maxval=theta_max)
    A_n = jax.random.uniform(k[2], (B,), minval=0.5, maxval=8.0)
    P_n = jax.random.uniform(k[3], (B,), minval=0.2, maxval=2.5)
    Aa_n = jax.random.uniform(k[4], (B,), minval=0.5, maxval=8.0)
    eps_n = jax.random.uniform(k[5], (B,), minval=0.0, maxval=0.05)
    z = jnp.zeros((B,))
    H = jnp.stack([0.05 * jax.random.normal(k[6], (B,)), jax.random.uniform(k[7], (B,)),
                   jax.random.uniform(k[8], (B,))], axis=1)
    X = jnp.stack([t_n, th_n, z, z, eps_n, A_n, P_n, Aa_n], axis=1)
    if IN_DIM > X.shape[1]:                     # public history as input (KETS_USE_H=1)
        X = jnp.concatenate([X, H], axis=1)
    ctx = jax.random.uniform(jax.random.fold_in(key, 9), (B, CTX_DIM), minval=-1.0, maxval=1.0)
    return X, ctx, _target(t_n, th_n)


def fit_mm_warmstart(path, steps, batch=4096, seed=0):
    if os.path.exists(path):
        print(f"FI warm start found: {path}", flush=True)
        return path
    params, static = eqx.partition(minor_policy(jax.random.PRNGKey(seed)), eqx.is_array)
    opt = optax.adam(1e-3)

    def loss_fn(q, key):
        X, C, Y = _sample(key, batch)
        out = jax.vmap(eqx.combine(q, static))(X, C)
        return jnp.mean((jnp.tanh(out[:, 0]) - Y) ** 2) + 1e-2 * jnp.mean((out[:, 1] - 1.0) ** 2)

    @jax.jit
    def train(q, s, key):
        def one(c, i):
            q_, s_ = c
            l, g = jax.value_and_grad(loss_fn)(q_, jax.random.fold_in(key, i))
            u, s2 = opt.update(g, s_, q_)
            return (eqx.apply_updates(q_, u), s2), l
        (qf, _), ls = jax.lax.scan(one, (q, s), jnp.arange(steps))
        return qf, ls
    t0 = time.time()
    params, ls = train(params, opt.init(params), jax.random.PRNGKey(seed + 1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    eqx.tree_serialise_leaves(path, eqx.combine(params, static))
    print(f"FI warm start fitted: mse {float(ls[0]):.4f} -> {float(ls[-1]):.5f} "
          f"in {time.time() - t0:.0f}s -> {path}", flush=True)
    return path


os.makedirs(args.model_dir, exist_ok=True)
mm_path = fit_mm_warmstart(os.path.join(args.model_dir, 'fi_warmstart.eqx'), args.mm_steps)
print("=" * 76)
print(f"MAJOR-MINOR market clearing | rung {args.rung} | ref {CTX_REF} | seed {args.seed} | "
      + (f"box: c [{C_MIN}, {C_MAX}] s [{S_MIN}, {S_MAX}] " if LAYOUT == 'v5' else f"box: alloc [{ALLOC_MIN}, {ALLOC_MAX}] ")
      + f"kappa [{KAPPA_MIN}, {KAPPA_MAX}] "
      f"floor [{FLOOR_MIN}, {FLOOR_MAX}] mm [{MM_MIN}, {MM_MAX}]")
print(f"setting: layout={LAYOUT} major_price_taker={args.major_price_taker} deterministic={DETERMINISTIC} penalty={PENALTY_MODE} no_major={NO_MAJOR} cap_f x{CAP_F_MULT:g}")
print(f"save -> {args.model_dir}   JAX devices: {jax.devices()}")
print("=" * 76, flush=True)
fictitious_play_majmin(
    env, args.rung, CTX_REF, K=args.nb_policy, T=args.t_grid, n=args.n_sim, iters=args.iters,
    iters0=args.iters_major, lr=args.lr, lr_min=args.lr_min, B=args.batch, B0=args.batch_major,
    S=args.verify_samples, S_fixed=args.fixed_samples, M=args.eval_samples,
    key=jax.random.PRNGKey(args.seed), save_dir=args.model_dir, mm_warmstart=mm_path,
    iters_warm=args.iters_warm, iters0_warm=args.iters_major_warm, n_bank=args.bank,
    lr_warm=args.lr_warm, bank_chunk=args.bank_chunk, warm_from=(10 ** 9 if args.fresh_br else 2),
    major_price_taker=args.major_price_taker,
    meta={'seed': args.seed, 'solver': f'majmin_rung{args.rung}', 'floor_max': FLOOR_MAX,
          'gamma_box': ({'c': [C_MIN, C_MAX], 's': [S_MIN, S_MAX], 'Afloor': [FLOOR_MIN, FLOOR_MAX],
                         'mm_share': [MM_MIN, MM_MAX], 'kappa': [KAPPA_MIN, KAPPA_MAX]} if LAYOUT == 'v5' else
                        {'init_alloc': [ALLOC_MIN, ALLOC_MAX], 'kappa': [KAPPA_MIN, KAPPA_MAX],
                         'Afloor': [FLOOR_MIN, FLOOR_MAX], 'mm_share': [MM_MIN, MM_MAX]}),
          'setting': {'layout': LAYOUT, 'deterministic': DETERMINISTIC, 'penalty': PENALTY_MODE, 'no_major': NO_MAJOR,
                      'cap_f_mult': CAP_F_MULT, 'ctx_ref': list(CTX_REF)}})
print("\nDone.", flush=True)
