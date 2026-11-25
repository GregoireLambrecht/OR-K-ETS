"""Rung 2 (PMFG at one fixed context) with market clearing VERSION 2
(MARKET_CLEARING.md rev. 7be619c): blocking corridor [floor, 200] with
two-sided rationing, per-agent population sampling, two-BR selection.

    python3 main_pmfg_v2.py --scenario current_kets --model-dir results_mc2_pmfg/... \
        --n-sim 50 --floor-mode block --nb-policy 30 --t-grid 30 --iters 8000 --br-restarts 2
"""
import os
import sys
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

p = argparse.ArgumentParser(description='K-ETS rung 2, market clearing v2')
p.add_argument('--scenario', default='current_kets')
p.add_argument('--floor-max', type=float, default=30.0)
p.add_argument('--model-dir', required=True)
p.add_argument('--nb-policy', type=int, default=30)
p.add_argument('--n-sim', type=int, default=128, help='simulated agents per type and market')
p.add_argument('--floor-mode', default='block', choices=['block', 'support'])
p.add_argument('--agents', type=int, default=20,
               help='only sets the type weights w_j (context_weights rounding), as in v1')
p.add_argument('--batch', type=int, default=256)
p.add_argument('--verify-samples', type=int, default=200)
p.add_argument('--eval-samples', type=int, default=32)
p.add_argument('--eval-chunk', type=int, default=50)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--t-grid', type=int, default=30)
p.add_argument('--iters', type=int, default=8000)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--lr-min', type=float, default=1e-4)
p.add_argument('--br-restarts', type=int, default=2)
p.add_argument('--mm-steps', type=int, default=6000)
p.add_argument('--mm-batch', type=int, default=4096)
args = p.parse_args()

# read once at import by envs.params_v2
os.environ['KETS_FLOOR_MODE'] = args.floor_mode
os.environ['KETS_N_SIM'] = str(args.n_sim)

import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from envs.base_params import *
from envs.environment_v2 import MarketClearingEnvV2
from envs.models_v2 import TypePolicyNNV2
from envs.params_v2 import FLOOR_MODE, N_SIM
from utils_v2 import fictitious_play_rung2_v2
from audit_pmfg import audit_scenarios
assert FLOOR_MODE == args.floor_mode and N_SIM == args.n_sim

ctx = audit_scenarios(args.floor_max)[args.scenario]
env = MarketClearingEnvV2(
    kappa=GAMMA_REF[1], T=args.t_grid,
    generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
    market_impact_func=market_impact_base,          # unused under clearing
    generate_eps0_func=supply_noise_base,           # net supply shock
    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE,
    agent_per_policy=args.agents,
)


# ── market-maker warm start (same regression as main_pmfg.py, v2 network) ──
def _target(t_n, th_n, buy_frac=0.2):
    sell = -jnp.minimum(th_n / jnp.maximum(1.0 - t_n, 1e-3), 1.0)
    return jnp.where(t_n <= buy_frac, jnp.ones_like(sell), sell)


def _sample(key, B, theta_max=0.45):
    k = jax.random.split(key, 6)
    t_n = jax.random.uniform(k[0], (B,))
    th_n = jax.random.uniform(k[1], (B,), minval=0.0, maxval=theta_max)
    A_n = jax.random.uniform(k[2], (B,), minval=0.5, maxval=8.0)
    P_n = jax.random.uniform(k[3], (B,), minval=0.2, maxval=2.5)
    Aa_n = jax.random.uniform(k[4], (B,), minval=0.5, maxval=8.0)
    eps_n = jax.random.uniform(k[5], (B,), minval=0.0, maxval=0.05)
    z = jnp.zeros((B,))
    return jnp.stack([t_n, th_n, z, z, eps_n, A_n, P_n, Aa_n], axis=1), _target(t_n, th_n)


BETA_PULL = 1e-2


def fit_mm_warmstart(path, steps, batch, seed=0):
    if os.path.exists(path):
        print(f"market-maker warm start found: {path}", flush=True)
        return path
    model = TypePolicyNNV2(key=jax.random.PRNGKey(seed))
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(1e-3)
    st = opt.init(params)

    def loss_fn(q, key):
        S, Y = _sample(key, batch)
        out = jax.vmap(eqx.combine(q, static))(S)
        return (jnp.mean((jnp.tanh(out[:, 0]) - Y) ** 2)
                + BETA_PULL * jnp.mean((out[:, 1] - 1.0) ** 2))

    @jax.jit
    def train(q, s, key):
        def one(carry, n):
            q_, s_ = carry
            l, g = jax.value_and_grad(loss_fn)(q_, jax.random.fold_in(key, n))
            u, s_new = opt.update(g, s_, q_)
            return (eqx.apply_updates(q_, u), s_new), l
        (q_f, _), ls = jax.lax.scan(one, (q, s), jnp.arange(steps))
        return q_f, ls

    t0 = time.time()
    params, losses = train(params, st, jax.random.PRNGKey(seed + 1))
    losses = np.asarray(losses)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    eqx.tree_serialise_leaves(path, eqx.combine(params, static))
    print(f"market-maker warm start fitted: {steps} steps, mse {losses[0]:.5f} -> "
          f"{losses[-1]:.7f} in {time.time() - t0:.0f}s -> {path}", flush=True)
    return path


os.makedirs(args.model_dir, exist_ok=True)
mm_path = fit_mm_warmstart(os.path.join(args.model_dir, 'mm_warmstart_v2.eqx'),
                           args.mm_steps, args.mm_batch)
print("=" * 76)
print(f"PMFG MARKET CLEARING v2  |  scenario {args.scenario} = {tuple(ctx)}  |  seed {args.seed}")
print(f"save -> {args.model_dir}   JAX devices: {jax.devices()}")
print(f"T {args.t_grid}, BR {args.br_restarts} x {args.iters} steps, lr {args.lr} cosine -> "
      f"{args.lr_min:g}, K = {args.nb_policy}, B = {args.batch}, S = {args.verify_samples}, "
      f"M = {args.eval_samples}, n = {args.n_sim}/type, floor mode {args.floor_mode}")
print("=" * 76, flush=True)

history, records = fictitious_play_rung2_v2(
    env, gamma_ref=tuple(float(x) for x in ctx), K=args.nb_policy, T=args.t_grid,
    iters=args.iters, lr=args.lr, lr_min=args.lr_min, B=args.batch, S=args.verify_samples,
    M=args.eval_samples, n=args.n_sim, eval_chunk=args.eval_chunk,
    key=jax.random.PRNGKey(args.seed), save_dir=args.model_dir, mm_warmstart=mm_path,
    br_restarts=args.br_restarts,
    meta={'solver': 'pmfg_v2', 'scenario': args.scenario, 'seed': args.seed,
          'context': [float(x) for x in ctx]})
print(f"\nDone — {len(history)} policy triples, {len(records)} records.", flush=True)
