"""PMFG fictitious play at ONE fixed context, unconditional policies.

market-clearing branch: the price is cleared every step (envs/environment.py
_population_step), eps0 is the exogenous net supply, the policies output a
demand schedule (alpha, beta) + (xi, eta). Use new result directories
(results_mc_pmfg/...): checkpoints from main do not load.

    python3 main_pmfg.py --scenario current_kets --seed 1 --model-dir results_pmfg/current_kets_seed1

The computational counterpart of the BE-MFG solve: the same fictitious-play
loop (fresh common noise every gradient step, frozen evaluation block), the
same budget per best response, but at a single scenario with plain
TypePolicyNN policies. This is utils.fictitious_play_rung2 with the scenario
bound as a full 4-vector.

Per iteration it records, in <model-dir>/progress.json, the exploitability per
type and population-weighted (against a fresh unconditional best response on
the frozen block), the best-response training time `br_seconds` (training
only, the exploitability evaluation is not in it), and the full training-loss
curves in <model-dir>/losses/br_loss_<k>.npz.

The market maker starts from the same buy-then-sell regression as the BE runs,
fitted here on a TypePolicyNN (the FiLM warm start does not load into it).
"""

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from envs.environment import ExogenousMarketEnvJAX
from envs.base_params import *
from envs.models import TypePolicyNN
from utils import fictitious_play_rung2
from audit_pmfg import audit_scenarios

p = argparse.ArgumentParser(description='K-ETS — PMFG fictitious play at one scenario')
p.add_argument('--scenario', default='current_kets')
p.add_argument('--floor-max', type=float, default=30.0,
               help='box top that defines high_floor_mid_alloc; irrelevant otherwise')
p.add_argument('--model-dir', type=str, required=True)
p.add_argument('--nb-policy', type=int, default=50)
p.add_argument('--agents', type=int, default=20)
p.add_argument('--batch', type=int, default=256)
p.add_argument('--verify-samples', type=int, default=500)
p.add_argument('--eval-samples', type=int, default=32)
p.add_argument('--eval-chunk', type=int, default=50)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--t-grid', type=str, default='365')
p.add_argument('--iters', type=str, default='12000')
p.add_argument('--lr', type=str, default='0.001')
p.add_argument('--lr-schedule', type=str, default='cosine', choices=['cosine'])
p.add_argument('--lr-min', type=float, default=1e-4)
p.add_argument('--br-restarts', type=int, default=1,
               help='independent fresh BRs per type per FP iteration; best kept by train loss')
p.add_argument('--mm-steps', type=int, default=6000)
p.add_argument('--mm-batch', type=int, default=4096)
args = p.parse_args()


def _parse(spec, cast):
    return tuple(cast(x) for x in spec.split(','))


T_GRID = _parse(args.t_grid, int)
ITERS = _parse(args.iters, int)
LRS = _parse(args.lr, float)
ctx = audit_scenarios(args.floor_max)[args.scenario]

env = ExogenousMarketEnvJAX(
    kappa=GAMMA_REF[1], T=T_GRID[-1],
    generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
    market_impact_func=market_impact_base,
    generate_eps0_func=supply_noise_base,      # net supply (market clearing)
    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE,
    agent_per_policy=args.agents,
)

# ── unconditional market-maker warm start: the buy-then-sell regression of
#    pretrain_mm_r3.py on a TypePolicyNN, same synthetic states, no context ──
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
    return jnp.stack([t_n, th_n, z, z, eps_n, A_n, P_n, Aa_n], axis=1), \
        _target(t_n, th_n)


BETA_PULL = 1e-2


def fit_mm_warmstart(path, steps, batch, seed=0):
    if os.path.exists(path):
        print(f"market-maker warm start found: {path}", flush=True)
        return path
    model = TypePolicyNN(key=jax.random.PRNGKey(seed))
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(1e-3)
    st = opt.init(params)

    def loss_fn(q, key):
        # market clearing: at an unchanged price (a = A_pre) the centred
        # schedule trades cap_f * tanh(alpha), so fit tanh(alpha) to the
        # heuristic and pull beta weakly towards 1 (MARKET_CLEARING.md §2.5)
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
    print(f"market-maker warm start fitted: {steps} steps, batch {batch}, "
          f"mse {losses[0]:.5f} -> {losses[-1]:.7f} in {time.time() - t0:.0f}s "
          f"-> {path}", flush=True)
    return path


os.makedirs(args.model_dir, exist_ok=True)
mm_path = fit_mm_warmstart(os.path.join(args.model_dir, 'mm_warmstart_pmfg.eqx'),
                           args.mm_steps, args.mm_batch)

print("=" * 76, flush=True)
print(f"PMFG + MARKET CLEARING (unconditional FP)  |  scenario {args.scenario} = {tuple(ctx)}  "
      f"|  seed {args.seed}")
print(f"save       -> {args.model_dir}")
print(f"JAX devices: {jax.devices()}")
print(f"T {T_GRID}, iters {ITERS}, lr {LRS} {args.lr_schedule} -> {args.lr_min:g}, BR restarts {args.br_restarts}")
print(f"K = {args.nb_policy}, B = {args.batch}, S = {args.verify_samples}, "
      f"M = {args.eval_samples}")
print("=" * 76, flush=True)

history, records = fictitious_play_rung2(
    env, gamma_ref=tuple(float(x) for x in ctx), K=args.nb_policy,
    T_grid=T_GRID, iters=ITERS, lrs=LRS, B=args.batch,
    S=args.verify_samples, M=args.eval_samples, eval_chunk=args.eval_chunk,
    key=jax.random.PRNGKey(args.seed), save_dir=args.model_dir, plot_dir=None,
    mm_warmstart=mm_path, lr_schedule=args.lr_schedule, lr_min=args.lr_min,
    br_restarts=args.br_restarts,
    meta={'solver': 'pmfg', 'scenario': args.scenario, 'seed': args.seed,
          'context': [float(x) for x in ctx]},
)
print(f"\nDone — {len(history)} policy triples, {len(records)} records.", flush=True)
