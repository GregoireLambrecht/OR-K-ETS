"""Rung 3 entry point — one BE-MFG solve covering the whole scenario family.

Differences from Rung 2 (main_rung2.py), which is left untouched:
  * every market in the training batch draws its own fresh scenario alongside
    its fresh common and idiosyncratic noise, so one solve covers the family;
  * the metric is measured on four frozen blocks — the prior expectation plus
    three fixed scenarios (plan.md §8.5.2);
  * policies are FiLM-conditioned, so the market-maker warm start must be the
    Rung 3 one (pretrain_mm_r3.py).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import jax
import jax.numpy as jnp
import equinox as eqx

from envs.environment import ExogenousMarketEnvJAX
from envs.base_params import *
from utils import (fictitious_play_rung3, FIXED_SCENARIOS, CTX_NAMES,
                   CTX_REF, CONTEXT_DIM, _G_LO, _G_HI)


p = argparse.ArgumentParser(description='K-ETS — Rung 3 (BE-MFG)')
p.add_argument('mode', choices=['choice', 'bau'])
p.add_argument('--model-dir', type=str, default='results_rung3')
p.add_argument('--plot-dir',  type=str, default=None)
p.add_argument('--nb-policy', type=int, default=50, help='FP iterations K')
p.add_argument('--agents',    type=int, default=20,
               help='agents per policy block')
p.add_argument('--batch', type=int, default=256,
               help='independent markets per gradient step; each draws its own '
                    '(gamma, common noise, idiosyncratic noise)')
p.add_argument('--verify-samples', type=int, default=500,
               help='S: frozen joint (gamma, noise) draws for the prior block')
p.add_argument('--fixed-samples', type=int, default=250,
               help='S_fixed: frozen noise draws for EACH fixed scenario')
p.add_argument('--eval-samples', type=int, default=32,
               help='M: paired agents per type per frozen row')
p.add_argument('--eval-chunk', type=int, default=50,
               help='chunk size when pricing a frozen block')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--t-grid', type=str, default='365')
p.add_argument('--iters',  type=str, default='8000')
p.add_argument('--lr',     type=str, default='0.001')
p.add_argument('--lr-schedule', type=str, default=None, choices=['cosine'])
p.add_argument('--lr-min',      type=float, default=None)
p.add_argument('--mm-warmstart', type=str,
               default='mm_warmstart_r3/mm_policy.eqx',
               help='saved FiLM market-maker warm start; "" disables it')
args = p.parse_args()


def _parse(spec, cast):
    return tuple(cast(x) for x in spec.split(','))


T_GRID = _parse(args.t_grid, int)
ITERS  = _parse(args.iters,  int)
LRS    = _parse(args.lr,     float)
if not (len(T_GRID) == len(ITERS) == len(LRS)):
    raise SystemExit(f"--t-grid/--iters/--lr must be equal length, got "
                     f"{len(T_GRID)}/{len(ITERS)}/{len(LRS)}")

save_dir = os.path.join(args.model_dir, args.mode)
plot_dir = args.plot_dir or os.path.join(save_dir, 'figs')

env = ExogenousMarketEnvJAX(
    kappa=GAMMA_REF[1], T=T_GRID[-1],
    generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
    market_impact_func=market_impact_base,
    generate_eps0_func=white_noise_A_base,
    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE,
    agent_per_policy=args.agents,
)
if args.mode == 'bau':
    bp, bq = list(PRIVATE_GENERATOR), list(BIG_PUBLIC_GENERATOR)
    bp[8], bq[8] = TEC_PRIVATE, TEC_PUBLIC
    env = eqx.tree_at(lambda e: e.agent_params, env,
                      jnp.array([bp, bq, MARKET_MAKER]))

n1, n2, n3 = env.type_counts
print("=" * 70, flush=True)
print(f"Rung 3 (BE-MFG)  |  mode: {args.mode.upper()}  |  save: {save_dir}")
print(f"plots      -> {plot_dir}")
print(f"JAX devices: {jax.devices()}")
print(f"context    = {CTX_NAMES}   (uniform prior q, {CONTEXT_DIM} dims)")
for _n, _lo, _hi in zip(CTX_NAMES, _G_LO, _G_HI):
    print(f"             {_n:<15} [{float(_lo):.4g}, {float(_hi):.4g}]")
print(f"reference  = {dict(zip(CTX_NAMES, CTX_REF))}")
print(f"floor box  = [{FLOOR_MIN}, {FLOOR_MAX}]   "
      f"(KETS_FLOOR_MAX={os.environ.get('KETS_FLOOR_MAX', 'unset, default')})")
print("fixed rows = " + ", ".join(n for n, _ in FIXED_SCENARIOS))
print(f"T grid     = {T_GRID}, iters {ITERS}, lr {LRS}")
print(f"population = {args.agents}/block  ->  private {n1}, large {n2}, mm {n3}")
print(f"training   : B = {args.batch} markets/gradient step, each with a FRESH "
      f"(gamma, common noise, idio noise)")
print(f"evaluation : prior S = {args.verify_samples}, each fixed scenario "
      f"S = {args.fixed_samples}, M = {args.eval_samples}, "
      f"chunk {args.eval_chunk}")
print(f"K = {args.nb_policy}   mm warm start = {args.mm_warmstart or 'OFF'}")
if args.mode == 'bau':
    print(f"tech fixed: private {TEC_PRIVATE:.4f}, public {TEC_PUBLIC:.4f}")
print("=" * 70, flush=True)

history, records = fictitious_play_rung3(
    env, K=args.nb_policy, T_grid=T_GRID, iters=ITERS, lrs=LRS,
    B=args.batch, S=args.verify_samples, S_fixed=args.fixed_samples,
    M=args.eval_samples, eval_chunk=args.eval_chunk,
    lr_schedule=args.lr_schedule, lr_min=args.lr_min,
    mm_warmstart=(args.mm_warmstart or None),
    key=jax.random.PRNGKey(args.seed),
    save_dir=save_dir, plot_dir=plot_dir,
    meta={'mode': args.mode, 'seed': args.seed, 'T_grid_cli': list(T_GRID)},
)

print(f"\nDone — {len(history)} policy triples.", flush=True)
for r in records:
    b = r['blocks']
    print(f"  it {r['iteration']:>3}   "
          f"prior {b['prior']['weighted_relative']:+.2%}   "
          + "   ".join(f"{n} {b[n]['weighted_relative']:+.2%}"
                       for n, _ in FIXED_SCENARIOS), flush=True)
