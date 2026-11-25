"""Where and why the v2 cleared price hits the corridor bounds (floor / A_CEIL).

Takes the FINAL mixture of a v2 rung-2 run (policy triples 0..K), clears S
markets (the run's frozen verification draws + extra fresh draws), and relates
every step at the ceiling / floor to: the day t, the cumulative supply shock up
to t, the market's aggregate position at year end (E - theta, > 0 = short), the
good-price path, and the year-end penalty price kappa * A_avg.

    python3 analyze_bounds_v2.py --run-dir results_mc2_pmfg/current_kets_T30_K30_n50_block_seed42_br2x8k

Writes <run-dir>/bounds/{summary.json, *.png/pdf}.
"""
import os
import json
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
ap.add_argument('--extra-samples', type=int, default=1000, help='fresh draws on top of the frozen set')
ap.add_argument('--chunk', type=int, default=50)
ap.add_argument('--sigma-scale', type=float, default=1.0,
                help='multiply the supply shock (sigma_S) by this factor; policies unchanged')
ap.add_argument('--n-plot', type=int, default=300, help='paths drawn in paths_by_gap')
ap.add_argument('--gap-lim', type=float, default=350.0, help='colour range of the year-end gap (fixed for comparisons)')
ap.add_argument('--lw', type=float, default=0.4)
ap.add_argument('--T-eval', type=int, default=None,
                help='evaluate the mixture at this horizon (networks see normalised inputs); default: training T')
args = ap.parse_args()
with open(os.path.join(args.run_dir, 'progress.json')) as f:
    prog = json.load(f)
os.environ['KETS_FLOOR_MODE'] = prog['floor_mode']
os.environ['KETS_N_SIM'] = str(prog['n_sim'])

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from envs.base_params import *
from envs.environment_v2 import MarketClearingEnvV2, N_TYPES
from envs.models_v2 import make_buffer, buffer_set
from envs.params_v2 import A_CEIL_V2, K_MAX
from utils_v2 import (load_triple_v2, make_verification_set_v2, make_price_fn_v2,
                      block_comp_keys)
from utils import chunked_tree, draw_common_noise, draw_idio
import paper_style as ps

T_train, n = int(prog['T_grid'][0]), int(prog['n_sim'])
T = args.T_eval or T_train
ctx = tuple(prog['context'])
env = MarketClearingEnvV2(
    kappa=GAMMA_REF[1], T=T, generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
    market_impact_func=market_impact_base, generate_eps0_func=supply_noise_base,
    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=int(prog['agent_per_policy']))
env = env.set_context(jnp.asarray(ctx, jnp.float32)).set_T(T)

# final mixture: every saved triple
K = len(prog['records'])
hist = [load_triple_v2(args.run_dir, i) for i in range(K + 1)]
bufs, statics = [], []
for j in range(N_TYPES):
    b, st = make_buffer(hist[0][j], K_MAX)
    for i, tr in enumerate(hist):
        b = buffer_set(b, i, tr[j])
    bufs.append(b); statics.append(st)
bufs, statics, k = tuple(bufs), tuple(statics), len(hist)
price_fn = make_price_fn_v2(env, statics, return_diag=True)

# frozen verification draws (as in the run) + fresh draws
S0 = int(prog['S'])
ver = make_verification_set_v2(env, T, S0, int(prog['M']), n)
P1, e1 = draw_common_noise(env, T, args.extra_samples, jax.random.PRNGKey(777))
P = jnp.concatenate([ver['P'], P1])
eps0 = args.sigma_scale * jnp.concatenate([ver['eps0'], e1])   # S_t = sigma_S(gamma) * eps * sqrt(dt)
S = P.shape[0]
ck = jnp.concatenate([block_comp_keys(S0, k),
                      jax.random.split(jax.random.PRNGKey(778), args.extra_samples)])
A, rho, d = chunked_tree(lambda a, b: price_fn(bufs, jnp.int32(k), ck[a:b], P[a:b],
                                                eps0[a:b], ver['idio_pop']), S, args.chunk)
A, rho = np.asarray(A), np.asarray(rho)
d = {kk: np.asarray(v) for kk, v in d.items()}
Ac = A[:, 1:]                                   # (S, T) cleared prices
ceil, floor = d['on_ceil'].astype(bool), d['on_floor'].astype(bool)
eps = np.asarray(eps0)
supply = float(env.sigma_S()) * eps * np.sqrt(float(env.dt))  # (S, T) exogenous net supply S_t (eps already scaled)
cum_S = np.cumsum(supply, axis=1)               # > 0: outside traders sold to the agents
gap = d['agg_gap_T']                            # (S,) E - theta at T, > 0 market short
short = gap > 0
A_avg = np.cumsum(Ac, axis=1) / np.arange(1, T + 1)
kap = float(env.kappa)
Pn = np.asarray(P)[:, 1:]

days = np.arange(T)
ceil_any, floor_end = ceil.any(axis=1), floor[:, -1]
summ = {
    'run_dir': args.run_dir, 'sigma_scale': args.sigma_scale, 'T_train': T_train, 'mixture_size': k, 'S': S, 'T': T, 'kappa': kap,
    'A_CEIL': A_CEIL_V2, 'floor': float(env.Afloor),
    'frac_steps_ceil': float(ceil.mean()), 'frac_steps_floor': float(floor.mean()),
    'frac_paths_hit_ceil': float(ceil_any.mean()),
    'frac_paths_end_on_floor': float(floor_end.mean()),
    'market_short_frac': float(short.mean()),
    'P(hit ceil | short)': float(ceil_any[short].mean()) if short.any() else None,
    'P(hit ceil | long)': float(ceil_any[~short].mean()) if (~short).any() else None,
    'P(end on floor | long)': float(floor_end[~short].mean()) if (~short).any() else None,
    'P(end on floor | short)': float(floor_end[short].mean()) if short.any() else None,
    'first_ceil_day_quantiles': (np.quantile(np.argmax(ceil[ceil_any], axis=1), [.1, .5, .9]).tolist()
                                 if ceil_any.any() else None),
    'ceil_steps_by_day': ceil.mean(axis=0).tolist(),
    'floor_steps_by_day': floor.mean(axis=0).tolist(),
    'final_price_quantiles': np.quantile(Ac[:, -1], [.05, .1, .25, .5, .75, .9, .95]).tolist(),
    'kappa_Aavg_final_median_short': float(np.median(kap * A_avg[short, -1])) if short.any() else None,
    'final_price_median_short': float(np.median(Ac[short, -1])) if short.any() else None,
    'final_price_median_long': float(np.median(Ac[~short, -1])) if (~short).any() else None,
    'corr(cum_supply_T, gap_T)': float(np.corrcoef(cum_S[:, -1], gap)[0, 1]),
    'corr(mean_P, gap_T)': float(np.corrcoef(Pn.mean(axis=1), gap)[0, 1]),
    'cum_supply_T_std': float(cum_S[:, -1].std()),
    'rho_buy_min': float(rho[..., 1].min()), 'rho_sell_min': float(rho[..., 0].min()),
}
tag = ('' if args.sigma_scale == 1.0 else f'_sigma{args.sigma_scale:g}') + ('' if T == T_train else f'_T{T}')
out = os.path.join(args.run_dir, 'bounds' + tag)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, 'summary.json'), 'w') as f:
    json.dump(summ, f, indent=2)
for kk, v in summ.items():
    if not isinstance(v, list) or len(v) <= 8:
        print(f"{kk:32s} {v}")

ps.apply()
# 1. share of steps on each bound, by day
fig, ax = ps.line_panel(figsize=(3.1, 2.1))
ax.plot(days + 1, 100 * ceil.mean(axis=0), '-o', ms=2, color='#c0392b', label=f'ceiling ({A_CEIL_V2:g})')
ax.plot(days + 1, 100 * floor.mean(axis=0), '-s', ms=2, color='#2a78d6', label='floor')
ax.set_xlabel('Day $t$'); ax.set_ylabel('Paths on the bound (%)'); ax.legend(frameon=False)
ps.save(fig, os.path.join(out, 'bound_share_by_day'), check=False)

# 2. price paths coloured by the year-end aggregate position
fig, ax = ps.line_panel(figsize=(3.1, 2.3))
lim = args.gap_lim if args.gap_lim else np.quantile(np.abs(gap), 0.95)
cmap = plt.get_cmap('coolwarm')
idx = np.random.default_rng(0).choice(S, min(S, args.n_plot), replace=False)
for i in idx:
    ax.plot(np.arange(T + 1), A[i], color=cmap(0.5 + 0.5 * np.clip(gap[i] / lim, -1, 1)),
            lw=args.lw, alpha=0.6)
ax.axhline(A_CEIL_V2, color='k', lw=0.6, ls='--'); ax.axhline(float(env.Afloor), color='k', lw=0.6, ls=':')
ax.set_xlabel('Day $t$'); ax.set_ylabel('Cleared price (USD/tCO$_2$)')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
cb = fig.colorbar(sm, ax=ax, pad=0.02); cb.set_label('Year-end gap $E-\\theta$ (> 0 short)', fontsize=6)
ps.save(fig, os.path.join(out, 'paths_by_gap'), check=False)

# 3. cumulative supply shock vs year-end gap, marking ceiling / floor paths
fig, ax = ps.line_panel(figsize=(3.1, 2.3))
m_c, m_f = ceil_any, floor_end & ~ceil_any
other = ~(m_c | m_f)
ax.scatter(cum_S[other, -1], gap[other], s=2, c='0.6', label='interior')
ax.scatter(cum_S[m_f, -1], gap[m_f], s=3, c='#2a78d6', label='ends on floor')
ax.scatter(cum_S[m_c, -1], gap[m_c], s=3, c='#c0392b', label='hits ceiling')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Cumulative supply shock (allowances per agent)')
ax.set_ylabel('Year-end gap $E-\\theta$'); ax.legend(frameon=False, fontsize=6)
ps.save(fig, os.path.join(out, 'supply_vs_gap'), check=False)

# 4. final price vs kappa * A_avg
fig, ax = ps.line_panel(figsize=(3.1, 2.3))
ax.scatter(kap * A_avg[:, -1], Ac[:, -1], s=2, c=np.where(short, '#c0392b', '#2a78d6'))
xx = np.linspace(0, max(kap * A_avg[:, -1].max(), A_CEIL_V2), 10)
ax.plot(xx, xx, 'k--', lw=0.5); ax.axhline(A_CEIL_V2, color='k', lw=0.5, ls=':')
ax.set_xlabel(r'$\kappa\,\bar A_T$ (year-end penalty price)')
ax.set_ylabel('Last cleared price $A_{T-1}$')
ps.save(fig, os.path.join(out, 'final_price_vs_penalty'), check=False)
print(f"-> {out}")
