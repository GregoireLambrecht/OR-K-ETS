"""LP equilibrium trajectories alone (no FP run needed), on the benchmark grid.
Run with the benchmark environment (run_benchmark.sh exports it; MODE=lp).

    python3 plot_lp_trajectories.py --out results_majmin/lp_only [--res 8 --mm 0 0.2 0.5]

trajectories_lp.pdf: one page per scenario (price a*, daily production per agent and
type, cumulative emissions, holdings of the existence proof); lp_<regime>_mm<..>.png:
one scenario per regime and mm; lp_overview.png: a*, and the number of fossil days
per producer, over init_alloc x kappa^1.
"""
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--res', type=int, default=8)
ap.add_argument('--mm', type=float, nargs='+', default=[0.0, 0.2, 0.5])
ap.add_argument('--T', type=int, default=100)
args = ap.parse_args()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import paper_style as ps
from envs.base_params import ALLOC_MIN, ALLOC_MAX, KAPPA_MIN, KAPPA_MAX
from majmin_eval_common import make_env
from benchmark_lp import market_data, planner, equilibrium_paths, p_days

ps.apply()
T = args.T
env = make_env(T)
os.makedirs(args.out, exist_ok=True)
al = (np.floor(np.linspace(ALLOC_MIN, ALLOC_MAX, args.res) * T - 0.5).clip(min=np.ceil(ALLOC_MIN * T - 0.5)) + 0.5) / T
ka = np.linspace(KAPPA_MIN, KAPPA_MAX, args.res)
G = [(a, k, 0.0, mm) for mm in args.mm for a in al for k in ka]
NAMES = {0: ('private producer', 'C0'), 2: ('financial inst.', 'C2')}
rows = []


def page(g):
    m = market_data(env, g)
    pl = planner(m)
    paths = equilibrium_paths(m)
    t, t1 = np.arange(T), np.arange(T + 1)
    fig, axs = plt.subplots(1, 4, figsize=(9.0, 2.1))
    axs[0].plot(t, np.full(T, pl['a_star']), color='C3', lw=1.2, label=r'$a^\star$')
    axs[0].axhline(g[1], color='0.6', ls=':', lw=0.8, label=r'fine $\kappa^1$')
    p = p_days(m)
    d0 = m['types'][0]
    axs[0].plot(t, (p - d0['c'][0]) / d0['e'][0], color='0.3', ls='--', lw=0.8, label='fossil shutdown price')
    axs[0].legend(fontsize=5, frameon=False); axs[0].set_title('Price (USD/tCO$_2$)', fontsize=7)
    for d in m['types']:
        j = d['j']; lab, col = NAMES[j]; pth = paths[j]
        axs[1].plot(t, pth['q'][:, 0], color=col, lw=1, label=f'{lab}, fossil')
        axs[1].plot(t, pth['q'][:, 1], color=col, ls=':', lw=1, label=f'{lab}, gas')
        axs[2].plot(t1, pth['E'], color=col, lw=1, label=lab)
        axs[3].plot(t1, pth['theta'], color=col, lw=1, label=lab)
    axs[1].set_title('Production per agent and day', fontsize=7)
    axs[2].set_title('Cumulative emissions per agent', fontsize=7)
    axs[3].set_title('Holdings per agent (one timing)', fontsize=7)
    axs[1].legend(fontsize=4.5, frameon=False); axs[3].legend(fontsize=4.5, frameon=False)
    for ax in axs:
        ax.set_xlabel('Day $t$', fontsize=7); ax.tick_params(labelsize=6)
    q0 = paths[0]['q'][:, 0]
    fdays = float(q0.sum() / m['types'][0]['Cxi'])
    fig.suptitle(f"init_alloc {100 * g[0]:.1f}%, $\\kappa^1$ = {g[1]:.1f}, financial inst. {100 * g[3]:.0f}%  |  "
                 f"{pl['regime']}, $a^\\star$ = {pl['a_star']:.2f} (unique {pl['unique']})  |  "
                 f"fossil days per producer {fdays:.1f}", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    rows.append((g, pl['a_star'], pl['regime'], fdays))
    return fig, pl['regime']


picked = {}
with PdfPages(os.path.join(args.out, 'trajectories_lp.pdf')) as pdf:
    for g in G:
        fig, r = page(g)
        pdf.savefig(fig); plt.close(fig)
        picked.setdefault((r, g[3]), []).append(g)
for (r, mm), gs in picked.items():
    fig, _ = page(gs[len(gs) // 2])
    ps.save(fig, os.path.join(args.out, f'lp_{r}_mm{int(round(100 * mm))}'), check=False)

# overview maps: a* and fossil days, init_alloc x kappa^1, per mm
n = len(G)
rows = rows[:n]
fig, axs = plt.subplots(2, len(args.mm), figsize=(2.3 * len(args.mm) + 0.8, 4.0), squeeze=False)
for c, mm in enumerate(args.mm):
    for r_, (key, lab, cmap) in enumerate(((1, r'$a^\star$ (USD/tCO$_2$)', 'viridis'), (3, 'fossil days', 'cividis'))):
        Z = np.full((len(ka), len(al)), np.nan)
        for row in rows:
            g = row[0]
            if np.isclose(g[3], mm):
                Z[np.searchsorted(ka, g[1]), np.searchsorted(al, g[0])] = row[key]
        im = axs[r_][c].pcolormesh(100 * al, ka, Z, cmap=cmap, shading='nearest')
        fig.colorbar(im, ax=axs[r_][c], label=lab)
        axs[r_][c].set_title(f'financial inst. {100 * mm:.0f}%', fontsize=7)
        axs[r_][c].set_xlabel('Initial allocation (%)', fontsize=7)
        axs[r_][0].set_ylabel(r'Fine $\kappa^1$', fontsize=7)
fig.tight_layout()
ps.save(fig, os.path.join(args.out, 'lp_overview'), check=False)
print(f"-> {args.out}: trajectories_lp.pdf ({n} scenarios), lp_*.png, lp_overview.png", flush=True)
