"""Figures of the LP benchmark (benchmark_eval.py output).

    python3 plot_benchmark.py --bench results_majmin/<run>/benchmark

    convergence_exploitability   exact exploitability vs FP iteration (population mean and max, per type)
    convergence_price            max_t |A_t - a*| / a* vs FP iteration (mean, median)
    map_<q>                      init_alloc x kappa^1 maps at the last checkpoint, one panel per mm_share:
                                 a_star, price (time-mean A^K), price_err, E_pop, regime
    price_paths                  A^K_t against a* for one scenario per regime
    estimate_vs_exact            the run's trained estimate vs the exact exploitability at iteration K
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps

ap = argparse.ArgumentParser()
ap.add_argument('--bench', required=True)
ap.add_argument('--mm', type=float, nargs='+', default=[0.0, 0.2, 0.5], help='mm_share panels of the maps')
args = ap.parse_args()
ps.apply()
d = np.load(os.path.join(args.bench, 'benchmark.npz'), allow_pickle=False)
G, ck, a_star, regime = d['G'], d['ckpts'], d['a_star'], d['regime'].astype(str)
fig_dir = os.path.join(args.bench, 'figs')
REG = {'cap_slack': 0, 'interior': 1, 'fine': 2}

# ── convergence ──────────────────────────────────────────────────────────────
fig, ax = ps.line_panel()
ax.plot(ck, 100 * d['E_pop'].mean(axis=1), 'o-', color='k', label='population, mean', ms=3)
ax.plot(ck, 100 * d['E_pop'].max(axis=1), 's--', color='0.5', label='population, max', ms=3)
for a, (lab, col) in enumerate((('private producer', 'C0'), ('financial inst.', 'C2'))):
    ax.plot(ck, 100 * np.nanmean(d['E_rel'][:, :, a], axis=1), '^:', color=col, label=lab, ms=3)
ax.set_yscale('log'); ax.set_xlabel('Fictitious-play iteration $k$'); ax.set_ylabel('Exact exploitability (%)')
ax.legend(fontsize=6, frameon=False)
ps.save(fig, os.path.join(fig_dir, 'convergence_exploitability'))

fig, ax = ps.line_panel()
ax.plot(ck, 100 * d['price_err'].mean(axis=1), 'o-', color='k', label='mean', ms=3)
ax.plot(ck, 100 * np.median(d['price_err'], axis=1), 's--', color='0.5', label='median', ms=3)
ax.set_yscale('log'); ax.set_xlabel('Fictitious-play iteration $k$')
ax.set_ylabel(r'$\max_t |A_t - a^\star| / a^\star$ (%)')
ax.legend(fontsize=6, frameon=False)
ps.save(fig, os.path.join(fig_dir, 'convergence_price'))

# ── maps at the last checkpoint ──────────────────────────────────────────────
al, ka = np.unique(G[:, 0]), np.unique(G[:, 1])
mm_vals = [m for m in args.mm if np.any(np.isclose(G[:, 3], m))]
quant = {'a_star': (a_star, r'$a^\star$ (USD/tCO$_2$)', 'viridis'),
         'price': (d['A'][-1].mean(axis=1), r'mean $A^K_t$ (USD/tCO$_2$)', 'viridis'),
         'price_err': (100 * d['price_err'][-1], r'$\max_t|A_t-a^\star|/a^\star$ (%)', 'magma'),
         'E_pop': (100 * d['E_pop'][-1], 'exact exploitability (%)', 'magma'),
         'regime': (np.array([REG[r] for r in regime], float), 'regime (0 slack, 1 interior, 2 fine)', 'cividis')}
for nm, (v, lab, cmap) in quant.items():
    fig, axs = plt.subplots(1, len(mm_vals), figsize=(2.2 * len(mm_vals) + 0.6, 2.0), squeeze=False)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    for ax, mm in zip(axs[0], mm_vals):
        sel = np.isclose(G[:, 3], mm)
        Z = np.full((len(ka), len(al)), np.nan)
        for g, z in zip(G[sel], v[sel]):
            Z[np.searchsorted(ka, g[1]), np.searchsorted(al, g[0])] = z
        im = ax.pcolormesh(100 * al, ka, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        ax.set_title(f'financial inst. {100 * mm:.0f}%', fontsize=7)
        ax.set_xlabel('Initial allocation (%)')
    axs[0][0].set_ylabel(r'Fine $\kappa^1$ (USD/tCO$_2$)')
    fig.colorbar(im, ax=axs[0].tolist(), label=lab, shrink=0.9)
    ps.save(fig, os.path.join(fig_dir, f'map_{nm}'), check=False)

# ── price paths, one scenario per regime ─────────────────────────────────────
fig, ax = ps.line_panel(ps.RUNTIME, dict(left=0.45, right=0.62, bottom=0.32, top=0.05))
t = np.arange(d['A'].shape[-1])
for r, col in zip(('cap_slack', 'interior', 'fine'), ('C0', 'C1', 'C3')):
    idx = np.where(regime == r)[0]
    if len(idx) == 0:
        continue
    s = idx[len(idx) // 2]
    ax.plot(t, d['A'][-1][s], color=col, lw=1,
            label=f"{r}: alloc {100 * G[s, 0]:.0f}%, $\\kappa^1$ {G[s, 1]:.0f}")
    ax.axhline(a_star[s], color=col, ls='--', lw=0.8)
ax.set_xlabel('Day $t$'); ax.set_ylabel('Cleared price (USD/tCO$_2$)')
ax.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.0, 0.5))
ps.save(fig, os.path.join(fig_dir, 'price_paths'), check=False)

# ── trained estimate vs exact at iteration K ─────────────────────────────────
fig, ax = ps.line_panel()
sc = d['scales']
for a, (lab, col) in enumerate((('private producer', 'C0'), ('financial inst.', 'C2'))):
    x, y = 100 * d['exact_K'][:, a] / sc[a], 100 * d['est_K'][:, a] / sc[a]
    ok = d['wts'][:, a] > 0
    ax.scatter(x[ok], y[ok], s=4, color=col, label=lab, alpha=0.6)
lim = [0, np.nanmax(100 * d['exact_K'] / sc[None, :])]
ax.plot(lim, lim, color='0.5', lw=0.8)
ax.set_xlabel('Exact exploitability (%)'); ax.set_ylabel('Trained estimate (%)')
ax.legend(fontsize=6, frameon=False)
ps.save(fig, os.path.join(fig_dir, 'estimate_vs_exact'))

# ── trajectories: LP equilibrium vs FP mixture, every scenario ────────────────
from matplotlib.backends.backend_pdf import PdfPages
TYPES = (('private producer', 'C0'), ('financial inst.', 'C2'))


def traj_page(s):
    """Four panels for scenario s: price, daily production, cumulative emissions, holdings.
    Solid: FP mixture (last checkpoint); dashed: LP equilibrium."""
    T = d['A'].shape[-1]
    t, t1 = np.arange(T), np.arange(T + 1)
    fig, axs = plt.subplots(1, 4, figsize=(9.0, 2.1))
    ax = axs[0]
    ax.plot(t, d['A'][-1][s], color='k', lw=1, label=f"FP (k = {int(d['traj_m'])})")
    ax.axhline(a_star[s], color='C3', ls='--', lw=1, label=r'LP $a^\star$')
    ax.set_title('Price (USD/tCO$_2$)', fontsize=7); ax.legend(fontsize=5.5, frameon=False)
    for a, (lab, col) in enumerate(TYPES):
        if d['wts'][s, a] <= 0:
            continue
        for k_, (tech, ls) in enumerate((('fossil', '-'), ('gas', ':'))):
            axs[1].plot(t, d['fp_q'][s, :, a, k_], color=col, ls=ls, lw=1, label=f'{lab}, {tech} (FP)')
            axs[1].plot(t, d['lp_q'][s, a, :, k_], color=col, ls=ls, lw=1.6, alpha=0.4,
                        label=f'{lab}, {tech} (LP)')
        axs[2].plot(t1, d['fp_E'][s, :, a], color=col, lw=1, label=f'{lab} (FP)')
        axs[2].plot(t1, d['lp_E'][s, a], color=col, ls='--', lw=1, label=f'{lab} (LP)')
        axs[3].plot(t1, d['fp_theta'][s, :, a], color=col, lw=1, label=f'{lab} (FP)')
        axs[3].plot(t1, d['lp_theta'][s, a], color=col, ls='--', lw=1, label=f'{lab} (LP, one timing)')
    axs[1].set_title('Production per agent and day', fontsize=7)
    axs[2].set_title('Cumulative emissions per agent', fontsize=7)
    axs[3].set_title('Holdings per agent', fontsize=7)
    axs[1].legend(fontsize=4.5, frameon=False); axs[3].legend(fontsize=4.5, frameon=False)
    for ax in axs:
        ax.set_xlabel('Day $t$', fontsize=7); ax.tick_params(labelsize=6)
    g = G[s]
    fig.suptitle(f"init_alloc {100 * g[0]:.1f}%, $\\kappa^1$ = {g[1]:.1f}, financial inst. {100 * g[3]:.0f}%  |  "
                 f"{regime[s]}, $a^\\star$ = {a_star[s]:.2f}  |  exact exploitability {100 * d['E_pop'][-1][s]:.2f}%, "
                 f"price error {100 * d['price_err'][-1][s]:.1f}%", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


if 'fp_q' in d.files:
    os.makedirs(fig_dir, exist_ok=True)
    order = np.lexsort((G[:, 1], G[:, 0], G[:, 3]))          # by mm, then alloc, then kappa
    with PdfPages(os.path.join(fig_dir, 'trajectories.pdf')) as pdf:
        for s in order:
            fig = traj_page(s)
            pdf.savefig(fig); plt.close(fig)
    print(f"-> {os.path.join(fig_dir, 'trajectories.pdf')} ({len(order)} scenarios)")
    for mm in mm_vals:                                         # one PNG per regime and mm panel
        for r in ('cap_slack', 'interior', 'fine'):
            idx = np.where((regime == r) & np.isclose(G[:, 3], mm))[0]
            if len(idx):
                s = idx[len(idx) // 2]
                fig = traj_page(s)
                ps.save(fig, os.path.join(fig_dir, f'trajectory_{r}_mm{int(round(100 * mm))}'), check=False)
