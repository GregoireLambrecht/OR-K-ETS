"""Heterogeneity across agents on ONE common-noise sample.

    python3 plot_heterogeneity.py --run-dir results_floor30/choice --scenario current_kets

The converged FP mixture (all triples on disk) is bound at the scenario
context. One common-noise draw (P, eps0) and one simulated population fix the
allowance-price path A, so A and P are not random. Every policy of the mixture
is then rolled out for --agents agents per type with independent idiosyncratic
draws against that fixed A, recording the whole trajectory. The dispersion
across (mixture member x idiosyncratic draw) is the heterogeneity of the
equilibrium population.

Rows: allowance price A_t; trading f_t; fossil share eta_t; production xi_t;
pollution minus allowances E_t - theta_t. Columns: the three types.
Writes figs_audit/seed<seed>/sampling/heterogeneity/<scenario>_<study>.{pdf,png}
and caches the arrays next to the audit cells as heterogeneity.npz.
"""

import os
import sys
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _floor_max_of_run(argv):
    run_dir = None
    for i, a in enumerate(argv):
        if a == '--run-dir' and i + 1 < len(argv):
            run_dir = argv[i + 1]
        elif a.startswith('--run-dir='):
            run_dir = a.split('=', 1)[1]
    if run_dir is None:
        return None
    path = os.path.join(run_dir, 'progress.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        box = json.load(f).get('context_box')
    return float(box[2][1]) if box else None


_RUN_FLOOR_MAX = _floor_max_of_run(sys.argv)
if _RUN_FLOOR_MAX is not None:
    os.environ['KETS_FLOOR_MAX'] = repr(_RUN_FLOOR_MAX)

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import stack_policies
from utils import (draw_common_noise, draw_idio, load_triple_r3, bind_env,
                   bind_policy, make_price_fn_r3, CONTEXT_DIM, _G_HI)
from audit_pmfg import audit_scenarios
from plot_bemfg import TYPE_COLORS, TYPE_NAMES, SURFACE, DPI

HET_SEED = 777
F_TITLE, F_LABEL, F_TICK = 30, 26, 22
Q_OUTER, Q_INNER = (0.05, 0.95), (0.25, 0.75)
N_SAMPLE_PATHS = 6


def make_traj_fn(env_c, j):
    """Full trajectory of B type-j agents under one policy, against fixed A, P.
    -> states (T+1, B, 16), actions (T, B, 3)."""
    idx = env_c.dic_indexes
    T_int = int(env_c.T)

    @eqx.filter_jit
    def fn(policy, A_traj, P_traj, idio):
        B = idio.shape[0]
        X0 = jnp.broadcast_to(env_c.type_initial_state(j), (B, 16))

        def step(X_t, t):
            a_hat = jax.vmap(policy)(jax.vmap(env_c.normalize_state)(X_t))
            a_t = jax.vmap(env_c.unnormalize_action)(X_t, a_hat)
            X_n = jax.vmap(env_c.single_step_dynamics,
                           in_axes=(0, 0, 0, None, None))(
                X_t, a_t, idio[:, t], A_traj[t + 1], P_traj[t + 1])
            return X_n, (X_t, a_t)

        X_f, (Xs, As) = jax.lax.scan(step, X0, jnp.arange(T_int))
        Xs = jnp.concatenate([Xs, X_f[None]], axis=0)
        return Xs, As
    return fn


def simulate(args):
    with open(os.path.join(args.run_dir, 'progress.json')) as f:
        prog = json.load(f)
    n_avail = 0
    while os.path.exists(os.path.join(args.run_dir,
                                      f'policy_{n_avail}_type0.eqx')):
        n_avail += 1
    box = prog.get('context_box')
    floor_max = float(box[2][1]) if box else FLOOR_MAX
    if abs(float(_G_HI[2]) - floor_max) > 1e-6:
        raise SystemExit("FiLM normaliser box top != run box top")
    c = audit_scenarios(floor_max)[args.scenario]
    T = int(prog['T_grid'][-1])
    mode = prog.get('mode', 'choice')
    agents = int(prog.get('agent_per_policy', 20))

    env = ExogenousMarketEnvJAX(
        kappa=GAMMA_REF[1], T=T,
        generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
        market_impact_func=market_impact_base,
        generate_eps0_func=white_noise_A_base,
        generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
        A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=agents)
    if mode == 'bau':
        bp, bq = list(PRIVATE_GENERATOR), list(BIG_PUBLIC_GENERATOR)
        bp[8], bq[8] = TEC_PRIVATE, TEC_PUBLIC
        env = eqx.tree_at(lambda e: e.agent_params, env,
                          jnp.array([bp, bq, MARKET_MAKER]))
    ca = jnp.asarray(c, dtype=jnp.float32)
    env_c = bind_env(env, ca)

    print(f"HETEROGENEITY | {args.run_dir} | {args.scenario} {tuple(c)} | "
          f"mixture = {n_avail} triples, T = {T}", flush=True)

    # the mixture and ONE common-noise sample -> one price path
    history = [load_triple_r3(args.run_dir, i) for i in range(n_avail)]
    statics = tuple(stack_policies([history[0][j]])[1] for j in range(N_TYPES))
    stacked = tuple(stack_policies([h[j] for h in history])[0]
                    for j in range(N_TYPES))
    kc, kp, ka = jax.random.split(jax.random.PRNGKey(HET_SEED), 3)
    P, eps0 = draw_common_noise(env_c, T, 1, kc)
    idio_pop = draw_idio(env_c, T, env_c.type_counts, kp)
    ctx_rows = jnp.broadcast_to(ca, (1, CONTEXT_DIM))
    A = make_price_fn_r3(env, statics)(stacked, ctx_rows, P, eps0, idio_pop)[0]
    P = P[0]
    print(f"price path fixed: A in [{float(A.min()):.2f}, {float(A.max()):.2f}], "
          f"floor {float(c[2]):.2f}", flush=True)

    # every mixture member x --agents idiosyncratic draws per type
    idx = env_c.dic_indexes
    out = {'A': np.asarray(A), 'P': np.asarray(P), 'T': T,
           'context': np.asarray(c), 'n_policies': n_avail,
           'skip_first': args.skip_first}
    # The price path above uses the FULL mixture (that is the equilibrium);
    # the agent-level distributions drop the first --skip-first members, the
    # random-init policy and its immediate successors, which carry 1/K weight
    # each but sit far from every later best response.
    members = list(range(args.skip_first, n_avail))
    print(f"agent rollouts over mixture members {members[0]}..{members[-1]} "
          f"({len(members)} of {n_avail})", flush=True)
    for j in range(N_TYPES):
        fn = make_traj_fn(env_c, j)
        keys = jax.random.split(jax.random.fold_in(ka, j), n_avail)
        f_all, eta_all, xi_all, gap_all = [], [], [], []
        for k in members:
            idio = draw_idio(env_c, T, (args.agents,) * N_TYPES, keys[k])[j]
            pol = bind_policy(history[k][j], ca)
            Xs, As = fn(pol, A, P, idio)
            Xs, As = np.asarray(Xs), np.asarray(As)
            f_all.append(As[:, :, 0])
            xi_all.append(As[:, :, 1])
            eta_all.append(As[:, :, 2])
            E = (Xs[:, :, idx['ef']] * Xs[:, :, idx['xi_f']]
                 + Xs[:, :, idx['eg']] * Xs[:, :, idx['xi_g']]
                 + Xs[:, :, idx['eps']])
            gap_all.append(E - Xs[:, :, idx['theta']])
        out[f'f_{j}'] = np.concatenate(f_all, axis=1)        # (T, n)
        out[f'xi_{j}'] = np.concatenate(xi_all, axis=1)
        out[f'eta_{j}'] = np.concatenate(eta_all, axis=1)
        out[f'gap_{j}'] = np.concatenate(gap_all, axis=1)    # (T+1, n)
        print(f"  type {j}: {out[f'f_{j}'].shape[1]} agent paths", flush=True)
    return out


# ── figure ────────────────────────────────────────────────────────────────────

def _style(ax, title=None, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e6e5e1", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=F_TICK, length=6, width=1.2)
    if title:
        ax.set_title(title, fontsize=F_TITLE, pad=12, loc='left')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=F_LABEL)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=F_LABEL)


def fan(ax, t, Y, color, rng):
    """Y (len(t), n): 5-95 and 25-75 quantile bands, median, a few paths."""
    lo, hi = np.quantile(Y, Q_OUTER, axis=1)
    ax.fill_between(t, lo, hi, color=color, alpha=0.16, lw=0)
    lo, hi = np.quantile(Y, Q_INNER, axis=1)
    ax.fill_between(t, lo, hi, color=color, alpha=0.30, lw=0)
    for i in rng.choice(Y.shape[1], min(N_SAMPLE_PATHS, Y.shape[1]),
                        replace=False):
        ax.plot(t, Y[:, i], color=color, lw=0.7, alpha=0.55)
    ax.plot(t, np.median(Y, axis=1), color=color, lw=3.0)


def figure(d, out_base):
    T = int(d['T'])
    t_act, t_st = np.arange(T), np.arange(T + 1)
    rng = np.random.default_rng(0)
    rows = [('f', r'Trading $f_t$', t_act),
            ('eta', r'Fuel share $\eta_t$', t_act),
            ('xi', r'Production $\xi_t$', t_act),
            ('gap', 'Pollution $-$ Allowances', t_st)]

    fig = plt.figure(figsize=(30, 30), facecolor=SURFACE)
    gs = GridSpec(5, 3, figure=fig, height_ratios=[0.8, 1, 1, 1, 1],
                  hspace=0.42, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t_st, d['A'], color="#111111", lw=3.0, label=r'$A_t$')
    ax.plot(t_st, d['P'], color="#8a8983", lw=3.0, label=r'$P_t$')
    ax.legend(fontsize=F_LABEL, frameon=False, loc='upper left')
    ax.axhline(float(d['context'][2]), color="#8a8983", lw=2.0,
               ls=(0, (6, 4)))
    ax.text(1.0, float(d['context'][2]), 'floor', color="#8a8983",
            fontsize=F_TICK, ha='right', va='bottom',
            transform=ax.get_yaxis_transform())
    _style(ax, r'Allowance price $A_t$ and good price $P_t$', 'Price')

    for r, (key, title, t) in enumerate(rows, start=1):
        for j in range(N_TYPES):
            ax = fig.add_subplot(gs[r, j])
            Y = d[f'{key}_{j}']
            degenerate = np.abs(Y).max() < 1e-3
            if degenerate:
                ax.text(0.5, 0.5, 'no production', ha='center', va='center',
                        fontsize=F_LABEL, color="#8a8983",
                        transform=ax.transAxes)
                ax.set_yticks([])
                ax.set_xlim(t[0], t[-1])
            else:
                fan(ax, t, Y, TYPE_COLORS[j], rng)
                if key == 'gap':
                    ax.axhline(0, color="#8a8983", lw=1.5)
            _style(ax, None, None, 'day' if r == len(rows) else None)
            if r == 1:
                ax.set_title(TYPE_NAMES[j], fontsize=F_TITLE, pad=14,
                             loc='center')
            if j == 0:
                ax.set_ylabel(title, fontsize=F_LABEL, labelpad=14)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + '.pdf', facecolor=SURFACE, bbox_inches='tight')
    fig.savefig(out_base + '.png', facecolor=SURFACE, bbox_inches='tight',
                dpi=110)
    plt.close(fig)
    print(f"-> {out_base}.pdf (+ .png)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--scenario', default='current_kets')
    ap.add_argument('--skip-first', type=int, default=3,
                    help='mixture members left out of the agent rollouts '
                         '(the price path always uses the full mixture)')
    ap.add_argument('--agents', type=int, default=64,
                    help='idiosyncratic draws per mixture member and type')
    ap.add_argument('--seed-label', default='seed42')
    ap.add_argument('--replot', action='store_true',
                    help='reuse the cached heterogeneity.npz')
    ap.add_argument('--style', default='fan', choices=['fan', 'kde', 'hist', 'all'],
                    help='quantile fans, per-day KDE or histogram maps, or all')
    args = ap.parse_args()

    cache = os.path.join(args.run_dir, 'audit', args.scenario,
                         f'heterogeneity_skip{args.skip_first}.npz')
    if args.replot and os.path.exists(cache):
        d = dict(np.load(cache))
    else:
        d = simulate(args)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **d)
        print(f"cached -> {cache}")
    # the study is the results_<name> component of the run path, whatever the
    # nesting below it (results_floor30/choice or results_floor30/choice_seed3/choice)
    study = next((p for p in args.run_dir.split('/') if p.startswith('results_')),
                 'run').replace('results_', '')
    out = os.path.join('figs_audit', args.seed_label, 'sampling',
                       'heterogeneity', args.scenario,
                       f'{args.scenario}_{study}')
    if args.style in ('fan', 'all'):
        figure(d, out)
    if args.style in ('kde', 'all'):
        figure_kde(d, out + '_kde')
    if args.style in ('hist', 'all'):
        figure_kde(d, out + '_hist', method='hist')




# ── density-map style ─────────────────────────────────────────────────────────
from matplotlib.colors import LinearSegmentedColormap

N_GRID = 220          # value-axis resolution of the density map
BW_MIN_FRAC = 0.015   # bandwidth floor, as a fraction of the panel's value range


def kde_over_time(Y, grid):
    """Y (T, n) -> density (T, G): a Gaussian KDE per time step.

    Bandwidth is Scott's rule per time step, floored at BW_MIN_FRAC of the grid
    range so a bang-bang or constant distribution (zero variance) still draws
    as a narrow ridge instead of a division by zero. Each row is normalised to
    its maximum, so colour = density relative to that day's mode."""
    T, n = Y.shape
    span = grid[-1] - grid[0]
    out = np.empty((T, grid.size))
    for t in range(T):
        v = Y[t]
        h = max(1.06 * v.std() * n ** (-0.2), BW_MIN_FRAC * span)
        z = (grid[:, None] - v[None, :]) / h
        dens = np.exp(-0.5 * z * z).sum(axis=1)
        out[t] = dens / dens.max()
    return out


N_BINS = 60           # value-axis bins of the histogram map


def hist_over_time(Y, edges):
    """Y (T, n) -> (T, B): a histogram per time step on fixed bins, each row
    normalised to its maximum so colour = share of agents relative to that
    day's fullest bin."""
    out = np.empty((Y.shape[0], edges.size - 1))
    for t in range(Y.shape[0]):
        h, _ = np.histogram(Y[t], bins=edges)
        out[t] = h / max(h.max(), 1)
    return out


def type_cmap(color):
    return LinearSegmentedColormap.from_list('type', [SURFACE, color], N=256)


def density_panel(ax, t, Y, color, method='kde'):
    lo, hi = Y.min(), Y.max()
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    if method == 'kde':
        grid = np.linspace(lo - pad, hi + pad, N_GRID)
        D = kde_over_time(Y, grid)
        ax.pcolormesh(t, grid, D.T, cmap=type_cmap(color), vmin=0, vmax=1,
                      shading='nearest', rasterized=True)
        ax.set_ylim(grid[0], grid[-1])
    else:
        edges = np.linspace(lo - pad, hi + pad, N_BINS + 1)
        D = hist_over_time(Y, edges)
        t_edges = np.concatenate([t - 0.5, [t[-1] + 0.5]])
        ax.pcolormesh(t_edges, edges, D.T, cmap=type_cmap(color), vmin=0,
                      vmax=1, shading='flat', rasterized=True)
        ax.set_ylim(edges[0], edges[-1])
    ax.set_xlim(t[0], t[-1])


def figure_kde(d, out_base, method='kde'):
    T = int(d['T'])
    t_act, t_st = np.arange(T), np.arange(T + 1)
    rows = [('f', r'Trading $f_t$', t_act),
            ('eta', r'Fuel share $\eta_t$', t_act),
            ('xi', r'Production $\xi_t$', t_act),
            ('gap', 'Pollution $-$ Allowances', t_st)]

    fig = plt.figure(figsize=(31.5, 30), facecolor=SURFACE)
    gs = GridSpec(5, 4, figure=fig, height_ratios=[0.8, 1, 1, 1, 1],
                  width_ratios=[1, 1, 1, 0.045], hspace=0.42, wspace=0.22)

    ax = fig.add_subplot(gs[0, :3])
    ax.plot(t_st, d['A'], color="#111111", lw=3.0, label=r'$A_t$')
    ax.plot(t_st, d['P'], color="#8a8983", lw=3.0, label=r'$P_t$')
    ax.legend(fontsize=F_LABEL, frameon=False, loc='upper left')
    ax.axhline(float(d['context'][2]), color="#8a8983", lw=2.0,
               ls=(0, (6, 4)))
    ax.text(1.0, float(d['context'][2]), 'floor', color="#8a8983",
            fontsize=F_TICK, ha='right', va='bottom',
            transform=ax.get_yaxis_transform())
    _style(ax, r'Allowance price $A_t$ and good price $P_t$', 'Price')
    ax.set_xlim(0, T)

    # one colour bar per type, stacked at the right of the density panels
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cgs = gs[1:, 3].subgridspec(N_TYPES, 1, hspace=0.35)
    for j in range(N_TYPES):
        cax = fig.add_subplot(cgs[j])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1),
                                         cmap=type_cmap(TYPE_COLORS[j])),
                          cax=cax)
        cb.set_ticks([0, 0.5, 1])
        cb.ax.tick_params(labelsize=F_TICK, length=6, width=1.2)
        cb.outline.set_visible(False)

    for r, (key, title, t) in enumerate(rows, start=1):
        for j in range(N_TYPES):
            ax = fig.add_subplot(gs[r, j])
            Y = d[f'{key}_{j}']
            if np.abs(Y).max() < 1e-3:
                ax.text(0.5, 0.5, 'no production', ha='center', va='center',
                        fontsize=F_LABEL, color="#8a8983",
                        transform=ax.transAxes)
                ax.set_yticks([])
                ax.set_xlim(t[0], t[-1])
            else:
                density_panel(ax, t, Y, TYPE_COLORS[j], method)
                if key == 'gap':
                    ax.axhline(0, color="#8a8983", lw=1.5)
            _style(ax, None, None, 'day' if r == len(rows) else None)
            ax.grid(False)
            if r == 1:
                ax.set_title(TYPE_NAMES[j], fontsize=F_TITLE, pad=14,
                             loc='center')
            if j == 0:
                ax.set_ylabel(title, fontsize=F_LABEL, labelpad=14)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + '.pdf', facecolor=SURFACE, bbox_inches='tight')
    fig.savefig(out_base + '.png', facecolor=SURFACE, bbox_inches='tight',
                dpi=110)
    plt.close(fig)
    print(f"-> {out_base}.pdf (+ .png)")


if __name__ == '__main__':
    main()
