"""Major–minor rung 3: price paths and agent heterogeneity at chosen scenarios.

For each scenario gamma, with the FINAL mixture of the run (all saved slots):

1. price paths: N_MARKETS markets (common noise, major component u = s mod k per
   market, minors' stratified components per agent), coloured by the year-end
   aggregate gap E - theta (> 0: market short); mean and 10-90% band;
2. heterogeneity (density maps, as figs_audit/seed*/sampling/heterogeneity/*_kde):
   ONE common-noise draw.
     major  : the same market re-simulated once per component u = 0..k-1 (the
              major's mixed strategy; the market is re-cleared each time), density
              over the k trajectories; the top panel shows every resulting A_t
              faintly and the drawn one (u*) in bold;
     minors : the market with u* gives frozen (A, rho, H); every minor component
              is rolled out with N_AGENTS idiosyncratic draws against it.
   Rows: trading f_t (executed), fuel share eta_t, production xi_t,
   pollution - allowances E_t - theta_t.

    python3 plot_majmin_scenarios.py --run-dir results_majmin/r3_T30_K30_n30_block_seed42
Writes <run-dir>/scenarios/<name>/{price_paths,kde}.{pdf,png} and summary.json.
"""
import os
import json
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
ap.add_argument('--markets', type=int, default=1000)
ap.add_argument('--agents', type=int, default=50, help='per minor component in the KDE')
ap.add_argument('--seed', type=int, default=2024)
ap.add_argument('--scenarios', nargs='+', default=['ref', 'alloc50'])
ap.add_argument('--tag', default='', help='suffix of the output folder, e.g. _bm1')
args = ap.parse_args()
with open(os.path.join(args.run_dir, 'progress.json')) as f:
    prog = json.load(f)
os.environ['KETS_FLOOR_MODE'] = prog['floor_mode']
os.environ['KETS_N_SIM'] = str(prog['n_sim'])
os.environ.setdefault('KETS_FLOOR_MAX', '30')

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from envs.base_params import *
from envs.environment import N_TYPES
from envs.environment_majmin import MajorMinorEnv, MINORS, MAJOR
from envs.models_majmin import make_buffer, buffer_set
from envs.params_v2 import K_MAX, A_CEIL_V2
from utils import norm_gamma, CTX_REF
from utils_majmin import load_net, make_sim, draw_markets
from plot_heterogeneity import (density_panel, type_cmap, _style, F_TITLE, F_LABEL, F_TICK)
from plot_bemfg import TYPE_COLORS, SURFACE

NAMES = ('Private producer (minor)', 'Large public producer (major)',
         'Financial institution (minor)')
SCEN = {'ref': tuple(CTX_REF), 'alloc50': (0.5,) + tuple(CTX_REF[1:])}

T, n = int(prog['T_grid'][0]), int(prog['n_sim'])
env = MajorMinorEnv(kappa=CTX_REF[1], T=T, generate_P_func=generate_prices_ou, A0=A0_BASE,
                    P0=P0, market_impact_func=market_impact_base,
                    generate_eps0_func=supply_noise_base,
                    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
                    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=20).set_T(T)
K = len(prog['records'])
nets = [[load_net(args.run_dir, i, j) for j in range(N_TYPES)] for i in range(K + 1)]
k = len(nets)
bufs, stats = [], []
for j in range(N_TYPES):
    b, st = make_buffer(nets[0][j], K_MAX)
    for i, tr in enumerate(nets):
        b = buffer_set(b, i, tr[j])
    bufs.append(b); stats.append(st)
minor_buf, minor_static = tuple(bufs[j] for j in MINORS), tuple(stats[j] for j in MINORS)
maj_buf, maj_static = bufs[MAJOR], stats[MAJOR]
sim = make_sim(env, minor_static, maj_static)
idx = env.dic_indexes


@jax.jit
def sim_many(gam, P, eps0, imaj, i0, i2, ck, u):
    return jax.vmap(lambda g, p, e, im, a, b, c, uu: sim(
        maj_buf, minor_buf, jnp.int32(k), g, p, e, im, (a, b), c, uu, False))(
        gam, P, eps0, imaj, i0, i2, ck, u)


def minor_traj(net, j, A, rho, H, P, idio, ctx, envg):
    """States (T+1, B, 16) and executed actions (T, B, 3) of type-j agents."""
    B = idio.shape[0]
    X0 = jnp.broadcast_to(envg.type_initial_state(j), (B, 16))

    def step(X, t):
        inp = jax.vmap(envg.net_input, in_axes=(0, None))(X, H[t])
        out = jax.vmap(lambda z: net(z, ctx))(inp)
        a = jax.vmap(envg.act_at, in_axes=(0, 0, None))(X, out, A[t + 1])
        r = rho[t + 1]
        a = a.at[:, 0].set(jnp.where(a[:, 0] >= 0, r[1] * a[:, 0], r[0] * a[:, 0]))
        Xn = jax.vmap(envg.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
            X, a, idio[:, t], A[t + 1], P[t + 1])
        return Xn, (X, a)
    Xf, (Xs, acts) = jax.lax.scan(step, X0, jnp.arange(T))
    return jnp.concatenate([Xs, Xf[None]], axis=0), acts


minor_traj_j = eqx.filter_jit(minor_traj)


def gap_of(X):
    return np.asarray(jnp.sum(X[..., idx['xi']] * X[..., idx['e']], axis=-1)
                      + X[..., idx['eps']] - X[..., idx['theta']])


def run_scenario(name, g):
    out = os.path.join(args.run_dir, 'scenarios', name + args.tag)
    os.makedirs(out, exist_ok=True)
    gj = jnp.asarray(g, jnp.float32)
    envg = env.bind(gj)
    prior = lambda kk, b: jnp.broadcast_to(gj, (b, 4))

    # ── 1. many price paths ─────────────────────────────────────────────────
    S = args.markets
    gam, P, eps0, imaj, imin, ck = draw_markets(env, jax.random.PRNGKey(args.seed), S, T, n, prior)
    u = jnp.arange(S) % k
    res, ch = [], 100
    for a in range(0, S, ch):
        sl = slice(a, a + ch)
        r = sim_many(gam[sl], P[sl], eps0[sl], imaj[sl], imin[0][sl], imin[1][sl], ck[sl], u[sl])
        res.append((np.asarray(r[0]), np.asarray(r[1]), r[4], r[5]))
    A = np.concatenate([x[0] for x in res]); rho = np.concatenate([x[1] for x in res])
    on_floor = np.concatenate([np.asarray(x[2]['on_floor']) for x in res])
    on_ceil = np.concatenate([np.asarray(x[2]['on_ceil']) for x in res])
    gaps = np.concatenate([np.asarray(jax.vmap(lambda xf0, xf2, x0: envg.minor_gaps((xf0, xf2), x0)[0])(
        x[3][0][0], x[3][0][1], x[3][1])) for x in res])
    summ = {'scenario': name, 'gamma': list(g), 'BM_UNIFORM': float(BM_UNIFORM), 'mixture_size': k, 'markets': S,
            'A_mean': float(A[:, 1:].mean()), 'A_final_q': np.quantile(A[:, -1], [.1, .5, .9]).tolist(),
            'A_on_floor': float(on_floor.mean()), 'A_on_ceil': float(on_ceil.mean()),
            'market_short_frac': float((gaps > 0).mean()),
            'rho_sell_mean_on_floor': float(rho[:, 1:, 0][on_floor].mean()) if on_floor.any() else 1.0,
            'sigma_S': float(envg.sigma_S()), 'Theta': float(envg.total_allocation()),
            'major_mass': float(envg.major_mass())}

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor='white')
    lim = np.quantile(np.abs(gaps), 0.95) + 1e-9
    cmap = plt.get_cmap('coolwarm')
    sel = np.random.default_rng(0).choice(S, min(S, 600), replace=False)
    t = np.arange(T + 1)
    for i in sel:
        ax.plot(t, A[i], color=cmap(0.5 + 0.5 * np.clip(gaps[i] / lim, -1, 1)), lw=0.35, alpha=0.6)
    ax.fill_between(t, np.quantile(A, .1, axis=0), np.quantile(A, .9, axis=0), color='k', alpha=0.08, lw=0)
    ax.plot(t, A.mean(axis=0), color='k', lw=2.0, label='mean')
    ax.axhline(float(g[2]), color='0.4', ls=':', lw=1); ax.axhline(A_CEIL_V2, color='0.4', ls='--', lw=1)
    ax.set_xlim(0, T); ax.set_xlabel('Day $t$'); ax.set_ylabel('Cleared price (USD/tCO$_2$)')
    ax.set_title(f"{name}: $\\gamma$ = {tuple(round(x, 2) for x in g)}  —  {S} markets, mixture of {k}",
                 fontsize=10, loc='left')
    sm = ScalarMappable(cmap=cmap, norm=Normalize(-lim, lim))
    fig.colorbar(sm, ax=ax, pad=0.02).set_label('Year-end gap $E-\\theta$ (> 0: market short)')
    ax.legend(frameon=False, loc='upper left')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'price_paths.{ext}'), dpi=200)
    plt.close(fig)

    # ── 2. heterogeneity on ONE common-noise draw ───────────────────────────
    g1, P1, e1, im1, imn1, ck1 = draw_markets(env, jax.random.PRNGKey(args.seed + 1), 1, T, n, prior)
    us = jnp.arange(k)
    rk = sim_many(jnp.repeat(g1, k, 0), jnp.repeat(P1, k, 0), jnp.repeat(e1, k, 0),
                  jnp.repeat(im1, k, 0), jnp.repeat(imn1[0], k, 0), jnp.repeat(imn1[1], k, 0),
                  jnp.repeat(ck1, k, 0), us)
    A_k, rho_k, H_k = np.asarray(rk[0]), np.asarray(rk[1]), np.asarray(rk[2])
    ys_k, (Xf_k, X0f_k) = rk[4], rk[5]
    u_star = int(jax.random.randint(jax.random.PRNGKey(args.seed + 2), (), 0, k))
    d = {'T': T, 'A': A_k[u_star], 'P': np.asarray(P1[0]), 'A_all': A_k}
    # major: k trajectories (one per component), same market noise
    Xmaj = np.concatenate([np.asarray(ys_k['X0']), np.asarray(X0f_k)[:, None]], axis=1)  # (k, T+1, 16)
    amaj = np.asarray(ys_k['a0'])                                                        # (k, T, 3)
    d['f_1'], d['xi_1'], d['eta_1'] = amaj[..., 0].T, amaj[..., 1].T, amaj[..., 2].T
    d['gap_1'] = gap_of(jnp.asarray(Xmaj)).T
    # minors: every component x args.agents against the frozen u* market
    ctx = norm_gamma(gj)
    kI = jax.random.PRNGKey(args.seed + 3)
    for j in MINORS:
        Xs_all, a_all = [], []
        for l in range(k):
            idio = jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
                jax.random.split(jax.random.fold_in(kI, 100 * j + l), args.agents),
                env.agent_params[j, 7], T)
            Xs, acts = minor_traj_j(nets[l][j], j, jnp.asarray(A_k[u_star]), jnp.asarray(rho_k[u_star]),
                                    jnp.asarray(H_k[u_star]), P1[0], idio, ctx, envg)
            Xs_all.append(np.asarray(Xs)); a_all.append(np.asarray(acts))
        Xs = np.concatenate(Xs_all, axis=1); acts = np.concatenate(a_all, axis=1)
        d[f'f_{j}'], d[f'xi_{j}'], d[f'eta_{j}'] = acts[..., 0], acts[..., 1], acts[..., 2]
        d[f'gap_{j}'] = gap_of(jnp.asarray(Xs))
    np.savez_compressed(os.path.join(out, 'heterogeneity.npz'), **d)

    rows = [('f', r'Trading $f_t$ (executed)', np.arange(T)), ('eta', r'Fuel share $\eta_t$', np.arange(T)),
            ('xi', r'Production $\xi_t$', np.arange(T)), ('gap', 'Pollution $-$ Allowances', np.arange(T + 1))]
    fig = plt.figure(figsize=(31.5, 30), facecolor=SURFACE)
    gs = GridSpec(5, 4, figure=fig, height_ratios=[0.8, 1, 1, 1, 1], width_ratios=[1, 1, 1, 0.045],
                  hspace=0.42, wspace=0.22)
    ax = fig.add_subplot(gs[0, :3])
    ts = np.arange(T + 1)
    for uu in range(k):
        ax.plot(ts, A_k[uu], color=TYPE_COLORS[MAJOR], lw=1.0, alpha=0.25)
    ax.plot(ts, d['A'], color='#111111', lw=3.0, label=r'$A_t$ (major on component $u^\ast$)')
    ax.plot([], [], color=TYPE_COLORS[MAJOR], lw=2, alpha=0.5, label=r'$A_t$ for every major component $u$')
    ax.plot(ts, d['P'], color='#8a8983', lw=3.0, label=r'$P_t$')
    ax.axhline(float(g[2]), color='#8a8983', lw=2.0, ls=(0, (6, 4)))
    ax.text(1.0, float(g[2]), 'floor', color='#8a8983', fontsize=F_TICK, ha='right', va='bottom',
            transform=ax.get_yaxis_transform())
    ax.legend(fontsize=F_LABEL - 4, frameon=False, loc='upper left')
    _style(ax, rf'Allowance price $A_t$ and good price $P_t$ — {name}, $\gamma$ = '
               f'{tuple(round(x, 2) for x in g)}', 'Price')
    ax.set_xlim(0, T)
    cgs = gs[1:, 3].subgridspec(N_TYPES, 1, hspace=0.35)
    for j in range(N_TYPES):
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=type_cmap(TYPE_COLORS[j])),
                          cax=fig.add_subplot(cgs[j]))
        cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=F_TICK); cb.outline.set_visible(False)
    for r, (key, title, tt) in enumerate(rows, start=1):
        for j in range(N_TYPES):
            ax = fig.add_subplot(gs[r, j])
            Y = d[f'{key}_{j}']
            if np.abs(Y).max() < 1e-3:
                ax.text(0.5, 0.5, 'no production', ha='center', va='center', fontsize=F_LABEL,
                        color='#8a8983', transform=ax.transAxes)
                ax.set_yticks([]); ax.set_xlim(tt[0], tt[-1])
            else:
                density_panel(ax, tt, Y, TYPE_COLORS[j], 'kde')
                if key == 'gap':
                    ax.axhline(0, color='#8a8983', lw=1.5)
            _style(ax, None, None, 'day' if r == len(rows) else None)
            ax.grid(False)
            if r == 1:
                ax.set_title(NAMES[j] + ('\n(k components, same market)' if j == MAJOR else
                                         f'\n({k} components x {args.agents} agents)'),
                             fontsize=F_TITLE - 4, pad=14)
            if j == 0:
                ax.set_ylabel(title, fontsize=F_LABEL)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'kde.{ext}'), dpi=100, facecolor=SURFACE)
    plt.close(fig)
    summ['heterogeneity'] = {'u_star': u_star, 'A_u_star_mean': float(d['A'][1:].mean()),
                             'A_across_u_final_q': np.quantile(A_k[:, -1], [.1, .5, .9]).tolist()}
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summ, f, indent=2)
    print(json.dumps({kk: v for kk, v in summ.items()}, indent=1), flush=True)
    print(f"-> {out}", flush=True)


for nm in args.scenarios:
    run_scenario(nm, SCEN[nm])
