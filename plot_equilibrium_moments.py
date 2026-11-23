"""Do the seeds select the same equilibrium? Moments of the conditional law.

    python3 plot_equilibrium_moments.py --study results_floor30 [--nm 64 --agents 8]

One fixed sample of Nm rows (gamma, eps0) — gamma from the training prior,
eps0 the common noise — shared by every seed. For each seed and each row, the
mixture (members --skip-first..K) is rolled out with --agents idiosyncratic
draws per member and type, giving at every t an empirical distribution
sum delta_{X_t} across agents. Moment 1 = mean, moment 2 = E[X^2], averaged
over the Nm rows: two matrices [rows x T] per seed, rows =

    A_t                        the allowance price (common, mixture-dependent)
    per type: theta, xi_f, xi_g (states), f, xi, eta (actions)
    (financial institution: theta and f only)

Normalisation: entry-wise mean over the seeds, M_bar; each seed's matrix is
divided by M_bar entry by entry, so 1 = agreement with the consensus. Entries
whose consensus is ~0 (e.g. xi_g when eta = 1) are masked.

Writes figs_audit/all_seeds/equilibrium/moments_<study>.{pdf,png} and caches
the raw moments in <study>/equilibrium_moments.npz.
"""

import os
import sys
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _floor_max_of_study(argv):
    study = 'results_floor30'
    for i, a in enumerate(argv):
        if a == '--study' and i + 1 < len(argv):
            study = argv[i + 1]
    p = os.path.join(study, 'choice', 'progress.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        box = json.load(f).get('context_box')
    return float(box[2][1]) if box else None


_FM = _floor_max_of_study(sys.argv)
if _FM is not None:
    os.environ['KETS_FLOOR_MAX'] = repr(_FM)

import glob
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import stack_policies
from utils import (chunked, draw_common_noise, draw_idio, load_triple_r3,
                   bind_env, bind_policy, make_price_fn_r3, CONTEXT_DIM, _G_HI)
from plot_bemfg import find_runs, SURFACE, DPI

MOM_SEED = 31337
SEED_NAME = {42: 5}          # the original run is shown as seed 5
PARAMS_BY_TYPE = (PRIVATE_GENERATOR, BIG_PUBLIC_GENERATOR, MARKET_MAKER)
F_TITLE, F_LABEL, F_TICK = 26, 22, 18
TYPE_SHORT = ('Priv.', 'Pub.', 'FI')
STATE_KEYS = ('theta', 'xi_f', 'xi_g')
ACTION_KEYS = ('f', 'xi', 'eta')
ROW_LABEL = {'A': r'$A_t$', 'theta': r'$\theta_t$', 'xi_f': r'$\bar\xi^f_t$',
             'xi_g': r'$\bar\xi^g_t$', 'f': r'$f_t$', 'xi': r'$\xi_t$',
             'eta': r'$\eta_t$'}


def _cap_of(label):
    """Bound of an action row: cap_f / cap_xi of the type, 1 for eta; NaN
    for the price and the states (which keep the RMS row scale)."""
    if ':' not in label:
        return np.nan
    j, k = label.split(':')
    p = np.asarray(PARAMS_BY_TYPE[int(j)])
    return {'f': float(p[5]), 'xi': float(p[6]), 'eta': 1.0}.get(k, np.nan)


def _states_size(M1, labels, all_labels):
    """Typical size of each state row at each day, from the consensus FIRST
    moment M1 (all rows, T): A_t and theta_t by their own mean (theta floored
    at 10% of its RMS: the market maker starts at 0), the cumulative
    productions by the mean of their sum."""
    size = np.empty((len(labels), M1.shape[1]))
    for r_i, l in enumerate(labels):
        if l == 'A':
            size[r_i] = M1[all_labels.index('A')]
        else:
            j, k = l.split(':')
            if k == 'theta':
                row = M1[all_labels.index(l)]
                size[r_i] = np.maximum(row, 0.1 * np.sqrt((row ** 2).mean()))
            else:
                size[r_i] = (M1[all_labels.index(f'{j}:xi_f')]
                             + M1[all_labels.index(f'{j}:xi_g')])
    return size


def _states_dev(M, M1_all, labels, all_labels, mom):
    """Normalised deviation of each seed from the consensus: (M - mean_seeds M)
    / size^p with p = 1 (moment 1) or 2 (moment 2). M: (seeds, rows, T);
    M1_all: (seeds, all rows, T) first moments. -> R (seeds, rows, T)."""
    Mbar = M.mean(axis=0)
    rs = _states_size(M1_all.mean(axis=0), labels, all_labels)
    rs = rs ** (2 if mom == 'm2' else 1)
    tiny = np.abs(rs) < 1e-9 * np.nanmax(np.abs(rs))
    R = (M - Mbar) / np.where(tiny, 1.0, rs)
    return np.where(np.broadcast_to(tiny, R.shape), np.nan, R)


def rows_for_type(j):
    if j == N_TYPES - 1:                      # financial institution
        return ('theta', 'f')
    return STATE_KEYS + ACTION_KEYS


def make_traj_fn(env_T, j):
    """Trajectories of M type-j agents under one policy, per (context, noise)
    row -> states (R, T+1, M, 16), actions (R, T, M, 3)."""
    T_int = int(env_T.T)

    @eqx.filter_jit
    def fn(policy, gam, A, P, idio):
        def one(g, a, p):
            env_c = bind_env(env_T, g)
            pol = bind_policy(policy, g)
            M = idio.shape[0]
            X0 = jnp.broadcast_to(env_c.type_initial_state(j), (M, 16))

            def step(X_t, t):
                a_hat = jax.vmap(pol)(jax.vmap(env_c.normalize_state)(X_t))
                a_t = jax.vmap(env_c.unnormalize_action)(X_t, a_hat)
                X_n = jax.vmap(env_c.single_step_dynamics,
                               in_axes=(0, 0, 0, None, None))(
                    X_t, a_t, idio[:, t], a[t + 1], p[t + 1])
                return X_n, (X_t, a_t)

            X_f, (Xs, As) = jax.lax.scan(step, X0, jnp.arange(T_int))
            return jnp.concatenate([Xs, X_f[None]], axis=0), As
        return jax.vmap(one)(gam, A, P)
    return fn


def simulate(args, runs):
    with open(os.path.join(runs[0], 'progress.json')) as f:
        prog = json.load(f)
    box = prog.get('context_box')
    floor_max = float(box[2][1]) if box else FLOOR_MAX
    if abs(float(_G_HI[2]) - floor_max) > 1e-6:
        raise SystemExit("FiLM normaliser box top != run box top")
    T = int(prog['T_grid'][-1])
    agents = int(prog.get('agent_per_policy', 20))
    env = ExogenousMarketEnvJAX(
        kappa=GAMMA_REF[1], T=T,
        generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
        market_impact_func=market_impact_base,
        generate_eps0_func=white_noise_A_base,
        generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
        A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=agents)
    idx = env.dic_indexes

    # ONE sample (gamma, eps0) of size Nm, fixed for the whole study
    kg, kc, kp, ki = jax.random.split(jax.random.PRNGKey(MOM_SEED), 4)
    gam = env.generate_context(kg, args.nm)
    P, eps0 = draw_common_noise(env, T, args.nm, kc)
    idio_pop = draw_idio(env, T, env.type_counts, kp)
    idio = draw_idio(env, T, (args.agents,) * N_TYPES, ki)
    print(f"MOMENTS | {len(runs)} seeds | Nm = {args.nm} (gamma, eps0) rows "
          f"from the prior, {args.agents} agents per member and type, T = {T}",
          flush=True)

    labels = ['A'] + [f'{j}:{k}' for j in range(N_TYPES) for k in rows_for_type(j)]
    out = {'labels': np.array(labels), 'T': T, 'nm': args.nm}
    traj_fns = [make_traj_fn(env, j) for j in range(N_TYPES)]

    for run in runs:
        n_avail = 0
        while os.path.exists(os.path.join(run, f'policy_{n_avail}_type0.eqx')):
            n_avail += 1
        history = [load_triple_r3(run, i) for i in range(n_avail)]
        statics = tuple(stack_policies([history[0][j]])[1] for j in range(N_TYPES))
        stacked = tuple(stack_policies([h[j] for h in history])[0]
                        for j in range(N_TYPES))
        members = list(range(args.skip_first, n_avail))
        A = chunked(lambda a, b: make_price_fn_r3(env, statics)(
            stacked, gam[a:b], P[a:b], eps0[a:b], idio_pop), args.nm, args.chunk)
        A = jnp.asarray(A)                                     # (Nm, T+1)

        # per-row moments (mean over the agents of that row only), so the raw
        # seed x common-noise data is kept and never has to be regenerated
        Nm = args.nm
        m1 = {'A': np.asarray(A[:, :T], np.float32)}                   # (Nm, T)
        m2 = {'A': np.asarray(A[:, :T] ** 2, np.float32)}
        for j in range(N_TYPES):
            keys = rows_for_type(j)
            s1 = {k: np.zeros((Nm, T), np.float64) for k in keys}
            s2 = {k: np.zeros((Nm, T), np.float64) for k in keys}
            for k_m in members:
                Xs, As = _chunked_pair(traj_fns[j], history[k_m][j], gam, A, P,
                                       idio[j], Nm, args.chunk)
                for k in keys:
                    if k in STATE_KEYS:
                        v = Xs[:, :T, :, idx[k]]               # (Nm, T, M)
                    else:
                        v = As[:, :, :, ACTION_KEYS.index(k)]  # (Nm, T, M)
                    s1[k] += np.asarray(v.mean(axis=2))
                    s2[k] += np.asarray((v ** 2).mean(axis=2))
            for k in keys:
                m1[f'{j}:{k}'] = (s1[k] / len(members)).astype(np.float32)
                m2[f'{j}:{k}'] = (s2[k] / len(members)).astype(np.float32)
        seed = int(run.split('choice_seed')[1].split('/')[0]) \
            if 'choice_seed' in run else 42
        raw1 = np.stack([m1[l] for l in labels])                        # (rows, Nm, T)
        raw2 = np.stack([m2[l] for l in labels])
        out[f'm1_seed{seed}_rows'] = raw1
        out[f'm2_seed{seed}_rows'] = raw2
        out[f'm1_seed{seed}'] = raw1.mean(axis=1)                       # (rows, T)
        out[f'm2_seed{seed}'] = raw2.mean(axis=1)
        print(f"  seed {seed:<3d} done ({len(members)} members)", flush=True)
    return out


def _chunked_pair(fn, policy, gam, A, P, idio, n, chunk):
    Xs, As = [], []
    for i in range(0, n, chunk):
        x, a = fn(policy, gam[i:i + chunk], A[i:i + chunk], P[i:i + chunk], idio)
        Xs.append(x); As.append(a)
    return jnp.concatenate(Xs), jnp.concatenate(As)


def figure(d, out_base, mode='ratio', view='all'):
    all_labels = list(d['labels'])
    # actions view: A_t plus f, xi, eta per type; states view: A_t, theta and
    # the cumulative productions xi_f, xi_g per type
    wanted = {'all': None, 'actions': ACTION_KEYS, 'states': STATE_KEYS}[view]
    keep = [i for i, l in enumerate(all_labels)
            if wanted is None or l == 'A' or l.split(':')[-1] in wanted]
    labels = [all_labels[i] for i in keep]
    m1_all = {mom: np.stack([d[f'{mom}_seed{s}'] for s in
              sorted(int(k.split('seed')[-1]) for k in d if k.startswith('m1_') and not k.endswith('_rows'))])
              for mom in ('m1',)}
    seeds = sorted(int(k.split('seed')[-1]) for k in d if k.startswith('m1_') and not k.endswith('_rows'))
    T = int(d['T'])
    rows = []
    for l in labels:
        if l == 'A':
            rows.append(ROW_LABEL['A'])
        else:
            j, k = l.split(':')
            rows.append(f"{TYPE_SHORT[int(j)]} {ROW_LABEL[k]}")

    fig, axes = plt.subplots(2, len(seeds), figsize=(6.2 * len(seeds), 13 if len(rows) > 8 else 8),
                             facecolor=SURFACE, sharex=True, sharey=True)
    if mode == 'ratio':
        norm = TwoSlopeNorm(vcenter=1.0, vmin=0.5, vmax=1.5)
        ticks, cb_label = [0.5, 0.75, 1.0, 1.25, 1.5], 'seed moment / mean over seeds'
    elif mode == 'states':
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-10.0, vmax=10.0)          # in %
        ticks, cb_label = [-10, -5, 0, 5, 10], \
            '(seed moment $-$ mean over seeds) / typical size'
    elif mode == 'cap':
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.1, vmax=0.1)
        ticks, cb_label = [-0.1, -0.05, 0.0, 0.05, 0.1], \
            '(seed moment $-$ mean over seeds) / bound'
    else:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.25, vmax=0.25)
        ticks, cb_label = [-0.25, -0.125, 0.0, 0.125, 0.25], \
            '(seed moment $-$ mean over seeds) / row scale'
    last = None
    stats = {}
    for r, mom in enumerate(('m1', 'm2')):
        M = np.stack([d[f'{mom}_seed{s}'][keep] for s in seeds])    # (5, rows, T)
        Mbar = M.mean(axis=0)
        if mode == 'ratio':
            scale = np.abs(M).max(axis=(0, 2), keepdims=True)[0]   # per row
            mask = np.abs(Mbar) < 1e-3 * scale                      # consensus ~ 0
            R = np.where(mask, np.nan, M / np.where(mask, 1.0, Mbar))
            D = np.abs(R - 1.0)
        else:
            # row scale = RMS over time of the consensus: cannot vanish where the
            # consensus crosses zero (mean trade) or sits near zero (green output)
            rs = np.sqrt((Mbar ** 2).mean(axis=1, keepdims=True))
            if mode == 'cap':
                # actions are bounded: f by cap_f, xi by cap_xi (per type), eta
                # by 1 -> normalise by the bound (squared for moment 2); A_t and
                # the states keep the RMS row scale
                caps = np.array([[_cap_of(l)] for l in labels])
                rs = np.where(np.isnan(caps), rs, caps ** (2 if mom == 'm2' else 1))
            if mode == 'states':
                # entry-wise typical size of each state at time t, from the
                # consensus FIRST moment: theta by its own mean, the cumulative
                # productions xi_f and xi_g by the mean of their SUM (so a tiny
                # green share is not compared to itself), A_t by its RMS over
                # time; squared for moment 2
                M1 = m1_all['m1'].mean(axis=0)                       # (all rows, T)
                size = np.empty_like(Mbar)
                for r_i, l in enumerate(labels):
                    if l == 'A':
                        size[r_i] = M1[all_labels.index('A')]      # entry-wise, like theta
                    else:
                        j, k = l.split(':')
                        if k == 'theta':
                            # floored at 10% of its RMS over time: the market
                            # maker starts at theta = 0 exactly
                            row = M1[all_labels.index(l)]
                            size[r_i] = np.maximum(row, 0.1 * np.sqrt((row ** 2).mean()))
                        else:
                            size[r_i] = (M1[all_labels.index(f'{j}:xi_f')]
                                         + M1[all_labels.index(f'{j}:xi_g')])
                rs = size ** (2 if mom == 'm2' else 1)
            if mom == 'm1':
                print("      row scales (RMS over time of the consensus, moment 1):")
                for l, v in zip(labels, rs[:, 0]):
                    print(f"        {l:<10s} {v:12.4g}")
            tiny = np.abs(rs) < 1e-9 * np.nanmax(np.abs(rs))
            mask = np.broadcast_to(tiny, Mbar.shape)
            R = (M - Mbar) / np.where(tiny, 1.0, rs)
            R = np.where(mask, np.nan, R)
            D = np.abs(R)
        stats[mom] = np.nanmean(D, axis=2)                          # (seeds, rows)
        worst = np.nanmax(D, axis=(0, 2))                            # per row
        print(f"{mom} [{mode}]: max deviation over seeds/rows/t = "
              f"{np.nanmax(D):.3f}; masked entries {mask.mean():.1%}")
        for l, w in zip(labels, worst):
            print(f"      {l:<10s} max dev {w:.3f}")
        for c, s in enumerate(seeds):
            ax = axes[r, c]
            ax.set_facecolor('#d9d9d4')                              # masked = grey
            last = ax.pcolormesh(np.arange(T), np.arange(len(rows)),
                                 R[c] * (100.0 if mode == 'states' else 1.0),
                                 cmap='RdBu_r', norm=norm, shading='nearest',
                                 rasterized=True)
            ax.set_yticks(np.arange(len(rows)))
            ax.set_yticklabels(rows, fontsize=F_TICK)
            ax.tick_params(labelsize=F_TICK, length=4)
            if r == 0 and c == 0:
                ax.invert_yaxis()          # shared axis: invert once, A_t on top
            if r == 0:
                ax.set_title(f'seed {SEED_NAME.get(s, s)}', fontsize=F_TITLE, pad=12)
            if r == 1:
                ax.set_xlabel('day', fontsize=F_LABEL)
            if c == 0:
                ax.set_ylabel('moment 1' if mom == 'm1' else 'moment 2',
                              fontsize=F_LABEL, labelpad=16)
    fig.subplots_adjust(left=0.07, right=0.92, top=0.94, bottom=0.07,
                        wspace=0.08, hspace=0.10)
    cax = fig.add_axes([0.935, 0.15, 0.012, 0.7])
    cb = fig.colorbar(last, cax=cax)
    cb.set_ticks(ticks)
    if mode == 'states':
        cb.set_ticklabels([f'{t:+d}%' if t else '0%' for t in ticks])
    cb.ax.tick_params(labelsize=F_TICK)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + '.pdf', facecolor=SURFACE, bbox_inches='tight')
    fig.savefig(out_base + '.png', facecolor=SURFACE, bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print(f"-> {out_base}.pdf (+ .png)")
    _table(stats, labels, rows, seeds, out_base + '_table')
    if mode == 'states' and all(f'm1_seed{s}_rows' in d for s in seeds):
        ci = bootstrap_ci(d, labels, all_labels, keep, seeds)
        _table(stats, labels, rows, seeds, out_base + '_table_ci', ci)
        _table_pm(stats, labels, rows, seeds, out_base + '_table_pm', ci)


def _cell_grid(ax, S, rows, seeds, cmap, vmax, title, ylabels, fmt='{:.1f}'):
    ax.imshow(S, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    for i in range(S.shape[0]):
        for c in range(S.shape[1]):
            v = S[i, c]
            ax.text(c, i, fmt.format(v), ha='center', va='center',
                    fontsize=F_TICK, color='white' if v > 0.6 * vmax else '#111111')
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([f'seed {SEED_NAME.get(s, s)}' for s in seeds],
                       fontsize=F_TICK)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows if ylabels else [''] * len(rows), fontsize=F_TICK)
    if title:
        ax.set_title(title, fontsize=F_TITLE, pad=12)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def _table(stats, labels, rows, seeds, out_base, ci=None):
    """Summary: mean over time of the absolute deviation, in %, one cell per
    (state row, seed), one panel per moment. With `ci`, a second row of panels
    shows the length of the 95% confidence interval of each cell over the
    common-noise realisations (bootstrap over the (gamma, eps0) rows)."""
    n_r = 2 if ci is not None else 1
    fig, axes = plt.subplots(n_r, 2, figsize=(7.5 * 2, n_r * (1.1 * len(rows) + 2.5)),
                             facecolor=SURFACE, squeeze=False)
    for c, mom in enumerate(('m1', 'm2')):
        S = 100.0 * stats[mom].T                                      # (rows, seeds)
        _cell_grid(axes[0, c], S, rows, seeds, 'Reds', max(1.0, np.ceil(np.nanmax(S))),
                   'moment 1' if mom == 'm1' else 'moment 2', ylabels=(c == 0))
        if ci is not None:
            L = 100.0 * ci[mom].T
            _cell_grid(axes[1, c], L, rows, seeds, 'Greens',
                       max(0.1, np.nanmax(L)), '95% CI length', ylabels=(c == 0),
                       fmt='{:.2f}')
    fig.tight_layout(w_pad=1.5, h_pad=2.5)
    fig.savefig(out_base + '.pdf', facecolor=SURFACE, bbox_inches='tight')
    fig.savefig(out_base + '.png', facecolor=SURFACE, bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print(f"-> {out_base}.pdf (+ .png)")


def _table_pm(stats, labels, rows, seeds, out_base, ci):
    """The moment tables with the 95% CI written as value +/- half-length in
    every cell, red scale on the value."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5 * 2, 1.35 * len(rows) + 3),
                             facecolor=SURFACE)
    for c, mom in enumerate(('m1', 'm2')):
        S = 100.0 * stats[mom].T
        H = 100.0 * ci[mom].T / 2.0
        vmax = max(1.0, np.ceil(np.nanmax(S)))
        ax = axes[c]
        ax.imshow(S, cmap='Reds', vmin=0, vmax=vmax, aspect='auto')
        for i in range(S.shape[0]):
            for k in range(S.shape[1]):
                v = S[i, k]
                ax.text(k, i, f'{v:.1f} $\\pm$ {H[i, k]:.2f}', ha='center',
                        va='center', fontsize=F_LABEL + 2,
                        color='white' if v > 0.6 * vmax else '#111111')
        ax.set_xticks(np.arange(len(seeds)))
        ax.set_xticklabels([f'seed {SEED_NAME.get(s, s)}' for s in seeds],
                           fontsize=F_TITLE)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows if c == 0 else [''] * len(rows), fontsize=F_TITLE)
        ax.set_title('moment 1' if mom == 'm1' else 'moment 2',
                     fontsize=F_TITLE + 6, pad=14)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out_base + '.pdf', facecolor=SURFACE, bbox_inches='tight')
    fig.savefig(out_base + '.png', facecolor=SURFACE, bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print(f"-> {out_base}.pdf (+ .png)")


def bootstrap_ci(d, labels, all_labels, keep, seeds, n_boot=200, seed=0):
    """95% CI length, over the common-noise realisations, of every table cell
    (mean over time of |normalised deviation|): resample the (gamma, eps0)
    rows with replacement, recompute seed means, consensus, sizes and cells."""
    raw = {mom: np.stack([d[f'{mom}_seed{s}_rows'] for s in seeds])
           for mom in ('m1', 'm2')}                                  # (seeds, all rows, Nm, T)
    Nm = raw['m1'].shape[2]
    rng = np.random.default_rng(seed)
    cells = {mom: [] for mom in raw}
    for b in range(n_boot):
        idx = rng.integers(0, Nm, Nm)
        M1_all = raw['m1'][:, :, idx, :].mean(axis=2)                # (seeds, all rows, T)
        for mom in raw:
            M = raw[mom][:, :, idx, :].mean(axis=2)[:, keep]
            R = _states_dev(M, M1_all, labels, all_labels, mom)
            cells[mom].append(np.nanmean(np.abs(R), axis=2))         # (seeds, rows)
    out = {}
    for mom in raw:
        C = np.stack(cells[mom])                                      # (B, seeds, rows)
        out[mom] = np.quantile(C, 0.975, axis=0) - np.quantile(C, 0.025, axis=0)
    print(f"bootstrap CI: {n_boot} resamples of the {Nm} common-noise rows")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='results_floor30')
    ap.add_argument('--nm', type=int, default=512)
    ap.add_argument('--agents', type=int, default=8)
    ap.add_argument('--skip-first', type=int, default=3)
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--replot', action='store_true')
    ap.add_argument('--norm', default='both',
                    choices=['ratio', 'rowscale', 'cap', 'states', 'both'])
    ap.add_argument('--view', default='all', choices=['all', 'actions', 'states'],
                    help='rows to show; files tagged action_ / state_')
    ap.add_argument('--actions-only', action='store_true',
                    help='rows A_t, f, xi, eta only; files tagged action_')
    args = ap.parse_args()

    cache = os.path.join(args.study, f'equilibrium_moments_nm{args.nm}.npz')
    if args.replot and os.path.exists(cache):
        d = dict(np.load(cache))
    else:
        runs = find_runs(args.study)
        d = simulate(args, runs)
        np.savez_compressed(cache, **d)
        print(f"cached -> {cache}")
    name = os.path.basename(args.study.rstrip('/')).replace('results_', '')
    view = 'actions' if args.actions_only else args.view
    tag = {'all': '', 'actions': 'action_', 'states': 'state_'}[view]
    base = os.path.join('figs_audit', 'all_seeds', 'equilibrium',
                        tag + f'moments_{name}')
    for mode in ('ratio', 'rowscale', 'cap', 'states'):
        if args.norm in (mode, 'both'):
            figure(d, base + '_' + mode, mode, view)


if __name__ == '__main__':
    main()
