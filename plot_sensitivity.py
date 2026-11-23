"""Sensitivity of the equilibrium to two context dimensions — 3-D surfaces.

    python3 plot_sensitivity.py --run-dir results_floor30/choice \
        --x init_alloc --y kappa [--nx 15 --ny 15 --samples 1]

The full FP mixture is bound on a grid of (x, y) with the other two context
dimensions at their current_kets values, and priced on --samples common-noise
draws shared by every grid point. Two surfaces:

    z1  time-average allowance price  mean_t A_t
    z2  total revenue (cumulative reward incl. terminal penalty) of a private
        producer, averaged over mixture members and idiosyncratic draws

Writes figs_audit/<seed>/sensibility_analysis/<x>_<y>/<x>_<y>_<study>.{pdf,png}
and caches the grid as <run_dir>/audit/sensitivity_<x>_<y>.npz.
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
from matplotlib.colors import LinearSegmentedColormap

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import stack_policies
from utils import (chunked, draw_common_noise, draw_idio, load_triple_r3,
                   bind_env, bind_policy,
                   make_price_fn_r3, make_batched_return_fn_r3, CTX_NAMES,
                   CONTEXT_DIM, _G_LO, _G_HI)
from audit_pmfg import audit_scenarios
from plot_bemfg import TYPE_COLORS, TYPE_NAMES, SURFACE, DPI

SENS_SEED = 9001
F_TITLE, F_LABEL, F_TICK = 30, 24, 18
AXIS_LABEL = {'init_alloc': 'Initial allocation (%)',
              'kappa': r'Tax $\kappa$',
              'Afloor': 'Price floor',
              'financial_inst': 'Financial-institution share (%)'}
# ranges to sweep: the training box, except kappa which runs from 1: below
# kappa = 1 buying allowances is never optimal, the grid there is uninformative
SWEEP = {'init_alloc': (0.0, 1.0), 'kappa': (1.0, 7.0),
         'Afloor': None, 'financial_inst': (0.0, 0.5)}
# the share is rounded to whole agents out of N = 20 inside the environment, so
# it only changes every 5%: one grid point per step, whatever --nx/--ny says
POINTS = {'financial_inst': 11}


def make_batched_final_fn(env_T, j):
    """Total reward AND final cumulative emissions of type-j agents, per grid
    row -> (R (S, M), E (S, M)). Same rollout as make_batched_return_fn_r3."""
    idx = env_T.dic_indexes

    @eqx.filter_jit
    def fn(policy, gam, A, P, idio):
        def one(g, a, p):
            X, R = bind_env(env_T, g).rollout_agent_batch(
                bind_policy(policy, g), j, a, p, idio)
            E = (X[:, idx['ef']] * X[:, idx['xi_f']]
                 + X[:, idx['eg']] * X[:, idx['xi_g']] + X[:, idx['eps']])
            return jnp.stack([R, E])          # (2, M)
        return jnp.moveaxis(jax.vmap(one)(gam, A, P), 1, 0)   # (2, S, M)
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
    base = np.asarray(audit_scenarios(floor_max)['current_kets'], dtype=np.float32)
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

    ix, iy = CTX_NAMES.index(args.x), CTX_NAMES.index(args.y)
    rx = SWEEP[args.x] or (float(_G_LO[ix]), float(_G_HI[ix]))
    ry = SWEEP[args.y] or (float(_G_LO[iy]), float(_G_HI[iy]))
    args.nx = POINTS.get(args.x, args.nx)
    args.ny = POINTS.get(args.y, args.ny)
    xs = np.linspace(rx[0], rx[1], args.nx)
    ys = np.linspace(ry[0], ry[1], args.ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    G = args.nx * args.ny
    ctx = np.tile(base, (G, 1))
    ctx[:, ix] = X.ravel()
    ctx[:, iy] = Y.ravel()
    ctx = jnp.asarray(ctx)
    # population shares at every grid context (they move with financial_inst)
    shares = np.asarray(jax.vmap(lambda g: bind_env(env, g).context_weights())(ctx))

    history = [load_triple_r3(args.run_dir, i) for i in range(n_avail)]
    statics = tuple(stack_policies([history[0][j]])[1] for j in range(N_TYPES))
    stacked = tuple(stack_policies([h[j] for h in history])[0]
                    for j in range(N_TYPES))
    price_fn = make_price_fn_r3(env, statics)
    final_fns = [make_batched_final_fn(env, j) for j in range(N_TYPES)]

    kc, kp, ke = jax.random.split(jax.random.PRNGKey(SENS_SEED), 3)
    P, eps0 = draw_common_noise(env, T, args.samples, kc)
    idio_pop = draw_idio(env, T, env.type_counts, kp)
    # the agents ARE the simulated population that sets the price: n_j per
    # policy (type_counts), same idiosyncratic draws as the price path
    n_types = tuple(int(n) for n in env.type_counts)
    members = list(range(args.first_member, min(args.first_member + args.n_members, n_avail)))
    print(f"SENSITIVITY | {args.run_dir} | {args.x} x {args.y} = {args.nx} x "
          f"{args.ny} grid, others at current_kets {tuple(base)} | "
          f"{args.samples} common-noise draw(s); population = policies "
          f"{members[0]}..{members[-1]} x {n_types} agents per type",
          flush=True)

    # per-draw surfaces are kept (float32), so the aggregation over seeds can
    # average at fixed common noise and then take the spread over draws
    S = args.samples
    avg_price = np.zeros((S, G), np.float32)
    revenue = np.zeros((S, N_TYPES, G), np.float32)
    emissions = np.zeros((S, N_TYPES, G), np.float32)
    for s in range(S):
        Ps = jnp.broadcast_to(P[s], (G,) + P[s].shape)
        Es = jnp.broadcast_to(eps0[s], (G,) + eps0[s].shape)
        A = chunked(lambda a, b: price_fn(stacked, ctx[a:b], Ps[a:b], Es[a:b],
                                          idio_pop), G, args.chunk)
        avg_price[s] = np.asarray(A.mean(axis=1))
        for j in range(N_TYPES):
            R, E = np.zeros(G), np.zeros(G)
            for k in members:
                RE = chunked(lambda a, b: jnp.moveaxis(final_fns[j](
                    history[k][j], ctx[a:b], A[a:b], Ps[a:b], idio_pop[j]), 0, 1),
                    G, args.chunk)                       # (S, 2, M)
                Rk, Ek = RE[:, 0], RE[:, 1]
                R += np.asarray(Rk.mean(axis=1))
                E += np.asarray(Ek.mean(axis=1))
            revenue[s, j] = R / len(members)
            emissions[s, j] = E / len(members)
        if (s + 1) % 16 == 0 or s + 1 == S:
            print(f"  draw {s + 1}/{S} done", flush=True)
    # pollution per producer: sum of the producers' cumulative emissions over
    # the simulated producer population (private + public agents of every
    # policy), divided by their number. emissions[s, j] is already the mean
    # over the n_j x members agents of type j, so weight by n_j.
    # the private : public split follows the context's population weights
    # (context_weights), the same weights the price aggregation uses
    w_prod = shares[:, :N_TYPES - 1]                              # (G, 2)
    pollution = (np.einsum('gj,sjg->sg', w_prod, emissions[:, :N_TYPES - 1])
                 / w_prod.sum(axis=1)[None, :])
    shp = (S, args.nx, args.ny)
    out = {'x': xs, 'y': ys, 'base': base, 'samples': S,
           'first_member': args.first_member, 'n_members': len(members),
           'type_counts': np.array(n_types), 'agents_total': agents,
           # means over draws (what the single-seed figures draw)
           'avg_price': avg_price.mean(axis=0).reshape(args.nx, args.ny),
           'pollution': pollution.mean(axis=0).reshape(args.nx, args.ny),
           # per-draw surfaces
           'avg_price_draws': avg_price.reshape(shp),
           'pollution_draws': pollution.reshape(shp)}
    for j in range(N_TYPES):
        out[f'revenue_{j}'] = revenue[:, j].mean(axis=0).reshape(args.nx, args.ny)
        out[f'emissions_{j}'] = emissions[:, j].mean(axis=0).reshape(args.nx, args.ny)
        out[f'revenue_{j}_draws'] = revenue[:, j].reshape(shp)
    return out


def _surface(ax, X, Y, Z, color, zlabel, title, xlab, ylab):
    cmap = LinearSegmentedColormap.from_list('s', ['#f2f2ef', color], N=256)
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='#ffffff', linewidth=0.4,
                    antialiased=True, alpha=0.95)
    ax.set_xlabel(xlab, fontsize=F_LABEL, labelpad=22)
    ax.set_ylabel(ylab, fontsize=F_LABEL, labelpad=22)
    ax.set_zlabel(zlabel, fontsize=F_LABEL, labelpad=30, rotation=90)
    ax.set_title(title, fontsize=F_TITLE, pad=18)
    ax.tick_params(labelsize=F_TICK, pad=8)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.set_facecolor(SURFACE)
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev=VIEW['elev'], azim=VIEW['azim'])


VIEW = {'elev': 24, 'azim': -135}      # overridden by --elev / --azim

# axis values as displayed: shares as percentages
AXIS_SCALE = {'init_alloc': 100.0, 'financial_inst': 100.0}


def _save(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + '.pdf', facecolor=SURFACE, bbox_inches='tight',
                pad_inches=0.8)
    fig.savefig(out_base + '.png', facecolor=SURFACE, bbox_inches='tight',
                pad_inches=0.8, dpi=DPI)
    plt.close(fig)
    print(f"-> {out_base}.pdf (+ .png)")


def figure(d, out_base, xname, yname, types):
    X, Y = np.meshgrid(d['x'] * AXIS_SCALE.get(xname, 1.0),
                       d['y'] * AXIS_SCALE.get(yname, 1.0), indexing='ij')
    xl, yl = AXIS_LABEL[xname], AXIS_LABEL[yname]

    # 1. market outcomes: price and total pollution
    azim_rev = VIEW['azim']
    VIEW['azim'] = VIEW.get('azim_market', azim_rev)
    fig = plt.figure(figsize=(26, 11), facecolor=SURFACE)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    _surface(ax, X, Y, d['avg_price'], '#111111', 'Price',
             'Average allowance price', xl, yl)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    _surface(ax, X, Y, d['pollution'] / 1e3, '#555555', 'Emissions (thousands)',
             'Pollution per producer', xl, yl)
    fig.subplots_adjust(left=0.02, right=0.93, wspace=0.12)
    _save(fig, out_base + '_price_pollution')
    VIEW['azim'] = azim_rev

    # 2. revenues, one surface per requested type
    n = len(types)
    fig = plt.figure(figsize=(13 * n, 11), facecolor=SURFACE)
    for i, j in enumerate(types, start=1):
        ax = fig.add_subplot(1, n, i, projection='3d')
        _surface(ax, X, Y, d[f'revenue_{j}'] / 1e6, TYPE_COLORS[j],
                 'Revenue (millions)', f'{TYPE_NAMES[j]} total revenue', xl, yl)
    fig.subplots_adjust(left=0.02, right=0.95, wspace=0.12)
    _save(fig, out_base + '_revenue')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--x', default='init_alloc', choices=CTX_NAMES)
    ap.add_argument('--y', default='kappa', choices=CTX_NAMES)
    ap.add_argument('--nx', type=int, default=30)
    ap.add_argument('--ny', type=int, default=30)
    ap.add_argument('--samples', type=int, default=1,
                    help='common-noise draws (shared across the grid)')
    ap.add_argument('--first-member', type=int, default=0)
    ap.add_argument('--n-members', type=int, default=50,
                    help='mixture at checkpoint 50 = policies 0..49')
    ap.add_argument('--agents', type=int, default=32,
                    help='(unused: the population is the simulated one)')
    ap.add_argument('--skip-first', type=int, default=3)
    ap.add_argument('--types', type=int, nargs='+', default=[0, 1, 2],
                    help='types whose revenue surface is drawn')
    ap.add_argument('--chunk', type=int, default=75)
    ap.add_argument('--seed-label', default='seed42')
    ap.add_argument('--replot', action='store_true')
    ap.add_argument('--elev', type=float, default=24)
    ap.add_argument('--azim', type=float, default=-135,
                    help='azimuth of the revenue surfaces')
    ap.add_argument('--azim-market', type=float, default=-135,
                    help='azimuth of the price / pollution surfaces')
    args = ap.parse_args()
    VIEW.update(elev=args.elev, azim=args.azim, azim_market=args.azim_market)

    cache = os.path.join(args.run_dir, 'audit',
                         f'sensitivity_{args.x}_{args.y}_s{args.samples}.npz')
    if args.replot and os.path.exists(cache):
        d = dict(np.load(cache))
    else:
        d = simulate(args)
        np.savez_compressed(cache, **d)
        print(f"cached -> {cache}")
    # the study is the results_<name> component of the run path, whatever the
    # nesting below it (results_floor30/choice or results_floor30/choice_seed3/choice)
    study = next((p for p in args.run_dir.split('/') if p.startswith('results_')),
                 'run').replace('results_', '')
    figure(d, os.path.join('figs_audit', args.seed_label,
                           'sensibility_analysis', f'{args.x}_{args.y}',
                           f'{args.x}_{args.y}_{study}'), args.x, args.y,
           args.types)


if __name__ == '__main__':
    main()
