"""Allowance-price paths of the equilibrium mixture, one curve per scenario.

    python3 plot_allowance_paths.py --run-dir results_floor30/choice [--samples 250]

The full FP mixture is bound at each scenario context and priced on the SAME
--samples common-noise draws (common random numbers, so the curves differ by
policy, not by luck). Drawn: mean path with a +/- 1 s.d. band per scenario.
Writes figs_audit/<seed>/sampling/allowances/price_paths_<study>.{pdf,png} and
caches the paths as <run_dir>/audit/<scenario>/price_paths.npz.
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

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import stack_policies
from utils import (chunked, draw_common_noise, draw_idio, load_triple_r3,
                   bind_env, make_price_fn_r3, CONTEXT_DIM, _G_HI)
from audit_pmfg import audit_scenarios
from plot_bemfg import SURFACE, DPI

PATH_SEED = 4242
SCEN_ORDER = ('current_kets', 'low_alloc', 'high_tax', 'no_market_maker',
              'high_floor_mid_alloc')
SCEN_LABEL = {'current_kets': 'Current K-ETS',
              'low_alloc': 'Low allocation',
              'high_tax': 'High tax',
              'no_market_maker': 'No financial institution',
              'high_floor_mid_alloc': 'High floor, mid allocation'}
SCEN_COLOR = {'current_kets': '#111111',
              'low_alloc': '#2a78d6',
              'high_tax': '#eb6834',
              'no_market_maker': '#1baf7a',
              'high_floor_mid_alloc': '#8e44ad'}
F_TITLE, F_LABEL, F_TICK, F_LEG = 32, 30, 26, 26


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
    scen = audit_scenarios(floor_max)
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

    history = [load_triple_r3(args.run_dir, i) for i in range(n_avail)]
    statics = tuple(stack_policies([history[0][j]])[1] for j in range(N_TYPES))
    stacked = tuple(stack_policies([h[j] for h in history])[0]
                    for j in range(N_TYPES))
    price_fn = make_price_fn_r3(env, statics)

    # ONE noise set for every scenario: common random numbers
    kc, kp = jax.random.split(jax.random.PRNGKey(PATH_SEED))
    P, eps0 = draw_common_noise(env, T, args.samples, kc)
    idio_pop = draw_idio(env, T, env.type_counts, kp)
    print(f"PRICE PATHS | {args.run_dir} | {n_avail} triples, T={T}, "
          f"S={args.samples} common-noise draws shared by all scenarios",
          flush=True)

    out = {'T': T, 'S': args.samples}
    for name in SCEN_ORDER:
        ca = jnp.asarray(scen[name], dtype=jnp.float32)
        ctx_rows = jnp.broadcast_to(ca, (args.samples, CONTEXT_DIM))
        A = chunked(lambda a, b: price_fn(stacked, ctx_rows[a:b], P[a:b],
                                          eps0[a:b], idio_pop),
                    args.samples, args.chunk)
        A = np.asarray(A)
        out[name] = A
        out[name + '_floor'] = float(scen[name][2])
        print(f"  {name:<22s} mean final {A[:, -1].mean():7.2f}   "
              f"on floor {np.mean(A <= scen[name][2] + 1e-6):5.1%}", flush=True)
        np.savez_compressed(os.path.join(args.run_dir, 'audit', name,
                                         'price_paths.npz'),
                            A=A, context=np.asarray(scen[name]), S=args.samples)
    return out


def figure(d, out_base):
    T = int(d['T'])
    t = np.arange(T + 1)
    fig, ax = plt.subplots(figsize=(18, 9), facecolor=SURFACE)
    for name in SCEN_ORDER:
        if name not in d:
            continue
        A = d[name]
        m, s = A.mean(axis=0), A.std(axis=0)
        ax.fill_between(t, m - s, m + s, color=SCEN_COLOR[name], alpha=0.16,
                        lw=0)
        ax.plot(t, m, color=SCEN_COLOR[name], lw=3.2, label=SCEN_LABEL[name])
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e6e5e1", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=F_TICK, length=6, width=1.2)
    ax.set_title(r'Allowance price $A_t$: mean $\pm$ 1 s.d. over '
                 f'{int(d["S"])} common-noise samples',
                 fontsize=F_TITLE, pad=14, loc='left')
    ax.set_xlabel('day', fontsize=F_LABEL)
    ax.set_ylabel('Price', fontsize=F_LABEL)
    ax.set_xlim(0, T)
    ax.legend(fontsize=F_LEG, frameon=False, loc='upper left')
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base + '.pdf', facecolor=SURFACE, bbox_inches='tight')
    fig.savefig(out_base + '.png', facecolor=SURFACE, bbox_inches='tight',
                dpi=DPI)
    plt.close(fig)
    print(f"-> {out_base}.pdf (+ .png)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--samples', type=int, default=250)
    ap.add_argument('--chunk', type=int, default=50)
    ap.add_argument('--seed-label', default='seed42')
    ap.add_argument('--replot', action='store_true')
    args = ap.parse_args()

    if args.replot:
        d = {}
        for name in SCEN_ORDER:
            p = os.path.join(args.run_dir, 'audit', name, 'price_paths.npz')
            if os.path.exists(p):
                z = np.load(p)
                d[name] = z['A']
                d['T'] = z['A'].shape[1] - 1
                d['S'] = int(z['S'])
    else:
        d = simulate(args)
    # the study is the results_<name> component of the run path, whatever the
    # nesting below it (results_floor30/choice or results_floor30/choice_seed3/choice)
    study = next((p for p in args.run_dir.split('/') if p.startswith('results_')),
                 'run').replace('results_', '')
    figure(d, os.path.join('figs_audit', args.seed_label, 'sampling',
                           'allowances', f'price_paths_{study}'))


if __name__ == '__main__':
    main()
