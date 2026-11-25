"""Per-iteration diagnostic sheets for a market-clearing fixed-scenario run.

    python3 plot_mc_iterations.py --run-dir results_mc_pmfg/current_kets_T30_K30_seed42

For every FP iteration k (mixture = policies 0..k-1, new best response =
policy_k), on ONE fixed common-noise draw (helper_plot.PLOT_SEED), draws the
same seven panels as the rung-1/2/3 sheets (helper_plot.plot_iteration) plus
five market-clearing panels:

    reservation prices   median R_t per type of the mixture vs the cleared A_t,
                         with the clearing interval [a-, a+]
    market balance       net supply S_t, withheld supply W_t (floor) and the
                         agents' aggregate trade sum_j w_j f_j  (= S_t - W_t)
    holdings             allowances held theta_t per type
    compliance gap       cumulative emissions - holdings, E_t - theta_t (> 0 = short)
    schedule slope       mean beta_t per type of the mixture

Writes <run-dir>/figs_iter/iter_<k>.{pdf,png}.
"""
import os
import sys
import json
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _floor_max_of_run(argv):
    """Rung 3: the FiLM normaliser box is fixed at import from KETS_FLOOR_MAX,
    so set it from the run's progress.json before envs/ is imported."""
    for i, a in enumerate(argv):
        if a == '--run-dir' and i + 1 < len(argv):
            p = os.path.join(argv[i + 1], 'progress.json')
            if os.path.exists(p):
                box = json.load(open(p)).get('context_box')
                return float(box[2][1]) if box else None
    return None


_FM = _floor_max_of_run(sys.argv)
if _FM is not None:
    os.environ['KETS_FLOOR_MAX'] = repr(_FM)

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import stack_policies
from utils import load_triple, load_triple_r3, bind_env, bind_policy
from helper_plot import (make_plot_noise, _style, _overlay, _save, _emissions_cum,
                         TYPE_COLORS, TYPE_NAMES, INK, INK_2, INK_MUTED, ZERO_REF,
                         SURFACE, LW_MIX, LW_BR, DASH_BR, BAND_ALPHA, IDX)


def build_env(prog):
    T = int(prog['T_grid'][-1])
    env = ExogenousMarketEnvJAX(
        kappa=GAMMA_REF[1], T=T,
        generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
        market_impact_func=market_impact_base,
        generate_eps0_func=supply_noise_base,
        generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
        A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE,
        agent_per_policy=int(prog.get('agent_per_policy', 20)))
    env = env.set_context(jnp.asarray(prog['context'], jnp.float32))
    return env.set_T(T)


def build_env_r3(prog, g):
    T = int(prog['T_grid'][-1])
    env = ExogenousMarketEnvJAX(
        kappa=GAMMA_REF[1], T=T,
        generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
        market_impact_func=market_impact_base,
        generate_eps0_func=supply_noise_base,
        generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
        A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE,
        agent_per_policy=int(prog.get('agent_per_policy', 20)))
    return bind_env(env.set_context(jnp.asarray(g, jnp.float32)).set_T(T),
                    jnp.asarray(g, jnp.float32))


def collect(env, history, new_triple, noise, statics):
    stacked = tuple(stack_policies([h[j] for h in history])[0] for j in range(N_TYPES))

    @eqx.filter_jit
    def mix_fn(stacked):
        return env.ensemble_trajectory(stacked, statics, noise['P_traj'],
                                       noise['eps0'], noise['idio_pop'])

    mix = jax.device_get(mix_fn(stacked))
    br = [jax.device_get(env.agent_batch_trajectory(
        new_triple[j], j, jnp.asarray(mix['A_traj']), noise['P_traj'], noise['idio_eval'][j]))
        for j in range(N_TYPES)]
    return mix, br


def sheet(env, mix, br, path, title):
    A, P = np.asarray(mix['A_traj']), np.asarray(noise_P := mix['P_traj'])
    clr = {k: np.asarray(v) for k, v in mix['clearing'].items()}
    floor = float(env.Afloor)
    w = np.asarray(env.context_weights())
    T = len(A) - 1
    ta, ts = np.arange(T), np.arange(T + 1)
    mstate = [np.asarray(mix['state'][j]) for j in range(N_TYPES)]
    bstate = [np.asarray(br[j]['state']) for j in range(N_TYPES)]
    mact = [np.asarray(mix['action'][j]) for j in range(N_TYPES)]
    bact = [np.asarray(br[j]['action']) for j in range(N_TYPES)]
    lo = [np.asarray(mix['action_lo'][j]) for j in range(N_TYPES)]
    hi = [np.asarray(mix['action_hi'][j]) for j in range(N_TYPES)]

    fig, axes = plt.subplots(3, 4, figsize=(16.0, 10.6), facecolor=SURFACE)
    fig.suptitle(title, color=INK, fontsize=13, x=0.006, ha="left", y=1.0)

    # ── row 1: market + trade + reward ───────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(ts, A, color=INK, lw=LW_MIX, marker='o', ms=2.5)
    ax.axhline(floor, color=ZERO_REF, lw=1.1, ls=(0, (3, 3)))
    ax.annotate("price floor", (0.015, floor), xycoords=("axes fraction", "data"),
                color=INK_MUTED, fontsize=7.5, va="bottom")
    ax.annotate(f"{clr['on_floor'].mean():.0%} of steps on the floor, "
                f"{clr['on_ceil'].mean():.0%} on the ceiling",
                (0.98, 0.05), xycoords="axes fraction", ha="right",
                color=INK_MUTED, fontsize=8)
    _style(ax, "Cleared allowance price $A_t$ ($t=0$: $A_{ref}$)", "USD/tCO$_2$", "step")

    ax = axes[0, 1]
    ax.plot(ts, noise_P[:T + 1], color=INK, lw=LW_MIX)
    _style(ax, "Good price $P_t$ (exogenous)", "USD/MWh", "step")

    ax = axes[0, 2]
    _overlay(ax, ta, [m[:, 0] for m in mact], [b[:, 0] for b in bact],
             [l[:, 0] for l in lo], [h[:, 0] for h in hi])
    ax.axhline(0.0, color=ZERO_REF, lw=1.0)
    _style(ax, "Mean trade $f_t$ (at the cleared price)", "allowances / step", "step")

    ax = axes[0, 3]
    _overlay(ax, ta, [np.asarray(mix['cum_r'][j]) for j in range(3)],
             [np.asarray(br[j]['cum_r']) for j in range(3)],
             [np.asarray(mix['cum_r_lo'][j]) for j in range(3)],
             [np.asarray(mix['cum_r_hi'][j]) for j in range(3)])
    ax.axhline(0.0, color=ZERO_REF, lw=1.0)
    _style(ax, "Cumulative running reward", "USD", "step")

    # ── row 2: production, fuel, emissions, slope ───────────────────────────
    ax = axes[1, 0]
    _overlay(ax, ta, [m[:, 1] for m in mact], [b[:, 1] for b in bact],
             [l[:, 1] for l in lo], [h[:, 1] for h in hi])
    _style(ax, "Mean production $\\xi_t$", "MWh / step", "step")

    ax = axes[1, 1]
    _overlay(ax, ta, [m[:, 2] for m in mact], [b[:, 2] for b in bact],
             [l[:, 2] for l in lo], [h[:, 2] for h in hi])
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Fossil share $\\eta_t$", "share", "step")

    ax = axes[1, 2]
    _overlay(ax, ta, [np.asarray(mix['emission'][j]) for j in range(3)],
             [np.asarray(br[j]['emission']) for j in range(3)],
             [np.asarray(mix['emission_lo'][j]) for j in range(3)],
             [np.asarray(mix['emission_hi'][j]) for j in range(3)])
    _style(ax, "Emissions $e_t$", "tCO$_2$ / step", "step")

    ax = axes[1, 3]
    for j in range(3):
        ax.plot(ta, clr['beta_mean'][:, j], color=TYPE_COLORS[j], lw=LW_MIX)
    ax.axhline(BETA_MIN, color=ZERO_REF, lw=1.0, ls=(0, (3, 3)))
    ax.annotate("$\\beta_{min}$", (0.015, BETA_MIN), xycoords=("axes fraction", "data"),
                color=INK_MUTED, fontsize=7.5, va="bottom")
    ax.set_yscale('log')
    _style(ax, "Schedule slope $\\beta_t$ (mixture mean)", "normalised", "step")

    # ── row 3: market clearing ──────────────────────────────────────────────
    ax = axes[2, 0]
    ax.fill_between(ta + 1, clr['a_minus'], clr['a_plus'], color=INK, alpha=0.15,
                    lw=0, step='mid')
    for j in range(3):
        ax.plot(ta, clr['R_median'][:, j], color=TYPE_COLORS[j], lw=LW_MIX)
    ax.plot(ts, A, color=INK, lw=LW_MIX, ls=DASH_BR)
    top = max(2.0 * A.max(), floor * 2)
    ax.set_ylim(min(0.0, A.min() - 1), top)
    _style(ax, "Reservation price $R_t$ (median) vs $A_t$ (dashed)",
           "USD/tCO$_2$", "step")

    ax = axes[2, 1]
    agg = clr['f_mean'] @ w
    ax.bar(ta, clr['S'], color=INK_MUTED, alpha=0.5, width=0.8, label="net supply $S_t$")
    ax.bar(ta, -clr['G'], color="#c23b22", alpha=0.7, width=0.8,
           label="withheld $-W_t$ (floor)")
    ax.plot(ta, agg, color=INK, lw=LW_MIX, marker='o', ms=2.5,
            label="agents' trade $\\sum_j w_j \\bar f_j$")
    ax.axhline(0.0, color=ZERO_REF, lw=1.0)
    ax.legend(fontsize=7, frameon=False, loc="best")
    _style(ax, "Market balance  ($\\sum_j w_j \\bar f_j = S_t - W_t$)",
           "allowances / agent / step", "step")

    ax = axes[2, 2]
    _overlay(ax, ts, [s[:, IDX['theta']] for s in mstate],
             [s[:, IDX['theta']] for s in bstate])
    _style(ax, "Allowances held $\\theta_t$", "tCO$_2$", "step")

    ax = axes[2, 3]
    _overlay(ax, ts, [_emissions_cum(s) - s[:, IDX['theta']] for s in mstate],
             [_emissions_cum(s) - s[:, IDX['theta']] for s in bstate])
    ax.axhline(0.0, color=ZERO_REF, lw=1.0)
    gap_T = sum(w[j] * (_emissions_cum(mstate[j]) - mstate[j][:, IDX['theta']])[-1]
                for j in range(3))
    ax.annotate(f"market at T: {'SHORT' if gap_T > 0 else 'long'} ({gap_T:+.0f})",
                (0.98, 0.05), xycoords="axes fraction", ha="right",
                color=INK_MUTED, fontsize=8)
    _style(ax, "Compliance gap $E_t-\\theta_t$  (> 0 = short)", "tCO$_2$", "step")

    handles = ([Line2D([], [], color=TYPE_COLORS[j], lw=LW_MIX, label=TYPE_NAMES[j])
                for j in range(3)]
               + [Line2D([], [], color=INK_2, lw=LW_MIX, label="mixture (incumbent)"),
                  Patch(facecolor=INK_2, alpha=BAND_ALPHA, linewidth=0,
                        label="mixture, 80% of agents"),
                  Line2D([], [], color=INK_2, lw=LW_BR, ls=DASH_BR,
                         label="new best response")])
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=9.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return _save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--block', default='ref',
                    help='rung 3: fixed scenario the sheets are drawn at')
    args = ap.parse_args()
    with open(os.path.join(args.run_dir, 'progress.json')) as f:
        prog = json.load(f)
    K = len(prog['records'])
    if prog.get('rung') == 3:
        g = prog['fixed_scenarios'][args.block]
        env = build_env_r3(prog, g)
        gj = jnp.asarray(g, jnp.float32)
        triples = [tuple(bind_policy(m, gj) for m in load_triple_r3(args.run_dir, k))
                   for k in range(K + 1)]
        out = os.path.join(args.run_dir, 'figs_iter', args.block)
        scen = f"rung 3 at {args.block}"
        prog['context'] = g
    else:
        env = build_env(prog)
        triples = [load_triple(args.run_dir, k) for k in range(K + 1)]
        out = os.path.join(args.run_dir, 'figs_iter')
        scen = prog.get('scenario', '')
    T = int(env.T)
    noise = make_plot_noise(env, T)
    statics = tuple(stack_policies([triples[0][j]])[1] for j in range(N_TYPES))
    for k in range(1, K + 1, args.every):
        mix, br = collect(env, triples[:k], triples[k], noise, statics)
        sheet(env, mix, br, os.path.join(out, f'iter_{k:03d}.pdf'),
              f"Market clearing  —  {scen} {tuple(prog['context'])}, iteration {k}, "
              f"T = {T}, mixture of {k}, seed {prog.get('seed')}")
        print(f"-> {out}/iter_{k:03d}.pdf", flush=True)


if __name__ == '__main__':
    main()
