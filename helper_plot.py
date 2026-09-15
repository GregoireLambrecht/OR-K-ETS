"""Figures for the BE-MFG study.

The merge of what were plot_rung1/2/3. Palette and mark conventions are defined
once at the top and used by every figure, so the per-iteration sheets, the
convergence sheet and the deviator sheet all read as one system: colour = agent
type, solid = mixture, dashed = new best response, shaded = 10-90% of agents.

Which sheet belongs to which rung
---------------------------------
The suffixes mirror the diagnostic ladder in utils.py — each rung adds one
source of difficulty, so a figure exists per level to localise a failure.

  plot_iteration, plot_convergence          Rung 1: one scenario, one frozen
                                            noise draw. Deterministic given the
                                            mixture, so the curves carry no
                                            Monte Carlo error.

  plot_iteration_r2, plot_convergence_r2    Rung 2: one scenario, stochastic
                                            common noise, measured on a frozen
                                            verification set. Same panels as
                                            Rung 1 so the two are read directly
                                            against each other.

  plot_iteration_r3, plot_convergence_r3,   Rung 3: the BE layer. The
  plot_deviator_r3                          per-iteration sheet is drawn at
                                            CTX_REF so it stays comparable with
                                            the sheets above; the convergence
                                            sheet keeps the Rung 1/2 COLUMNS
                                            unchanged and adds one ROW per
                                            evaluation block, the first being
                                            the prior expectation and the rest
                                            one-at-a-time deviations from the
                                            reference context. The deviator
                                            sheet carries V_br per block, which
                                            is what exposes a conditional best
                                            response that is under-trained in
                                            one corner of the context box.
"""

import os
import numpy as np
import matplotlib
from utils import (bind_env, bind_policy, BLOCK_ORDER,
                   FIXED_SCENARIOS, CTX_NAMES, CTX_REF)


# =============================================================================
# from plot_rung1.py
# =============================================================================

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── design tokens ────────────────────────────────────────────────────────────
TYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]     # blue, orange, aqua
TYPE_NAMES  = ["Private producer", "Large public producer", "Market maker"]

INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#8a8983"
GRID      = "#e6e5e1"
SURFACE   = "#fcfcfb"
ZERO_REF  = "#b9b8b2"

LW_MIX, LW_BR = 1.9, 1.7
DASH_BR = (0, (5, 2.4))
DPI     = 150

IDX = {"t": 0, "theta": 1, "xi_f": 2, "xi_g": 3, "eps": 4,
       "A": 5, "P": 6, "A_avg": 7, "ef": 8, "eg": 9,
       "cf": 10, "cg": 11, "cap_f": 12, "cap_xi": 13, "sigma": 14, "tech": 15}


def _style(ax, title=None, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=6, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5)


def _save(fig, path):
    """Write both PDF and PNG at the same basename."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    base = path[:-4] if path.lower().endswith(".pdf") else path
    fig.savefig(base + ".pdf", format="pdf", facecolor=SURFACE,
                bbox_inches="tight")
    fig.savefig(base + ".png", format="png", facecolor=SURFACE,
                bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return base + ".pdf"


def _emissions_cum(state):
    """Cumulative emissions from a mean-state trajectory (T+1, 16).
    ef and eg are constant within a type, so the mean of the product is the
    product with the mean."""
    return (state[:, IDX["ef"]] * state[:, IDX["xi_f"]]
            + state[:, IDX["eg"]] * state[:, IDX["xi_g"]]
            + state[:, IDX["eps"]])


def _emissions(state):
    """Per-step emissions e_t, length T.

    Taken as the first difference of the cumulative series rather than rebuilt
    from the mean action: ef*E[xi*eta] != ef*E[xi]*E[eta], whereas differencing
    is exact because the mean commutes with the difference. Includes the
    uncontrolled increment eps_t.
    """
    return np.diff(_emissions_cum(state))


BAND_ALPHA = 0.15


def _overlay(ax, ta, mix_series, br_series, lo=None, hi=None):
    """All three types on one axis: colour = type, dashed = best response.

    lo/hi, when given, shade the mixture's 80% interval across its agents.
    Bands are drawn first so the lines stay legible on top of them.
    """
    if lo is not None and hi is not None:
        for j in range(3):
            ax.fill_between(ta, lo[j], hi[j], color=TYPE_COLORS[j],
                            alpha=BAND_ALPHA, linewidth=0, zorder=1)
    for j in range(3):
        ax.plot(ta, mix_series[j], color=TYPE_COLORS[j], lw=LW_MIX, zorder=3)
        ax.plot(ta, br_series[j],  color=TYPE_COLORS[j], lw=LW_BR,
                ls=DASH_BR, zorder=3)


def plot_iteration(diag, path, title_suffix=""):
    """One diagnostic sheet per FP iteration.

    Two shared market panels (there is one market, so one price), then five
    per-quantity panels with all three types overlaid.
    """
    mix, br = diag["mix"], diag["br"]
    A, P    = diag["A_traj"], diag["P_traj"]
    floor   = diag["Afloor"]
    T       = len(A) - 1
    ta, ts  = np.arange(T), np.arange(T + 1)

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4), facecolor=SURFACE)
    fig.suptitle(f"Rung 1 diagnostics{title_suffix}",
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.0)

    # ── shared market ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(ts, A, color=INK, lw=LW_MIX)
    ax.axhline(floor, color=ZERO_REF, lw=1.1, ls=(0, (3, 3)))
    ax.annotate("price floor", (0.015, floor), xycoords=("axes fraction", "data"),
                color=INK_MUTED, fontsize=7.5, va="bottom")
    ax.annotate(f"{float(np.mean(A <= floor + 1e-6)):.0%} of steps on the floor",
                (0.98, 0.05), xycoords="axes fraction", ha="right",
                color=INK_MUTED, fontsize=8)
    _style(ax, "Allowance price $A_t$", "price", "day")

    ax = axes[0, 1]
    ax.plot(ts, P[:T + 1], color=INK, lw=LW_MIX)
    _style(ax, "Good price $P_t$ (exogenous)", "price", "day")

    # ── per-quantity panels, all types overlaid ──────────────────────────────
    ax = axes[0, 2]
    _overlay(ax, ta, [mix["action"][j][:, 0] for j in range(3)],
                     [br["action"][j][:, 0]  for j in range(3)],
             [mix["action_lo"][j][:, 0] for j in range(3)],
             [mix["action_hi"][j][:, 0] for j in range(3)])
    ax.axhline(0.0, color=ZERO_REF, lw=1.0)
    _style(ax, "Mean trade $f_t$", "allowances / step", "day")

    ax = axes[0, 3]
    _overlay(ax, ta, [mix["cum_r"][j] for j in range(3)],
                     [br["cum_r"][j]  for j in range(3)],
             [mix["cum_r_lo"][j] for j in range(3)],
             [mix["cum_r_hi"][j] for j in range(3)])
    ax.axhline(0.0, color=ZERO_REF, lw=1.0)
    _style(ax, "Cumulative running reward", "reward", "day")

    ax = axes[1, 0]
    _overlay(ax, ta, [mix["action"][j][:, 1] for j in range(3)],
                     [br["action"][j][:, 1]  for j in range(3)],
             [mix["action_lo"][j][:, 1] for j in range(3)],
             [mix["action_hi"][j][:, 1] for j in range(3)])
    _style(ax, "Mean production $\\xi_t$", "goods / step", "day")

    ax = axes[1, 1]
    _overlay(ax, ta, [mix["action"][j][:, 2] for j in range(3)],
                     [br["action"][j][:, 2]  for j in range(3)],
             [mix["action_lo"][j][:, 2] for j in range(3)],
             [mix["action_hi"][j][:, 2] for j in range(3)])
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Fossil share $\\eta_t$", "share", "day")

    ax = axes[1, 2]
    _overlay(ax, ta, [mix["emission"][j] for j in range(3)],
                     [br["emission"][j]  for j in range(3)],
             [mix["emission_lo"][j] for j in range(3)],
             [mix["emission_hi"][j] for j in range(3)])
    _style(ax, "Emissions $e_t$", "allowance-equivalents / step", "day")

    # legend occupies the last slot
    ax = axes[1, 3]
    ax.axis("off")
    handles = (
        [Line2D([], [], color=TYPE_COLORS[j], lw=LW_MIX, label=TYPE_NAMES[j])
         for j in range(3)]
        + [Line2D([], [], color="none", label=""),
           Line2D([], [], color=INK_2, lw=LW_MIX, label="mixture (incumbent)"),
           Patch(facecolor=INK_2, alpha=BAND_ALPHA, linewidth=0,
                 label="mixture, 80% of agents"),
           Line2D([], [], color=INK_2, lw=LW_BR, ls=DASH_BR,
                  label="new best response")]
    )
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=10,
              labelcolor=INK_2, handlelength=2.6, borderaxespad=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, path)


def plot_convergence(records, path, title_suffix=""):
    """Per-type exploitability across FP iterations — the convergence read."""
    it = [r["iteration"] for r in records]

    fig, axes = plt.subplots(1, 5, figsize=(19.5, 3.7), facecolor=SURFACE)
    fig.suptitle(f"Rung 1 convergence{title_suffix}",
                 color=INK, fontsize=13, x=0.005, ha="left", y=1.03)

    ax = axes[0]
    for j in range(3):
        g  = np.array([r["per_type"][j]["gap"] for r in records])
        se = np.array([r["per_type"][j]["gap_se"] for r in records])
        ax.plot(it, g, color=TYPE_COLORS[j], lw=LW_MIX, marker="o", ms=4)
        ax.fill_between(it, g - se, g + se, color=TYPE_COLORS[j],
                        alpha=0.16, linewidth=0)
    ax.axhline(0.0, color=ZERO_REF, lw=1.1)
    _style(ax, "Exploitability $\\mathcal{E}_j$", "reward", "FP iteration")

    ax = axes[1]
    for j in range(3):
        rel = np.array([r["per_type"][j]["gap_relative"] for r in records]) * 100
        ax.plot(it, rel, color=TYPE_COLORS[j], lw=LW_MIX, marker="o", ms=4)
    ax.axhline(0.0, color=ZERO_REF, lw=1.1)
    _style(ax, "$\\mathcal{E}_j$ as % of type reward scale", "%",
           "FP iteration")

    ax = axes[2]
    ax.plot(it, [r["weighted"] for r in records], color=INK, lw=LW_MIX,
            marker="o", ms=4)
    ax.axhline(0.0, color=ZERO_REF, lw=1.1)
    _style(ax, "Population-weighted $\\sum_j \\mathfrak{n}_j \\mathcal{E}_j$",
           "reward", "FP iteration")

    ax = axes[3]
    wrel = [r.get("weighted_relative",
                  float("nan")) * 100 for r in records]
    ax.plot(it, wrel, color=INK, lw=LW_MIX, marker="o", ms=4)
    ax.axhline(0.0, color=ZERO_REF, lw=1.1)
    _style(ax,
           "Population-weighted %  "
           "$\\sum_j \\mathfrak{n}_j\\,\\mathcal{E}_j/S_j$",
           "%", "FP iteration")

    ax = axes[4]
    ax.plot(it, [r["path_l2"] for r in records], color=INK, lw=LW_MIX,
            marker="o", ms=4)
    _style(ax, "$\\|A^{k+1}-A^{k}\\|_2$ (mixture path movement)", "L2",
           "FP iteration")
    if all(r["path_l2"] > 0 for r in records):
        ax.set_yscale("log")

    handles = [Line2D([], [], color=TYPE_COLORS[j], lw=LW_MIX,
                      marker="o", ms=4, label=TYPE_NAMES[j]) for j in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _save(fig, path)


# =============================================================================
# from plot_rung2.py
# =============================================================================

matplotlib.use("Agg")

import jax
import jax.numpy as jnp

from envs.environment import N_TYPES
from envs.models import stack_policies
from diagnostics import collect_iteration_diagnostics

PLOT_SEED = 4242      # the single common-noise draw every sheet is drawn on


def make_plot_noise(env_T, T, M=64, seed=PLOT_SEED):
    """One frozen common-noise realisation, shaped like a Rung 1 noise dict so
    diagnostics.collect_iteration_diagnostics and helper_plot.plot_iteration can
    be reused unchanged."""
    T_int = int(T)
    kP, kE, kp, ke = jax.random.split(jax.random.PRNGKey(seed), 4)

    def idio(key, n, j):
        return jax.vmap(env_T.generate_idiosyncratic_noise,
                        in_axes=(0, None, None))(
            jax.random.split(key, n), env_T.agent_params[j, 7], T_int)

    kps = jax.random.split(kp, N_TYPES)
    kes = jax.random.split(ke, N_TYPES)
    return {
        'P_traj':    env_T.generate_P(kP, T_int),
        'eps0':      env_T.generate_eps0(kE, T_int),
        'idio_pop':  tuple(idio(kps[j], env_T.type_counts[j], j)
                           for j in range(N_TYPES)),
        'idio_eval': tuple(idio(kes[j], M, j) for j in range(N_TYPES)),
    }


def plot_iteration_r2(env_T, history, new_triple, plot_noise, path,
                      title_suffix="", statics=None):
    """Mixture vs new best response on the fixed plotting draw."""
    if statics is None:
        statics = tuple(stack_policies([history[0][j]])[1]
                        for j in range(N_TYPES))
    diag = collect_iteration_diagnostics(env_T, history, new_triple,
                                         plot_noise, statics)
    return plot_iteration(diag, path, title_suffix=title_suffix)


def plot_convergence_r2(records, path, title_suffix=""):
    """Per-type exploitability across FP iterations, with the spread across the
    frozen common-noise set."""
    it = [r["iteration"] for r in records]

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.4), facecolor=SURFACE)
    fig.suptitle(f"Rung 2 convergence — common noise{title_suffix}",
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.0)

    ax = axes[0, 0]
    for j in range(3):
        g   = np.array([r["per_type"][j]["gap"] for r in records])
        p10 = np.array([r["per_type"][j].get("gap_p10", np.nan) for r in records])
        p90 = np.array([r["per_type"][j].get("gap_p90", np.nan) for r in records])
        ax.fill_between(it, p10, p90, color=TYPE_COLORS[j],
                        alpha=BAND_ALPHA, linewidth=0, zorder=1)
        ax.plot(it, g, color=TYPE_COLORS[j], lw=LW_MIX, marker="o", ms=3.5,
                zorder=3)
    ax.axhline(0.0, color=ZERO_REF, lw=1.1)
    _style(ax, "$\\mathcal{E}_j$  (band = 10-90% across common-noise draws)",
           "reward", "FP iteration")

    ax = axes[0, 1]
    for j in range(3):
        rel = np.array([r["per_type"][j]["gap_relative"] for r in records]) * 100
        ax.plot(it, rel, color=TYPE_COLORS[j], lw=LW_MIX, marker="o", ms=3.5)
    ax.axhline(10, color=ZERO_REF, lw=1.1, ls=(0, (3, 3)))
    ax.annotate("10%", (0.02, 10), xycoords=("axes fraction", "data"),
                color=INK_MUTED, fontsize=7.5, va="bottom")
    ax.axhline(0.0, color=ZERO_REF, lw=1.0)
    _style(ax, "$\\mathcal{E}_j$ as % of type reward scale", "%", "FP iteration")

    ax = axes[0, 2]
    for j in range(3):
        se = np.array([r["per_type"][j]["gap_se"] for r in records])
        ax.plot(it, se, color=TYPE_COLORS[j], lw=LW_MIX, marker="o", ms=3.5)
    ax.set_yscale("log")
    _style(ax, "Standard error of $\\mathcal{E}_j$\n(paired, over the frozen set)",
           "reward", "FP iteration")

    ax = axes[1, 0]
    ax.plot(it, [r["weighted"] for r in records], color=INK, lw=LW_MIX,
            marker="o", ms=3.5)
    ax.axhline(0.0, color=ZERO_REF, lw=1.1)
    _style(ax, "Population-weighted $\\sum_j \\mathfrak{n}_j \\mathcal{E}_j$",
           "reward", "FP iteration")

    ax = axes[1, 1]
    ax.plot(it, [r.get("weighted_relative", np.nan) * 100 for r in records],
            color=INK, lw=LW_MIX, marker="o", ms=3.5)
    ax.axhline(0.0, color=ZERO_REF, lw=1.1)
    _style(ax, "Population-weighted %  "
               "$\\sum_j \\mathfrak{n}_j\\,\\mathcal{E}_j/S_j$", "%",
           "FP iteration")

    # terminal allowance price across the frozen draws — the common-noise fan
    ax = axes[1, 2]
    med = np.array([r.get("A_final_median", np.nan) for r in records])
    p10 = np.array([r.get("A_final_p10", np.nan) for r in records])
    p90 = np.array([r.get("A_final_p90", np.nan) for r in records])
    ax.fill_between(it, p10, p90, color=INK, alpha=0.12, linewidth=0)
    ax.plot(it, med, color=INK, lw=LW_MIX, marker="o", ms=3.5)
    ax2 = None
    _style(ax, "Terminal price $A_T$ across the frozen draws\n"
               "(median, 10-90%)", "price", "FP iteration")

    handles = (
        [Line2D([], [], color=TYPE_COLORS[j], lw=LW_MIX, marker="o", ms=3.5,
                label=TYPE_NAMES[j]) for j in range(3)]
        + [Patch(facecolor=INK_2, alpha=BAND_ALPHA, linewidth=0,
                 label="10-90% across common-noise draws")]
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return _save(fig, path)


# =============================================================================
# from plot_rung3.py
# =============================================================================

matplotlib.use("Agg")

import equinox as eqx

from envs.base_params import GAMMA_REF

# Built from FIXED_SCENARIOS so the sheet cannot drift out of step with the
# blocks the FP loop actually evaluates. Each fixed row is a one-at-a-time
# deviation from `ref`, so the label names only the instrument that moved.
ROW_LABELS = {'prior': "$\\mathbb{E}_{\\gamma\\sim q}$  (prior)"}
for _n, _c in FIXED_SCENARIOS:
    _d = "  ".join(f"{k}={v:g}" for k, v, r in zip(CTX_NAMES, _c, CTX_REF)
                   if v != r)
    ROW_LABELS[_n] = _n if not _d else f"{_n}\n{_d}"


def make_plot_noise_r3(env_T, T, M=64, seed=PLOT_SEED):
    """The Rung 2 plotting draw, reused verbatim. The sheet is drawn at
    GAMMA_REF, and the idiosyncratic/common draws do not depend on gamma."""
    return make_plot_noise(env_T, T, M=M, seed=seed)


def plot_iteration_r3(env_T, history, new_triple, plot_noise, path,
                      title_suffix="", statics=None):
    """Mixture vs new best response, at GAMMA_REF on the fixed plotting draw."""
    g = jnp.asarray(CTX_REF, dtype=jnp.float32)
    env_g = bind_env(env_T, g)
    hist_g = [tuple(bind_policy(h[j], g) for j in range(N_TYPES))
              for h in history]
    new_g = tuple(bind_policy(new_triple[j], g) for j in range(N_TYPES))
    if statics is None:
        statics = tuple(stack_policies([hist_g[0][j]])[1]
                        for j in range(N_TYPES))
    diag = collect_iteration_diagnostics(env_g, hist_g, new_g,
                                         plot_noise, statics)
    return plot_iteration(diag, path, title_suffix=title_suffix)


def _blocks(records, name):
    """The per-iteration record of one block, tolerating older records."""
    return [r['blocks'][name] for r in records if name in r.get('blocks', {})]


def plot_convergence_r3(records, path, title_suffix=""):
    rows = [n for n in BLOCK_ORDER if _blocks(records, n)]
    if not rows:
        return None
    nrow = len(rows)

    fig, axes = plt.subplots(nrow, 5, figsize=(24.0, 3.5 * nrow),
                             facecolor=SURFACE, squeeze=False)
    fig.suptitle(f"Rung 3 convergence — BE layer{title_suffix}",
                 color=INK, fontsize=13, x=0.005, ha="left", y=1.0)

    for i, name in enumerate(rows):
        bs = _blocks(records, name)
        it = [r["iteration"] for r in records if name in r.get('blocks', {})]

        # col 1 — E_j absolute, band = 10-90% across the rows of the block
        ax = axes[i, 0]
        for j in range(N_TYPES):
            g   = np.array([b["per_type"][j]["gap"] for b in bs])
            p10 = np.array([b["per_type"][j].get("gap_p10", np.nan) for b in bs])
            p90 = np.array([b["per_type"][j].get("gap_p90", np.nan) for b in bs])
            ax.fill_between(it, p10, p90, color=TYPE_COLORS[j],
                            alpha=BAND_ALPHA, linewidth=0, zorder=1)
            ax.plot(it, g, color=TYPE_COLORS[j], lw=LW_MIX, marker="o", ms=3.0,
                    zorder=3)
        ax.axhline(0.0, color=ZERO_REF, lw=1.1)
        _style(ax, "$\\mathcal{E}_j$  (band = 10-90% over the block)",
               "reward", "FP iteration")
        ax.set_ylabel(ax.get_ylabel(), color=INK_2)
        ax.annotate(ROW_LABELS.get(name, name), (-0.30, 0.5),
                    xycoords="axes fraction", rotation=90, va="center",
                    ha="center", color=INK, fontsize=11)

        # col 2 — E_j as % of the type reward scale
        ax = axes[i, 1]
        for j in range(N_TYPES):
            rel = np.array([b["per_type"][j]["gap_relative"] for b in bs]) * 100
            ax.plot(it, rel, color=TYPE_COLORS[j], lw=LW_MIX, marker="o", ms=3.0)
        ax.axhline(10, color=ZERO_REF, lw=1.1, ls=(0, (3, 3)))
        ax.annotate("10%", (0.02, 10), xycoords=("axes fraction", "data"),
                    color=INK_MUTED, fontsize=7.5, va="bottom")
        ax.axhline(0.0, color=ZERO_REF, lw=1.0)
        _style(ax, "$\\mathcal{E}_j$ as % of type reward scale", "%",
               "FP iteration")

        # col 3 — population-weighted, absolute
        ax = axes[i, 2]
        ax.plot(it, [b["weighted"] for b in bs], color=INK, lw=LW_MIX,
                marker="o", ms=3.0)
        ax.axhline(0.0, color=ZERO_REF, lw=1.1)
        _style(ax, "Population-weighted "
                   "$\\sum_j \\mathfrak{n}_j \\mathcal{E}_j$",
               "reward", "FP iteration")

        # col 4 — population-weighted, %
        ax = axes[i, 3]
        ax.plot(it, [b["weighted_relative"] * 100 for b in bs], color=INK,
                lw=LW_MIX, marker="o", ms=3.0)
        ax.axhline(10, color=ZERO_REF, lw=1.1, ls=(0, (3, 3)))
        ax.axhline(0.0, color=ZERO_REF, lw=1.0)
        _style(ax, "Population-weighted %  "
                   "$\\sum_j \\mathfrak{n}_j\\,\\mathcal{E}_j/S_j$", "%",
               "FP iteration")

        # col 5 — price-path convergence
        ax = axes[i, 4]
        l2 = np.array([b["path_l2"] for b in bs], dtype=float)
        ax.plot(it, l2, color=INK, lw=LW_MIX, marker="o", ms=3.0)
        if np.any(np.isfinite(l2) & (l2 > 0)):
            ax.set_yscale("log")
        _style(ax, "Price-path convergence $\\|A^{k}-A^{k-1}\\|_2$",
               "L2", "FP iteration")

    handles = (
        [Line2D([], [], color=TYPE_COLORS[j], lw=LW_MIX, marker="o", ms=3.5,
                label=TYPE_NAMES[j]) for j in range(N_TYPES)]
        + [Line2D([], [], color=INK, lw=LW_MIX, marker="o", ms=3.5,
                  label="population-weighted aggregate"),
           Patch(facecolor=INK_2, alpha=BAND_ALPHA, linewidth=0,
                 label="10-90% over the block")]
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9.5, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.02 - 0.005 * nrow))
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.985))
    return _save(fig, path)


def plot_deviator_r3(records, path, title_suffix=""):
    """Companion figure: what did not fit in the 4x5 grid.

    V_br per scenario is the diagnostic of §8.5.2 — a single FiLM deviator has
    to be near-optimal at every gamma, and a corner where it is not shows as a
    depressed V_br relative to gamma_mid rather than as a large gap.
    """
    rows = [n for n in BLOCK_ORDER if _blocks(records, n)]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.0), facecolor=SURFACE)
    fig.suptitle(f"Rung 3 — deviator quality and price spread{title_suffix}",
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.02)

    styles = ['-', (0, (5, 2.4)), (0, (1, 1.6)), (0, (4, 1.5, 1, 1.5))]

    ax = axes[0]
    for name, ls in zip(rows, styles):
        bs = _blocks(records, name)
        it = [r["iteration"] for r in records if name in r.get('blocks', {})]
        for j in range(N_TYPES):
            ax.plot(it, [b["per_type"][j].get("V_br_relative", np.nan) * 100
                         for b in bs],
                    color=TYPE_COLORS[j], lw=LW_BR, ls=ls)
    ax.axhline(0.0, color=ZERO_REF, lw=1.1)
    _style(ax, "$V_{br}$ / scale per scenario\n(0 = collapsed deviator)",
           "%", "FP iteration")

    ax = axes[1]
    for name, ls in zip(rows, styles):
        bs = _blocks(records, name)
        it = [r["iteration"] for r in records if name in r.get('blocks', {})]
        for j in range(N_TYPES):
            ax.plot(it, [b["per_type"][j]["gap_se"] for b in bs],
                    color=TYPE_COLORS[j], lw=LW_BR, ls=ls)
    ax.set_yscale("log")
    _style(ax, "Standard error of $\\mathcal{E}_j$\n(paired, over the block)",
           "reward", "FP iteration")

    ax = axes[2]
    for name, ls in zip(rows, styles):
        bs = _blocks(records, name)
        it = [r["iteration"] for r in records if name in r.get('blocks', {})]
        ax.fill_between(it, [b["A_final_p10"] for b in bs],
                        [b["A_final_p90"] for b in bs],
                        color=INK, alpha=0.08, linewidth=0)
        ax.plot(it, [b["A_final_median"] for b in bs], color=INK, lw=LW_BR,
                ls=ls)
    _style(ax, "Terminal price $A_T$ per scenario\n(median, 10-90%)",
           "price", "FP iteration")

    handles = (
        [Line2D([], [], color=TYPE_COLORS[j], lw=LW_BR, label=TYPE_NAMES[j])
         for j in range(N_TYPES)]
        + [Line2D([], [], color=INK_2, lw=LW_BR, ls=ls,
                  label=ROW_LABELS.get(n, n)) for n, ls in zip(rows, styles)]
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return _save(fig, path)
