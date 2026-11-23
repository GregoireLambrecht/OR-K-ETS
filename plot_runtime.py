"""Computational cost of the scenario-extended solve against per-scenario solves.

    python3 plot_runtime.py [--study results_floor30] [--scenario current_kets]

Left  : cumulative best-response training time vs fictitious-play iteration,
        BE-MFG (context-conditional, one solve for the whole box) against PMFG
        (unconditional, one solve per scenario). Mean over seeds, band = +/- 1 s.d.
        Only BR training is counted: the exploitability evaluation is timed
        outside it in both solvers, and is a diagnostic, not part of the solve.
Writes figs_audit/all_seeds/computational/runtime_<study>.{pdf,png}.
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_bemfg import band
import paper_style as ps

C_BE, C_PMFG = '#111111', '#8a8983'
BAND_ALPHA = 0.18
RUNTIME_MARGINS = dict(left=0.42, right=0.06, bottom=0.32, top=0.05)


def br_seconds(progress_path):
    with open(progress_path) as f:
        recs = json.load(f)['records']
    recs = sorted(recs, key=lambda r: r['iteration'])
    return (np.array([r['iteration'] for r in recs]),
            np.array([r['br_seconds'] for r in recs]))


def collect(paths, label):
    runs = [br_seconds(p) for p in paths if os.path.exists(p)]
    if not runs:
        raise SystemExit(f"{label}: no progress.json found")
    K = max(int(it.max()) for it, _ in runs)
    cum = np.full((len(runs), K), np.nan)
    for s, (it, sec) in enumerate(runs):
        cum[s, it - 1] = np.cumsum(sec) / 3600.0
    print(f"{label}: {len(runs)} seeds, {K} iterations, "
          f"total {np.nanmean(cum[:, -1]):.2f} h "
          f"[{np.nanmin(cum[:, -1]):.2f}, {np.nanmax(cum[:, -1]):.2f}]")
    return np.arange(1, K + 1), cum


def draw_runtime(ax, it_be, be, it_pm, pm):
    """Mean over seeds, shaded +/- 1 s.d. band (narrow: may be invisible)."""
    for it, cum, color, ls, mk, label in (
            (it_be, be, C_BE, '-', 'o', 'Scenario-extended'),
            (it_pm, pm, C_PMFG, (0, (4, 1.5)), 's', 'Fixed scenario')):
        m, lo, hi = band(cum, 'std')
        ax.fill_between(it, lo, hi, color=color, alpha=BAND_ALPHA, lw=0)
        ax.plot(it, m, color=color, ls=ls, marker=mk, markevery=2, label=label)
    ax.set_xlabel('Fictitious-play iteration $k$')
    ax.set_ylabel('Time (hours)')
    ax.set_xlim(1, max(it_be[-1], it_pm[-1]))
    ax.legend(frameon=False, loc='upper left')


def main():
    ps.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='results_floor30')
    ap.add_argument('--scenario', default='current_kets')
    ap.add_argument('--outdir', default='figs_audit/all_seeds/computational')
    args = ap.parse_args()

    be_paths = [os.path.join(args.study, 'choice', 'progress.json')]
    be_paths += sorted(glob.glob(os.path.join(
        args.study, 'choice_seed*', 'choice', 'progress.json')))
    pm_paths = sorted(glob.glob(os.path.join(
        'results_pmfg', f'{args.scenario}_seed*', 'progress.json')))
    it_be, be = collect(be_paths, 'BE-MFG')
    it_pm, pm = collect(pm_paths, 'PMFG  ')
    be_total, pm_total = np.nanmean(be[:, -1]), np.nanmean(pm[:, -1])
    print(f"overhead of conditioning: {be_total / pm_total:.2f}x per solve; "
          f"break-even at n = {be_total / pm_total:.2f} scenarios")

    name = os.path.basename(args.study.rstrip('/')).replace('results_', '')
    fig, ax = ps.line_panel(ps.RUNTIME, RUNTIME_MARGINS)
    draw_runtime(ax, it_be, be, it_pm, pm)
    ps.save(fig, os.path.join(args.outdir, f'runtime_{name}'))


if __name__ == '__main__':
    main()
