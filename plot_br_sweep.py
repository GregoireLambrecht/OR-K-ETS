"""Plots for br_budget_sweep.py: per FP iteration k, exploitability per type vs
BR step budget (one marker per seed, line through the seed mean, dashed
horizontal line at the gap logged by the FP run), plus the BR training loss
curves per type coloured by budget.

    python3 plot_br_sweep.py --run-dir results_mc_pmfg/current_kets_T30_K30_seed42

Writes <run-dir>/br_sweep/figs/br_sweep_k<k>_{gap,loss}.{pdf,png} and
br_sweep_summary.{pdf,png} (weighted exploitability vs budget, all k).
"""
import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    ps.apply()
    sd = os.path.join(args.run_dir, 'br_sweep')
    figdir = os.path.join(sd, 'figs')
    files = sorted(glob.glob(os.path.join(sd, 'k*.json')))
    summary = []
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        k, runs = r['k'], r['runs']
        if not runs:
            continue
        budgets = sorted({x['iters'] for x in runs})

        # gap vs budget, one panel per type, same row size
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.9), sharex=True)
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.86, wspace=0.35)
        for j, ax in enumerate(axes):
            c = ps.TYPE_COLORS[j]
            for x in runs:
                ax.plot(x['iters'] / 1e3, 100 * x['per_type'][j]['gap_relative'],
                        ps.TYPE_MARKERS[j], color=c, alpha=0.55, ms=3)
            m = [np.mean([100 * x['per_type'][j]['gap_relative'] for x in runs if x['iters'] == n])
                 for n in budgets]
            ax.plot(np.array(budgets) / 1e3, m, '-', color=c, lw=1)
            o = 100 * r['original']['per_type'][j]['gap_relative']
            ax.axhline(o, color='0.3', ls='--', lw=0.7)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_title(ps.TYPE_NAMES[j], fontsize=7)
            ax.set_xlabel('BR steps (thousands)')
            ax.set_xticks(np.array(budgets) / 1e3)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        axes[0].set_ylabel('Exploitability (%)')
        fig.text(0.99, 0.97, f'FP iteration k = {k}   (dashed: logged gap of the FP run)',
                 ha='right', va='top', fontsize=7)
        ps.save(fig, os.path.join(figdir, f'br_sweep_k{k:03d}_gap'), check=False)

        # training-loss curves
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.9))
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.86, wspace=0.35)
        cmap = plt.get_cmap('viridis')
        for j, ax in enumerate(axes):
            for x in runs:
                L = np.asarray(x['loss_curve'])[:, j]
                t = np.linspace(0, x['iters'], len(L)) / 1e3
                ci = budgets.index(x['iters']) / max(1, len(budgets) - 1)
                ax.plot(t, L, color=cmap(ci), lw=0.6, alpha=0.8)
            ax.set_title(ps.TYPE_NAMES[j], fontsize=7)
            ax.set_xlabel('step (thousands)')
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        axes[0].set_ylabel('BR loss (−V / scale)')
        for n in budgets:
            axes[2].plot([], [], color=cmap(budgets.index(n) / max(1, len(budgets) - 1)),
                         label=f'{n // 1000}k')
        axes[2].legend(frameon=False, fontsize=6, loc='upper right')
        fig.text(0.99, 0.97, f'FP iteration k = {k}', ha='right', va='top', fontsize=7)
        ps.save(fig, os.path.join(figdir, f'br_sweep_k{k:03d}_loss'), check=False)

        summary.append((k, budgets, runs, 100 * r['original']['weighted_relative']))

    if summary:
        fig, ax = ps.line_panel(figsize=(3.1, 2.1))
        for i, (k, budgets, runs, o) in enumerate(summary):
            c = plt.get_cmap('tab10')(i)
            m = [np.mean([100 * x['weighted_relative'] for x in runs if x['iters'] == n])
                 for n in budgets]
            for x in runs:
                ax.plot(x['iters'] / 1e3, 100 * x['weighted_relative'], 'o', color=c, ms=2.5, alpha=0.5)
            ax.plot(np.array(budgets) / 1e3, m, '-', color=c, lw=1, label=f'k = {k}')
            ax.axhline(o, color=c, ls='--', lw=0.6)
        ax.set_xlabel('BR steps (thousands)')
        ax.set_ylabel('Weighted exploitability (%)')
        ax.legend(frameon=False, fontsize=6)
        ps.save(fig, os.path.join(figdir, 'br_sweep_summary'), check=False)


if __name__ == '__main__':
    main()
