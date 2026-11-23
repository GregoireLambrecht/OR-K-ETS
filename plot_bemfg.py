"""BE-MFG convergence across seeds — the paper's figure 1.

    python3 plot_bemfg.py --study results_floor30 [--band range|std]

Reads progress.json of every choice run of the study (results_floor30/choice
and results_floor30/choice_seed*/[choice/]) and draws, on the prior block:

    (a) exploitability per type            one line per type, band across seeds
    (b) population-weighted exploitability weighted per seed, then mean + band
    (c) price-path movement ||A^k - A^{k-1}||_2

Seeds that have not reached iteration k are left out of that iteration's
statistics, so a partial set of runs still gives a figure; the number of seeds
behind each iteration is printed and the minimum is written in the legend.
Writes one file per panel, bemfg_<study>_{per_type,weighted,path}.{pdf,png},
in the paper style (paper_style.py).
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helper_plot import TYPE_COLORS, SURFACE, DPI   # re-exported (other scripts)
import paper_style as ps

TYPE_NAMES = ("Private producer", "Large public producer", "Financial institution")
INK = "#111111"
C_MEAN = "#111111"
BAND_ALPHA = 0.18
REF_LEVEL = 5.0                     # % reference line on the exploitability panels


def find_runs(study):
    """Every choice run of the study: the seed-42 original plus choice_seed*."""
    runs = []
    for d in sorted(glob.glob(os.path.join(study, 'choice*'))):
        for cand in (d, os.path.join(d, 'choice')):
            if os.path.exists(os.path.join(cand, 'progress.json')):
                runs.append(cand)
                break
    return runs


def load(run_dir):
    with open(os.path.join(run_dir, 'progress.json')) as f:
        recs = json.load(f)['records']
    it = np.array([r['iteration'] for r in recs])
    pr = [r['blocks']['prior'] for r in recs]
    per = np.array([[b['per_type'][j]['gap_relative'] for j in range(3)]
                    for b in pr]) * 100                       # (n, 3) %
    wtd = np.array([b['weighted_relative'] for b in pr]) * 100
    l2 = np.array([b['path_l2'] for b in pr])
    return it, per, wtd, l2


def stack(runs):
    """Align runs on iteration; NaN where a seed has not got there yet."""
    data = [load(r) for r in runs]
    K = max(d[0].max() for d in data)
    it = np.arange(1, K + 1)
    per = np.full((len(runs), K, 3), np.nan)
    wtd = np.full((len(runs), K), np.nan)
    l2 = np.full((len(runs), K), np.nan)
    for s, (i, p, w, l) in enumerate(data):
        per[s, i - 1] = p
        wtd[s, i - 1] = w
        l2[s, i - 1] = l
    return it, per, wtd, l2


def band(x, kind):
    """(centre, lo, hi) over the seed axis, ignoring seeds without data."""
    m = np.nanmean(x, axis=0)
    if kind == 'std':
        s = np.nanstd(x, axis=0, ddof=0)
        return m, m - s, m + s
    return m, np.nanmin(x, axis=0), np.nanmax(x, axis=0)


def _style(ax, ylabel, log=False):
    """No title (the LaTeX caption names the panel)."""
    ax.set_xlabel('Fictitious-play iteration $k$')
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')


def _ref(ax):
    """Zero line and the dashed grey 5% reference level, labelled at the right."""
    ax.axhline(0, color="#b9b8b2", lw=0.5)
    ax.axhline(REF_LEVEL, color="#8a8983", lw=0.8, ls=(0, (4, 2)))
    ax.text(1.0, REF_LEVEL, f"{REF_LEVEL:g}%", color="#8a8983", fontsize=7,
            ha="right", va="bottom", transform=ax.get_yaxis_transform())


def draw_per_type(ax, it, per, kind, n_min):
    for j in range(3):
        m, lo, hi = band(per[:, :, j], kind)
        ax.fill_between(it, lo, hi, color=TYPE_COLORS[j], alpha=BAND_ALPHA, lw=0)
        ax.plot(it, m, color=TYPE_COLORS[j], ls=ps.TYPE_LINESTYLES[j],
                marker=ps.TYPE_MARKERS[j], label=TYPE_NAMES[j])
    _ref(ax)
    _style(ax, 'Exploitability (%)')
    # the only type legend of the row
    ax.legend(frameon=False, loc='upper right', handlelength=2.2,
              borderaxespad=0.1, labelspacing=0.25)


def draw_weighted(ax, it, wtd, kind, n_min, n_max):
    m, lo, hi = band(wtd, kind)
    ax.fill_between(it, lo, hi, color=C_MEAN, alpha=BAND_ALPHA, lw=0)
    ax.plot(it, m, color=C_MEAN, marker='o')
    _ref(ax)
    _style(ax, 'Exploitability (%)')


def draw_path(ax, it, l2, kind):
    m, lo, hi = band(l2, kind)
    ax.fill_between(it, lo, hi, color=C_MEAN, alpha=BAND_ALPHA, lw=0)
    ax.plot(it, m, color=C_MEAN, marker='o')
    _style(ax, r'$\|A^{k}-A^{k-1}\|_2$ (USD/tCO$_2$)', log=True)


def save(fig, base):
    ps.save(fig, base)


def panels(outdir, stem, it, per, wtd, l2, kind, n_min, n_max, l2_it=None):
    """One file per panel, at final size: <stem>_{per_type,weighted,path}."""
    todo = [('per_type', lambda ax: draw_per_type(ax, it, per, kind, n_min)),
            ('weighted', lambda ax: draw_weighted(ax, it, wtd, kind, n_min, n_max))]
    if l2 is not None:
        todo.append(('path', lambda ax: draw_path(
            ax, it if l2_it is None else l2_it, l2, kind)))
    for tag, fn in todo:
        fig, ax = ps.line_panel()
        fn(ax)
        save(fig, os.path.join(outdir, f'{stem}_{tag}'))


def main():
    ps.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='results_floor30')
    ap.add_argument('--band', default='range', choices=['range', 'std'])
    ap.add_argument('--outdir', default='figs_audit/all_seeds/exploitability/scenario_extended')
    args = ap.parse_args()

    runs = find_runs(args.study)
    if not runs:
        raise SystemExit(f"no choice runs with progress.json under {args.study}")
    it, per, wtd, l2 = stack(runs)
    n_seeds = np.sum(~np.isnan(wtd), axis=0)
    print(f"{len(runs)} runs: " + ", ".join(runs))
    print("seeds per iteration: " + " ".join(str(n) for n in n_seeds))
    n_min, n_max = int(n_seeds.min()), int(n_seeds.max())
    name = os.path.basename(args.study.rstrip('/')).replace('results_', '')

    # one file per panel (the old combined bemfg_<study>.pdf is left as is)
    panels(args.outdir, f'bemfg_{name}', it, per, wtd, l2, args.band, n_min, n_max)


if __name__ == '__main__':
    main()
