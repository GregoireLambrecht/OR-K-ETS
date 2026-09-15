"""One convergence sheet per floor study.

    python3 plot_audit_sheet.py results_floor20 results_floor30 results_floor40

Rows: the BE-MFG solve itself (exploitability on the prior block, from
progress.json — the number the training loop optimises), then the four PMFG
audit scenarios. Columns: exploitability per type, population-weighted
exploitability, and the mixture's price-path movement ||A^k - A^{k-1}||_2.
Choice is solid, BAU dashed, both modes of a study on the same axes.

Inputs per <study>/<mode>/:
    progress.json                          BE row
    audit/<scenario>/b*/k*/result.json     audit rows, columns 1-2
    audit/<scenario>/paths.json            audit rows, column 3 (audit_paths.py)
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from helper_plot import (TYPE_COLORS, TYPE_NAMES, INK, INK_2, INK_MUTED,
                         SURFACE, ZERO_REF, GRID, _style, _save)
# same definitions as collect_audit.py, which is a script and cannot be imported
SCEN_ORDER = ('current_kets', 'low_alloc', 'high_tax', 'no_market_maker',
              'high_floor_mid_alloc')
SCEN_LABEL = {'current_kets': 'Current K-ETS',
              'low_alloc': 'Low allocation',
              'high_tax': 'High tax',
              'no_market_maker': 'No market maker',
              'high_floor_mid_alloc': 'High floor, mid alloc'}
MODE_LS = {'choice': '-', 'bau': (0, (5, 2.4))}

MODES = ('choice', 'bau')
MODE_LABEL = {'choice': 'Choice', 'bau': 'BAU'}
REF_LEVEL = 5.0          # % reference line on the exploitability panels
ITER = 'Fictitious-play iteration'


def load_be(run_dir):
    with open(os.path.join(run_dir, 'progress.json')) as f:
        recs = json.load(f)['records']
    it = np.array([r['iteration'] for r in recs])
    pr = [r['blocks']['prior'] for r in recs]
    per = np.array([[b['per_type'][j]['gap_relative'] for j in range(3)]
                    for b in pr]) * 100
    wtd = np.array([b['weighted_relative'] for b in pr]) * 100
    l2 = np.array([b['path_l2'] for b in pr])
    return it, per, wtd, l2


def load_audit(run_dir, scen):
    cells = []
    for f in sorted(glob.glob(os.path.join(run_dir, 'audit', scen, 'b*',
                                           'k*', 'result.json'))):
        with open(f) as fh:
            cells.append(json.load(fh))
    if not cells:
        return None
    # one budget on file per scenario is the normal case; keep the largest
    bmax = max(c['iters'] for c in cells)
    cells = sorted((c for c in cells if c['iters'] == bmax),
                   key=lambda c: c['checkpoint'])
    it = np.array([c['checkpoint'] for c in cells])
    per = np.array([[c['per_type'][j]['pmfg']['gap_relative'] for j in range(3)]
                    for c in cells]) * 100
    wtd = np.array([c['weighted_pmfg'] for c in cells]) * 100
    wbe = np.array([c['weighted_be'] for c in cells]) * 100
    ctx = cells[-1]['context']
    pth = os.path.join(run_dir, 'audit', scen, 'paths.json')
    l2_it, l2 = None, None
    if os.path.exists(pth):
        with open(pth) as fh:
            p = json.load(fh)
        l2_it, l2 = np.array(p['iterations']), np.array(p['path_l2'])
    return it, per, wtd, wbe, ctx, l2_it, l2


def _ref(ax):
    ax.axhline(0, color=ZERO_REF, lw=0.9, zorder=1)
    ax.axhline(REF_LEVEL, color=INK_MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(1.0, REF_LEVEL, f'{REF_LEVEL:g}%', color=INK_MUTED, fontsize=7,
            ha='right', va='bottom', transform=ax.get_yaxis_transform())


def sheet(study, outdir):
    rows = ['be'] + list(SCEN_ORDER)
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.0 * len(rows)),
                             facecolor=SURFACE, sharex=True)
    have_paths = False
    for r, row in enumerate(rows):
        ax_t, ax_w, ax_p = axes[r]
        ctx_label = ''
        for mode in MODES:
            run_dir = os.path.join(study, mode)
            if not os.path.exists(os.path.join(run_dir, 'progress.json')):
                continue
            ls = MODE_LS[mode]
            if row == 'be':
                it, per, wtd, l2 = load_be(run_dir)
                mk = None
            else:
                got = load_audit(run_dir, row)
                if got is None:
                    continue
                it, per, wtd, wbe, ctx, l2_it, l2 = got
                mk = 'o'
                ctx_label = ('  (' + ', '.join(f'{v:g}' for v in ctx) + ')')
                ax_w.plot(it, wbe, color=INK_MUTED, lw=1.1, ls=ls, marker=mk,
                          ms=2.5, zorder=2)
            for j in range(3):
                ax_t.plot(it, per[:, j], color=TYPE_COLORS[j], lw=1.7, ls=ls,
                          marker=mk, ms=3, zorder=3)
            ax_w.plot(it, wtd, color=INK, lw=1.9, ls=ls, marker=mk, ms=3,
                      zorder=3)
            if row == 'be':
                ax_p.plot(it, l2, color=INK, lw=1.7, ls=ls, zorder=3)
            elif l2 is not None:
                ax_p.plot(l2_it, l2, color=INK, lw=1.7, ls=ls, zorder=3)
                have_paths = True
        _ref(ax_t); _ref(ax_w)
        ax_p.set_yscale('log')
        label = ('BE-MFG  (prior block)' if row == 'be'
                 else 'PMFG audit: ' + SCEN_LABEL[row])
        _style(ax_t, label + ctx_label if r > 0 else label,
               'exploitability (%)' if r == 0 else None)
        _style(ax_w, None, None)
        _style(ax_p, None, None)
        if r == 0:
            ax_t.set_title('Exploitability per type', color=INK, fontsize=10.5,
                           loc='left', pad=6)
            ax_w.set_title('Population-weighted exploitability', color=INK,
                           fontsize=10.5, loc='left', pad=6)
            ax_p.set_title(r'Price-path movement  $\|A^{k}-A^{k-1}\|_2$',
                           color=INK, fontsize=10.5, loc='left', pad=6)
            ax_t.text(0.0, 1.16, label, transform=ax_t.transAxes, color=INK_2,
                      fontsize=9)
        else:
            ax_t.set_title(label + ctx_label, color=INK_2, fontsize=9,
                           loc='left', pad=4)
        ax_t.set_ylabel('%', color=INK_2, fontsize=8.5)
        ax_w.set_ylabel('%', color=INK_2, fontsize=8.5)
        ax_p.set_ylabel('L2', color=INK_2, fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel(ITER, color=INK_2, fontsize=8.5)

    handles = ([Line2D([], [], color=TYPE_COLORS[j], lw=1.8, label=TYPE_NAMES[j])
                for j in range(3)]
               + [Line2D([], [], color=INK, lw=1.8, label='weighted, PMFG deviator'),
                  Line2D([], [], color=INK_MUTED, lw=1.1, label='weighted, BE deviator')]
               + [Line2D([], [], color=INK, lw=1.8, ls=MODE_LS[m],
                         label=MODE_LABEL[m]) for m in MODES])
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=8.5, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{os.path.basename(study.rstrip('/'))}  —  BE-MFG solve "
                 f"and PMFG audit at the four policy scenarios",
                 color=INK, fontsize=12.5, x=0.005, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    name = os.path.basename(study.rstrip('/'))
    out = _save(fig, os.path.join(outdir, f'audit_sheet_{name}.pdf'))
    print(f"-> {out} (+ .png)" + ("" if have_paths else
          "   [no paths.json yet: column 3 empty for the audit rows]"))


if __name__ == '__main__':
    studies = sys.argv[1:] or ['results_floor20', 'results_floor30',
                               'results_floor40']
    for s in studies:
        sheet(s, 'figs_audit')
