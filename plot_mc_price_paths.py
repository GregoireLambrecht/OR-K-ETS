"""Mean cleared allowance-price path at every FP iteration (market clearing).

    python3 plot_mc_price_paths.py --run-dir results_mc_pmfg/current_kets_T30_K5_seed42

Reads progress.json (records[*]['clearing']['A_path_mean'], the mean over the
frozen verification draws of [A_REF, A_0, ..., A_{T-1}]) and draws one line per
iteration, light to dark, with the 10-90% band of the last iteration and the
price floor. Writes <run-dir>/figs/price_paths.{pdf,png}.
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import paper_style as ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--block', default='ref',
                    help='rung 3: evaluation block to draw (ref = current_kets)')
    args = ap.parse_args()
    ps.apply()
    with open(os.path.join(args.run_dir, 'progress.json')) as f:
        prog = json.load(f)
    if prog.get('rung') == 3:            # one record per block
        recs = [dict(r['blocks'][args.block], iteration=r['iteration'])
                for r in prog['records']]
        ctx = prog['fixed_scenarios'].get(args.block)
        floor = float(ctx[2]) if ctx else None
        suffix = f'_{args.block}'
    else:
        recs = prog['records']
        floor = float(prog['context'][2]) if 'context' in prog else None
        suffix = ''
    recs = [r for r in recs if 'clearing' in r]
    T = len(recs[0]['clearing']['A_path_mean']) - 1
    t = np.arange(T + 1)

    fig, ax = ps.line_panel(ps.RUNTIME, dict(left=0.45, right=0.62, bottom=0.32, top=0.05))
    cmap = plt.get_cmap('viridis_r')
    norm = Normalize(1, max(r['iteration'] for r in recs))
    last = recs[-1]['clearing']
    ax.fill_between(t, last['A_path_p10'], last['A_path_p90'],
                    color=cmap(norm(recs[-1]['iteration'])), alpha=0.15, lw=0)
    for r in recs:
        ax.plot(t, r['clearing']['A_path_mean'], color=cmap(norm(r['iteration'])),
                marker='o', markevery=max(1, T // 10))
    if floor is not None:
        ax.axhline(floor, color='0.55', lw=0.8, ls=(0, (4, 2)))
        ax.text(1.0, floor, 'floor', color='0.55', fontsize=7, ha='right',
                va='bottom', transform=ax.get_yaxis_transform())
    ax.set_xlim(0, T)
    ax.set_xlabel('Step $t$ (0 = pre-open price $A_{\\mathrm{ref}}$)')
    ax.set_ylabel('Mean cleared price (USD/tCO$_2$)')
    cax = fig.add_axes([1 - 0.55 / 3.1, 0.32 / 2.1, 0.04, 1 - 0.37 / 2.1])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('FP iteration $k$')
    its = sorted({r['iteration'] for r in recs})
    cb.set_ticks([k for k in its if k == 1 or k % 5 == 0])
    cb.outline.set_linewidth(0.4)
    ps.save(fig, os.path.join(args.run_dir, 'figs', 'price_paths' + suffix))


if __name__ == '__main__':
    main()
