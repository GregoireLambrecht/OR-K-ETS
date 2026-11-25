"""Convergence graphs for one rung-2 (fixed-scenario) run, same drawing code as
plot_bemfg.py:

    convergence_per_type   exploitability per type vs FP iteration
    convergence_weighted   population-weighted exploitability vs FP iteration
    convergence_path       price-path movement ||A^k - A^{k-1}||_2 vs FP iteration

    python3 plot_mc_convergence.py --run-dir results_mc_pmfg/current_kets_T30_K5_seed42

Writes <run-dir>/figs/convergence_{per_type,weighted,path}.{pdf,png}.
"""
import os
import json
import argparse
import numpy as np

import paper_style as ps
from plot_bemfg import panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--block', default=None,
                    help='rung 3: evaluation block (prior, ref, ...); default prior')
    args = ap.parse_args()
    ps.apply()
    with open(os.path.join(args.run_dir, 'progress.json')) as f:
        prog = json.load(f)
    recs = prog['records']
    suffix = ''
    if prog.get('rung') == 3:
        blk = args.block or 'prior'
        recs = [dict(r['blocks'][blk], iteration=r['iteration']) for r in recs]
        suffix = f'_{blk}'
    it = np.array([r['iteration'] for r in recs])
    per = np.array([[p['gap_relative'] for p in r['per_type']] for r in recs])[None] * 100
    wtd = np.array([r['weighted_relative'] for r in recs])[None] * 100
    l2 = np.array([r['path_l2'] for r in recs])[None]
    panels(os.path.join(args.run_dir, 'figs'), 'convergence' + suffix, it, per, wtd, l2,
           'range', 1, 1)


if __name__ == '__main__':
    main()
