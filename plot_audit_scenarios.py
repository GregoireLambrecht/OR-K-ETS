"""PMFG audit across seeds, one figure per scenario — same three panels as
plot_bemfg.py, drawn from the audit cells instead of the training records.

    python3 plot_audit_scenarios.py --study results_floor30 [--band range|std]

For every scenario with audit cells under <study>/choice*/audit/<scenario>/:

    (1) E_PMFG per type at each audited checkpoint, band across seeds
    (2) population-weighted E_PMFG, weighted per seed, then mean + band
    (3) price-path movement ||A^k - A^{k-1}||_2 at the scenario context
        (audit_paths.py), every iteration, band across seeds

Seeds that have not been audited at a checkpoint are left out of that
checkpoint's statistics. Writes
figs_audit/all_seeds/exploitability/<scenario>/<scenario>_<study>_<panel>.{pdf,png}\n(paper style, one panel per file).
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_bemfg import find_runs, panels
import paper_style as ps

SCEN_ORDER = ('current_kets', 'low_alloc', 'high_tax', 'no_market_maker',
              'high_floor_mid_alloc')


def load_cells(run_dir, scen):
    cells = []
    for f in sorted(glob.glob(os.path.join(run_dir, 'audit', scen, 'b*',
                                           'k*', 'result.json'))):
        with open(f) as fh:
            cells.append(json.load(fh))
    if not cells:
        return None
    bmax = max(c['iters'] for c in cells)
    cells = sorted((c for c in cells if c['iters'] == bmax),
                   key=lambda c: c['checkpoint'])
    ck = np.array([c['checkpoint'] for c in cells])
    per = np.array([[c['per_type'][j]['pmfg']['gap_relative'] for j in range(3)]
                    for c in cells]) * 100
    wtd = np.array([c['weighted_pmfg'] for c in cells]) * 100
    return ck, per, wtd


def load_paths(run_dir, scen):
    p = os.path.join(run_dir, 'audit', scen, 'paths.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return np.array(d['iterations']), np.array(d['path_l2'])


def stack_scenario(runs, scen):
    """-> (ck, per[s,ck,3], wtd[s,ck]), (it, l2[s,it]) with NaN for missing."""
    cells = [(r, load_cells(r, scen)) for r in runs]
    cells = [(r, c) for r, c in cells if c is not None]
    if not cells:
        return None
    ck_all = sorted({int(k) for _, c in cells for k in c[0]})
    pos = {k: i for i, k in enumerate(ck_all)}
    per = np.full((len(cells), len(ck_all), 3), np.nan)
    wtd = np.full((len(cells), len(ck_all)), np.nan)
    for s, (_, (ck, p, w)) in enumerate(cells):
        for i, k in enumerate(ck):
            per[s, pos[int(k)]] = p[i]
            wtd[s, pos[int(k)]] = w[i]
    paths = [load_paths(r, scen) for r, _ in cells]
    paths = [p for p in paths if p is not None]
    it = l2 = None
    if paths:
        K = max(int(p[0].max()) for p in paths)
        it = np.arange(1, K + 1)
        l2 = np.full((len(paths), K), np.nan)
        for s, (pi, pl) in enumerate(paths):
            l2[s, pi - 1] = pl
    return np.array(ck_all), per, wtd, it, l2, [r for r, _ in cells]


def main():
    ps.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='results_floor30')
    ap.add_argument('--band', default='range', choices=['range', 'std'])
    ap.add_argument('--outdir', default='figs_audit/all_seeds/exploitability')
    ap.add_argument('--scenarios', nargs='+', default=list(SCEN_ORDER))
    args = ap.parse_args()

    runs = find_runs(args.study)
    name = os.path.basename(args.study.rstrip('/')).replace('results_', '')
    for scen in args.scenarios:
        got = stack_scenario(runs, scen)
        if got is None:
            print(f"{scen}: no audit cells, skipped")
            continue
        ck, per, wtd, it, l2, used = got
        n = np.sum(~np.isnan(wtd), axis=0)
        print(f"{scen}: {len(used)} seed(s) audited; seeds per checkpoint "
              + " ".join(str(x) for x in n)
              + ("" if l2 is not None else "   [no paths.json: panel 3 empty]"))
        n_min, n_max = int(n.min()), int(n.max())

        # one file per panel (the old combined <scen>_<study>.pdf is left as is)
        out = os.path.join(args.outdir, scen)
        panels(out, f'{scen}_{name}', ck, per, wtd, l2, args.band, n_min, n_max,
               l2_it=it)
        print(f"-> {out}/{scen}_{name}_{{per_type,weighted,path}}.pdf")


if __name__ == '__main__':
    main()
