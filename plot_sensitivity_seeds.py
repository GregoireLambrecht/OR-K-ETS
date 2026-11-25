"""Sensitivity surfaces averaged over seeds, with optional +/- 1 s.d. shadows.

    python3 plot_sensitivity_seeds.py --study results_floor30 --x init_alloc --y kappa [--azim ...]

Reads the per-seed caches written by plot_sensitivity.py
(<run>/audit/sensitivity_<x>_<y>.npz, all on the same common-noise draw) for
every choice run of the study, averages each surface over seeds and draws the
surfaces one panel per file, in the paper style (paper_style.py):

    <base>_price_pollution<sfx>_{price,pollution}.*
    <base>_revenue<sfx>_{priv,pub,fi}.*

with <base> = <x>_<y>_<study>[_s<samples>] and <sfx> = _std (mean +/- 1 s.d.
over seeds) or _nci (2.5% / 97.5% quantiles over the common-noise draws),
--variants. Same view and z-range for all variants of one grid.

under figs_audit/all_seeds/sensibility_analysis/<x>_<y>/.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from plot_bemfg import find_runs
import paper_style as ps
from paper_style import (AXIS_LABEL, AXIS_SCALE, TYPE_COLORS, TYPE_TAGS,
                         PRICE_LABEL, POLLUTION_LABEL, REVENUE_LABEL)

VIEW = {'elev': 24, 'azim': -135, 'azim_market': -135}

N_TYPES = 3
SHADOW_ALPHA = 0.22


def load_seeds(study, x, y, samples):
    runs = find_runs(study)
    ds = []
    for r in runs:
        p = os.path.join(r, 'audit', f'sensitivity_{x}_{y}_s{samples}.npz')
        if os.path.exists(p):
            ds.append(dict(np.load(p)))
        else:
            print(f"  (no cache for {r}, skipped)")
    if not ds:
        raise SystemExit("no sensitivity caches found")
    for d in ds[1:]:
        assert np.allclose(d['x'], ds[0]['x']) and np.allclose(d['y'], ds[0]['y']), \
            "grids differ between seeds"
    keys = ['avg_price', 'pollution'] + [f'revenue_{j}' for j in range(N_TYPES)]
    agg = {'x': ds[0]['x'], 'y': ds[0]['y'], 'n_seeds': len(ds)}
    for k in keys:
        stack = np.stack([d[k] for d in ds])                  # (seeds, nx, ny)
        agg[k] = stack.mean(axis=0)
        agg[k + '_std'] = stack.std(axis=0, ddof=0)           # seed variability
        if k + '_draws' in ds[0]:
            # average over seeds at FIXED common noise, then the spread over
            # draws: the sensitivity of the equilibrium outcome to the noise
            cons = np.stack([d[k + '_draws'] for d in ds]).mean(axis=0)   # (S, nx, ny)
            agg[k + '_nstd'] = cons.std(axis=0, ddof=1)
            # 95% prediction interval of the outcome over the draws: quantiles
            # rather than +/- s.d., the distribution being skewed and bounded
            agg[k + '_nlo'] = np.quantile(cons, 0.025, axis=0)
            agg[k + '_nhi'] = np.quantile(cons, 0.975, axis=0)
            agg['n_draws'] = cons.shape[0]
    print(f"{len(ds)} seeds aggregated")
    return agg


def _surface(ax, X, Y, Z, color, zlabel, xlab, ylab, Zsd=None, band=None,
             zlim=None, azim=None):
    """Mean surface + translucent shadows; surfaces rasterized, axes vector.
    No title: the LaTeX caption names the panel."""
    cmap = LinearSegmentedColormap.from_list('s', ['#f2f2ef', color], N=256)
    # translucent shadows: +/- 1 s.d. (Zsd) or an explicit (lower, upper) band
    shadows = ([Z - Zsd, Z + Zsd] if Zsd is not None else
               list(band) if band is not None else [])
    for Zs in shadows:
        ax.plot_surface(X, Y, Zs, color=color, alpha=SHADOW_ALPHA,
                        linewidth=0, antialiased=True, shade=False,
                        rasterized=True)
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='#ffffff', linewidth=0.2,
                    antialiased=True, alpha=0.95, rasterized=True)
    ax.set_xlabel(xlab, labelpad=-4)
    ax.set_ylabel(ylab, labelpad=-4)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlabel, labelpad=0, rotation=90)
    ax.tick_params(axis='both', pad=-3)
    ax.tick_params(axis='z', pad=-1)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor('0.75')
        a._axinfo['grid'].update(linewidth=0.3, color='0.85')
        a.line.set_linewidth(0.6)
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.view_init(elev=VIEW['elev'], azim=VIEW['azim'] if azim is None else azim)


def _panel(figsize):
    fig = plt.figure(figsize=figsize)
    # the 3-D box fills the figure; the labels sit in the margin it leaves
    ax = fig.add_axes([0.0, 0.0, 0.92, 1.0], projection='3d')
    ax.set_box_aspect(None, zoom=0.78)
    ax.grid(True)
    return fig, ax


# (key, divisor, z-label, colour, figsize, file tag)
def _quantities(types):
    q = [('avg_price', 1.0, PRICE_LABEL, '#111111', ps.SURF_MARKET, 'price', 'market'),
         ('pollution', 1e3, POLLUTION_LABEL, '#555555', ps.SURF_MARKET, 'pollution', 'market')]
    for j in types:
        q.append((f'revenue_{j}', 1e6, REVENUE_LABEL, TYPE_COLORS[j],
                  ps.SURF_REVENUE, TYPE_TAGS[j], 'revenue'))
    return q


def _shadows(d, key, scale, with_std):
    """-> (Zsd, band) for the variant, in display units."""
    if with_std == 'nci':
        return None, (d[key + '_nlo'] / scale, d[key + '_nhi'] / scale)
    if with_std in ('std', 'nstd'):
        return d[key + '_' + with_std] / scale, None
    return None, None


def _zlim(d, key, scale, variants):
    """Common z-range for every variant of a grid (union of all shadows)."""
    lo, hi = np.inf, -np.inf
    for v in variants:
        Z = d[key] / scale
        sd, bd = _shadows(d, key, scale, v)
        parts = [Z] + ([Z - sd, Z + sd] if sd is not None else []) \
            + (list(bd) if bd is not None else [])
        lo = min(lo, min(float(p.min()) for p in parts))
        hi = max(hi, max(float(p.max()) for p in parts))
    pad = 0.03 * (hi - lo)
    return lo - pad, hi + pad


def figures(d, out_base, xname, yname, types, with_std, variants):
    """One file per surface: <out_base>_price_pollution<sfx>_{price,pollution}
    and <out_base>_revenue<sfx>_{priv,pub,fi}. with_std: None = mean only;
    'std' = +/- 1 s.d. over seeds; 'nstd' = +/- 1 s.d. over common-noise draws;
    'nci' = 2.5% / 97.5% quantile surfaces over the draws. Same view and same
    z-range for every variant in `variants`, so they compare at a glance."""
    X, Y = np.meshgrid(d['x'] * AXIS_SCALE.get(xname, 1.0),
                       d['y'] * AXIS_SCALE.get(yname, 1.0), indexing='ij')
    xl, yl = AXIS_LABEL[xname], AXIS_LABEL[yname]
    sfx = '' if with_std is None else '_' + with_std
    for key, scale, zl, color, size, tag, group in _quantities(types):
        sd, bd = _shadows(d, key, scale, with_std)
        fig, ax = _panel(size)
        _surface(ax, X, Y, d[key] / scale, color, zl, xl, yl, sd, bd,
                 zlim=_zlim(d, key, scale, variants),
                 azim=VIEW['azim_market'] if group == 'market' else VIEW['azim'])
        grp = 'price_pollution' if group == 'market' else 'revenue'
        ps.save(fig, f'{out_base}_{grp}{sfx}_{tag}', check=False)


def main():
    ps.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='results_floor30')
    ap.add_argument('--x', default='init_alloc')
    ap.add_argument('--y', default='kappa')
    ap.add_argument('--types', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--elev', type=float, default=24)
    ap.add_argument('--azim', type=float, default=-135)
    ap.add_argument('--azim-market', type=float, default=-135)
    ap.add_argument('--samples', type=int, default=1,
                    help='which per-seed cache to aggregate (common-noise draws)')
    ap.add_argument('--variants', nargs='+', default=['std', 'nci'],
                    choices=['mean', 'std', 'nstd', 'nci'])
    ap.add_argument('--outdir', default=os.path.join('figs_audit', 'all_seeds',
                                                     'sensibility_analysis'))
    args = ap.parse_args()
    args.variants = [None if v == 'mean' else v for v in args.variants]
    VIEW.update(elev=args.elev, azim=args.azim, azim_market=args.azim_market)

    d = load_seeds(args.study, args.x, args.y, args.samples)
    name = os.path.basename(args.study.rstrip('/')).replace('results_', '')
    base = os.path.join(args.outdir,
                        f'{args.x}_{args.y}',
                        f'{args.x}_{args.y}_{name}' + (f'_s{args.samples}' if args.samples > 1 else ''))
    # only the variants the paper uses; the old combined files are left as is
    variants = [v for v in args.variants
                if v in (None, 'std') or 'avg_price_nstd' in d]
    for with_std in variants:
        figures(d, base, args.x, args.y, args.types, with_std, variants)


if __name__ == '__main__':
    main()
