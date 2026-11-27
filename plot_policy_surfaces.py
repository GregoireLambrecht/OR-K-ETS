"""Policy surfaces of the major–minor market: metrics, aggregations, figures
(POLICY_SURFACES.md §4, §5, §6.3, §7).

    python3 plot_policy_surfaces.py --surfaces <study>/surfaces

Reads fig{1..4}.npz of policy_surfaces.py and writes, in the style of
plot_sensitivity_seeds.py (_surface, paper_style):
    fig{n}.{pdf,png}      the paper band: seeds (Figs. 1, 2), common noise (Figs. 3, 4)
    fig{n}_alt.{pdf,png}  the other aggregation (appendix)
    summary.json          tau_floor, V_floor quantiles, Delta J^0 at CTX_REF, blind-major check
Metrics (traded prices ys['A'], days 0..T-1):
    1  A-bar   = mean_t A_t
    2  V_floor = sum_t G_t 1{on_floor} / sum_t O_t      (ratio of sums over markets per seed)
    3  L       = (A-bar - A-bar_blind) / A-bar_blind    (paired, market by market)
    4  sigma_A = sqrt(T) std_t(log A_t / A_{t-1}), ddof = 1
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

import paper_style as ps
from paper_style import AXIS_LABEL, AXIS_SCALE, PRICE_LABEL
from plot_sensitivity_seeds import _surface, _panel, VIEW

ap = argparse.ArgumentParser()
ap.add_argument('--surfaces', required=True)
ap.add_argument('--azim', type=float, default=-135)
ap.add_argument('--elev', type=float, default=24)
args = ap.parse_args()
VIEW.update(elev=args.elev, azim=args.azim, azim_market=args.azim)
ps.apply()
D = args.surfaces
LABEL = dict(AXIS_LABEL, mm_share=AXIS_LABEL['financial_inst'])
SCALE = dict(AXIS_SCALE, mm_share=AXIS_SCALE['financial_inst'])
CTX_REF = None


def load(i):
    p = os.path.join(D, f'fig{i}.npz')
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def seed_band(per_seed):
    """per_seed (L, nx, ny): mean over seeds +/- 1 s.d. over seeds."""
    return np.nanmean(per_seed, 0), np.nanstd(per_seed, 0, ddof=0), None


def noise_band(per_market):
    """per_market (L, nx, ny, S): average over seeds market by market, then the mean
    over markets and the 2.5 / 97.5 % quantiles over markets."""
    hat = per_market.mean(0)                     # NaN (undefined metric) kept market by market
    return np.nanmean(hat, -1), None, (np.nanquantile(hat, 0.025, -1), np.nanquantile(hat, 0.975, -1))


def draw(name, d, Z, sd, band, zlabel, color, zscale, zlim):
    X, Y = np.meshgrid(d['x'] * SCALE.get(str(d['xname']), 1.0),
                       d['y'] * SCALE.get(str(d['yname']), 1.0), indexing='ij')
    fig, ax = _panel(ps.SURF_MARKET)
    _surface(ax, X, Y, Z * zscale, color, zlabel, LABEL[str(d['xname'])], LABEL[str(d['yname'])],
             None if sd is None else sd * zscale,
             None if band is None else (band[0] * zscale, band[1] * zscale), zlim=zlim)
    ps.save(fig, os.path.join(D, name), check=False)


def zlim_of(variants, zscale):
    lo, hi = np.inf, -np.inf
    for Z, sd, band in variants:
        parts = [Z] + ([Z - sd, Z + sd] if sd is not None else []) + (list(band) if band is not None else [])
        lo = min(lo, min(float(np.nanmin(p)) for p in parts))
        hi = max(hi, max(float(np.nanmax(p)) for p in parts))
    lo, hi = lo * zscale, hi * zscale
    pad = 0.03 * (hi - lo) + 1e-9
    return lo - pad, hi + pad


def figure(i, d, per_seed, per_market, primary, zlabel, color, zscale):
    """primary: 'seed' or 'noise' -> fig{i} uses it, fig{i}_alt the other."""
    vs = {'seed': seed_band(per_seed), 'noise': noise_band(per_market)}
    alt = 'noise' if primary == 'seed' else 'seed'
    zl = zlim_of(list(vs.values()), zscale)
    draw(f'fig{i}', d, *vs[primary], zlabel, color, zscale, zl)
    draw(f'fig{i}_alt', d, *vs[alt], zlabel, color, zscale, zl)
    Z = vs[primary][0]
    print(f"fig{i}: {zlabel.replace(chr(10), ' ')}  range [{Z.min() * zscale:.3g}, {Z.max() * zscale:.3g}]  "
          f"(L={per_seed.shape[0]}, S={per_market.shape[-1]}; band {primary}, alt {alt})", flush=True)


def ref_index(d):
    """Grid point closest to the fixed coordinates (CTX_REF by default)."""
    fx = d['fixed']
    names = ('init_alloc', 'kappa', 'Afloor', 'mm_share')
    ix = int(np.argmin(np.abs(d['x'] - fx[names.index(str(d['xname']))])))
    iy = int(np.argmin(np.abs(d['y'] - fx[names.index(str(d['yname']))])))
    return ix, iy, [float(d['x'][ix]), float(d['y'][iy])]


summary = {}
q3 = lambda a: [float(x) for x in np.quantile(a, [0.025, 0.5, 0.975])]

d1 = load(1)
if d1 is not None:
    Abar = d1['A'].mean(-1)                                       # (L, nx, ny, S)
    figure(1, d1, Abar.mean(-1), Abar, 'seed', PRICE_LABEL, '#111111', 1.0)
    summary['fig1'] = {'A_bar_range': [float(Abar.mean((0, -1)).min()), float(Abar.mean((0, -1)).max())]}

d2 = load(2)
if d2 is not None:
    onf = d2['on_floor'].astype(bool)
    Gf = np.where(onf, d2['G'], 0.0)                              # (L, nx, ny, S, T)
    O = d2['O']
    assert (Gf >= -1e-6 * np.maximum(O, 1)).all(), 'negative rationed volume at the floor'
    V_seed = Gf.sum((-1, -2)) / np.maximum(O.sum((-1, -2)), 1e-12)          # ratio of sums over markets
    V_mkt = Gf.sum(-1) / np.maximum(O.sum(-1), 1e-12)                       # per market
    tau = onf.mean(-1)                                                       # time on the floor
    # §7 checks
    assert np.all((V_mkt >= -1e-6) & (V_mkt <= 1 + 1e-6)), 'V_floor outside [0, 1]'
    assert np.all((tau >= 0) & (tau <= 1))
    never = ~onf.any(-1)
    assert np.all(V_mkt[never] == 0), 'V_floor > 0 where the floor never binds'
    mism = ((tau == 0) != (V_mkt <= 1e-9))
    print(f"fig2 check: tau = 0 <=> V = 0 market by market: {mism.sum()} mismatch(es) "
          f"of {mism.size}", flush=True)
    assert mism.sum() == 0, 'tau_floor = 0 and V_floor = 0 disagree'
    figure(2, d2, V_seed, V_mkt, 'seed', 'Rationed volume\nat the floor (%)', '#1f5f8b', 100.0)
    ix, iy, gxy = ref_index(d2)
    summary['fig2'] = {
        'grid_point': gxy,
        'tau_floor_mean': float(tau[:, ix, iy].mean()),
        'tau_floor_q': q3(tau[:, ix, iy].mean(0)),
        'tau_floor_days_mean': float(tau[:, ix, iy].mean() * 365),
        'V_floor': float(V_seed[:, ix, iy].mean()),
        'V_floor_q': q3(V_mkt[:, ix, iy].mean(0)),
        'tau_floor_grid': tau.mean((0, -1)).tolist(),
        'V_floor_grid': V_seed.mean(0).tolist(),
    }

d3 = load(3)
if d3 is not None:
    Ab, Abb = d3['A'].mean(-1), d3['A_blind'].mean(-1)
    Lm = np.where(Abb > 0, (Ab - Abb) / np.where(Abb > 0, Abb, 1.0), np.nan)   # paired, per market
    figure(3, d3, np.nanmean(Lm, -1), Lm, 'noise', 'Price distortion\nof the major (%)', '#8b3a1f', 100.0)
    ix, iy, gxy = ref_index(d3)
    dJ = d3['r0'][:, ix, iy] - d3['r0_blind'][:, ix, iy]          # (L, S)
    hat = dJ.mean(0)
    summary['fig3'] = {'grid_point': gxy, 'L_ref': float(Lm[:, ix, iy].mean()),
                       'L_ref_q': q3(Lm[:, ix, iy].mean(0)),
                       'delta_J0_ref': float(hat.mean()),
                       'delta_J0_ref_se': float(hat.std(ddof=1) / np.sqrt(hat.size)),
                       'L_sign_change': {'positive_frac': float((Lm.mean((0, -1)) > 0).mean())}}
    bj = []
    for s in np.atleast_1d(d3['seeds']):
        f = os.path.join(str(s), 'counterfactual', 'blind_major.json')
        if os.path.exists(f):
            bj.append(json.load(open(f))['check_prior_block'])
    summary['fig3']['blind_major_check'] = bj

d4 = load(4)
if d4 is not None:
    A = d4['A'].astype(np.float64)
    assert (A >= 0).all()
    pos = (A > 0).all(-1)                         # log returns undefined if a price is 0 (floor 0)
    r = np.diff(np.log(np.where(A > 0, A, 1.0)), axis=-1)
    T = A.shape[-1]
    sig = np.where(pos, np.sqrt(T) * r.std(-1, ddof=1), np.nan)   # (L, nx, ny, S)
    assert np.isfinite(sig[pos]).all(), 'sigma_A not finite'
    if (~pos).any():
        print(f"fig4: sigma_A undefined (a zero price) in {(~pos).sum()} of {pos.size} markets, dropped", flush=True)
    figure(4, d4, np.nanmean(sig, -1), sig, 'noise', 'Price volatility\n$\\sigma_A$ (annualised)', '#5b3f8c', 1.0)
    ix, iy, gxy = ref_index(d4)
    summary['fig4'] = {'grid_point': gxy, 'sigma_A_ref': float(sig[:, ix, iy].mean()),
                       'sigma_A_ref_q': q3(sig[:, ix, iy].mean(0))}

# ── Fig. 5: expected number of days on the floor, init_alloc x Afloor ────────
if d2 is not None:
    days = d2['on_floor'].astype(bool).mean(-1) * 365.0          # (L, nx, ny, S): tau x 365
    figure(5, d2, days.mean(-1), days, 'seed', 'Expected days\non the floor', '#1f5f8b', 1.0)
    ix, iy, gxy = ref_index(d2)
    summary['fig5'] = {'grid_point': gxy, 'days_on_floor_ref': float(days[:, ix, iy].mean()),
                       'days_on_floor_ref_q': q3(days[:, ix, iy].mean(0)),
                       'days_on_floor_grid': days.mean((0, -1)).tolist()}

# ── Fig. 6: kappa^2 p against the allocation, coloured by regime ─────────────
#   p = mass-weighted probability of being short at year end (major + minors).
#   Terminal penalty kappa^2 * shortage * A-bar, so a held allowance is worth
#   kappa^2 p A-bar: interior (unpinned) kappa^2 p ~ 1; ceiling > 1; floor < 1.
#   Regime of a market: floor if the floor binds on >= half of the days, ceiling
#   if the ceiling does, interior otherwise.
if d1 is not None and 'p_short' in d1:
    import matplotlib.pyplot as plt
    L_, nx, ny, S_ = d1['p_short'].shape
    alloc = np.broadcast_to(d1['x'][None, :, None, None], (L_, nx, ny, S_))
    kap = np.broadcast_to(d1['y'][None, None, :, None], (L_, nx, ny, S_))
    kp = kap * d1['p_short']
    tf, tc = d1['on_floor'].astype(bool).mean(-1), d1['on_ceil'].astype(bool).mean(-1)
    reg = np.where(tf >= 0.5, 0, np.where(tc >= 0.5, 2, 1))
    names = ('floor', 'interior', 'ceiling')
    cols = ('#1f5f8b', '#555555', '#b8452c')
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    rng = np.random.default_rng(0)
    jit = (d1['x'][1] - d1['x'][0]) * 0.35
    for r_ in (1, 0, 2):
        m = reg == r_
        axs[0].scatter(100 * (alloc[m] + rng.uniform(-jit, jit, m.sum())), kp[m], s=3, alpha=0.35,
                       color=cols[r_], lw=0, label=f'{names[r_]} ({m.mean():.0%})', rasterized=True)
    axs[0].axhline(1.0, color='k', lw=0.8, ls='--')
    axs[0].set_xlabel(AXIS_LABEL['init_alloc']); axs[0].set_ylabel(r'$\kappa^2 p$')
    axs[0].legend(frameon=False, markerscale=3, fontsize=7, loc='upper right')
    axs[0].set_title('one dot per market', fontsize=8, loc='left')
    # mean over the markets of each (allocation, kappa) cell, interior markets only
    cm = plt.get_cmap('viridis')
    for jy in range(ny):
        mi = reg[:, :, jy] == 1
        y = np.where(mi, kp[:, :, jy], np.nan)
        with np.errstate(all='ignore'):
            yy = np.nanmean(y, axis=(0, -1))
        axs[1].plot(100 * d1['x'], yy, '-o', ms=2, lw=0.8, color=cm(jy / max(ny - 1, 1)))
    axs[1].axhline(1.0, color='k', lw=0.8, ls='--')
    axs[1].set_xlabel(AXIS_LABEL['init_alloc'])
    axs[1].set_title(r'interior markets, mean per $\kappa^2$ (colour: %.1f$\to$%.1f)'
                     % (d1['y'][0], d1['y'][-1]), fontsize=8, loc='left')
    for a in axs:
        for sd in ('top', 'right'):
            a.spines[sd].set_visible(False)
    fig.tight_layout()
    ps.save(fig, os.path.join(D, 'fig6'), check=False)
    with np.errstate(all='ignore'):
        summary['fig6'] = {
            'regime_share': {names[r_]: float((reg == r_).mean()) for r_ in range(3)},
            'kappa2_p_median': {names[r_]: float(np.median(kp[reg == r_])) if (reg == r_).any() else None
                                for r_ in range(3)},
            'kappa2_p_q': {names[r_]: q3(kp[reg == r_]) if (reg == r_).any() else None for r_ in range(3)}}
    print(f"fig6: kappa^2 p by regime (median): {summary['fig6']['kappa2_p_median']}  "
          f"shares {summary['fig6']['regime_share']}", flush=True)

with open(os.path.join(D, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: {kk: v for kk, v in s.items() if not kk.endswith('_grid')} for k, s in summary.items()},
                 indent=1))
print(f"-> {D}", flush=True)
