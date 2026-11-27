"""Job 4 of RUNS_SETTING.md: the LP of the deterministic competitive market without a
major, per unit of mass of private producers; it depends on the coverage ratio c only.

    KETS_DETERMINISTIC=1 python3 lp_job4.py --out results_majmin/job4_lp [--T 100]

    maximize   sum_{k,t} (P_t - c^k) q^k_t - kappa1 sigma
    subject to sum_k q^k_t <= C^xi (= 100)                       every day t      (capacity)
               sum_{k,t} e^k q^k_t <= c T C^xi e^fuel + sigma    (cap; dual lambda in [0, kappa1])
(the planner's problem (P) of benchmark_lp.tex with job 3's fixed fine kappa1: the same
penalty as the FP run; lambda* = min(merit-order price, kappa1), so kappa1 > max lambda
leaves lambda* independent of kappa1, which is checked)

P_t is the deterministic good-price path (the OU recursion without noise, from the
code), c^k, e^k, C^xi the private producer's technologies (envs/base_params.py).
lambda* = the cap's dual from HiGHS, cross-checked by the merit order: E(lambda) is a
nonincreasing step function, lambda* = inf{lambda >= 0 : E(lambda) <= cap}; when the
cap sits exactly on a step, the dual set is the interval [lambda-, lambda+] bounding it.
Outputs per c: lambda* (or the interval), production per technology over the year,
emissions, profit; and [c_lo, c_hi] = the c range where the cap binds and lambda- >= 2
(twice the job-3 floor), which sets the job-3 prior.
"""
import os
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--T', type=int, default=100)
ap.add_argument('--c', type=float, nargs=3, default=[0.30, 1.00, 0.01], metavar=('LO', 'HI', 'STEP'))
ap.add_argument('--floor', type=float, default=0.0, help='job-3 floor (Assumption 1(iv): 0)')
ap.add_argument('--kappa1', type=float, nargs='+', default=[60.0, 70.0, 80.0],
                help='fixed fine of job 3 (Assumption 1(iii)); the LP includes the shortfall at this cost')
args = ap.parse_args()
assert os.environ.get('KETS_DETERMINISTIC') == '1', 'run with KETS_DETERMINISTIC=1 (deterministic good prices)'

import numpy as np
import jax
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import paper_style as ps
from envs.base_params import generate_prices_ou, PRIVATE_GENERATOR

T = args.T
P = np.asarray(generate_prices_ou(jax.random.PRNGKey(0), T), np.float64)[:T]      # day t uses P_t
assert np.allclose(P, np.asarray(generate_prices_ou(jax.random.PRNGKey(1), T))[:T]), 'P must be deterministic'
_, ef, eg, cf, cg, _, capxi, _, _ = PRIVATE_GENERATOR
e = np.array([ef, eg]); cost = np.array([cf, cg]); e_fuel = e.max()
TECH = ('fossil', 'gas')
os.makedirs(args.out, exist_ok=True)


def cap_of(c):
    return c * T * capxi * e_fuel


def solve_lp(c, kappa1):
    """HiGHS. Variables q[t, k] (t-major), then the shortfall sigma."""
    m = len(e)
    obj = np.concatenate([-(P[:, None] - cost[None, :]).ravel(), [kappa1]])
    A_ub = [np.concatenate([np.tile(e, T), [-1.0]])]
    b_ub = [cap_of(c)]
    for t in range(T):
        r = np.zeros(T * m + 1); r[t * m:(t + 1) * m] = 1.0
        A_ub.append(r); b_ub.append(capxi)
    res = linprog(obj, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=[(0, None)] * (T * m + 1), method='highs')
    assert res.status == 0, res.message
    q = res.x[:T * m].reshape(T, m)
    return {'lam': -float(res.ineqlin.marginals[0]), 'profit': -float(res.fun), 'q': q,
            'sigma': float(res.x[-1]), 'emissions': float((q * e).sum())}


def merit_E(lam):
    """Emissions of the merit order at price lam (ties: produce less, the right limit)."""
    mu = P[:, None] - cost[None, :] - lam * e[None, :]
    best = mu.argmax(axis=1)
    on = mu.max(axis=1) > 1e-12
    return float(np.where(on, capxi * e[best], 0.0).sum())


def breakpoints():
    pts = [0.0]
    for k in range(len(e)):
        pts += list((P - cost[k]) / e[k])                               # shutdown of k
        for k2 in range(len(e)):
            if k2 != k and e[k] != e[k2]:
                pts.append((cost[k2] - cost[k]) / (e[k] - e[k2]))      # switch k -> k2
    pts = np.unique(np.asarray(pts))
    return pts[pts >= 0]


BP = breakpoints()


def merit_order_fine(c, kappa1):
    """With the fine: lambda* = min(merit-order price, kappa1); the interval is capped at kappa1."""
    lam, (lo, hi), binds = merit_order(c)
    return min(lam, kappa1), (min(lo, kappa1), min(hi, kappa1)), binds


def merit_order(c):
    """lambda* = inf{lam >= 0 : E(lam) <= cap}, and the dual interval [lam-, lam+] (all lam
    with E(lam+) <= cap <= E(lam-)), from E on the breakpoints (E is constant between them)."""
    cap = cap_of(c)
    tol = 1e-9 * max(cap, 1.0)
    if merit_E(0.0) < cap - tol:                      # the cap does not bind
        return 0.0, (0.0, 0.0), False
    if merit_E(0.0) <= cap + tol:                     # exactly at full emissions: weakly binding,
        up = [b for b in BP if merit_E(b + 1e-9) < cap - tol]       # duals [0, first drop of E]
        return 0.0, (0.0, float(min(up)) if up else 0.0), False
    lo, hi = None, None
    # E just right of each breakpoint; E just left of each breakpoint
    for i, b in enumerate(BP):
        e_right = merit_E(b + 1e-9)
        e_left = merit_E(max(b - 1e-9, 0.0))
        if e_right <= cap + tol and e_left >= cap - tol:
            lo = b if lo is None else min(lo, b)
            hi = b if hi is None else max(hi, b)
    assert lo is not None, f'c={c}: no breakpoint brackets the cap'
    return lo, (lo, hi), True


rows = []
for c in np.round(np.arange(args.c[0], args.c[1] + 1e-9, args.c[2]), 6):
    per_k = {}
    for k1 in args.kappa1:                  # the fine must not move the solution (kappa1 > lambda*)
        lpk = solve_lp(c, k1)
        lmk, (lok, hik), _ = merit_order_fine(c, k1)
        assert lok - 1e-6 <= lpk['lam'] <= hik + 1e-6, (c, k1, lpk['lam'], lok, hik)
        assert lpk['sigma'] <= 1e-6, f'c={c}, kappa1={k1}: the planner pays the fine (sigma = {lpk["sigma"]})'
        per_k[k1] = lpk
    lp = per_k[args.kappa1[0]]
    assert all(abs(v['profit'] - lp['profit']) <= 1e-6 * abs(lp['profit']) for v in per_k.values()), \
        f'c={c}: the LP value depends on kappa1'
    lam_mo, (lm, lp_), binds = merit_order_fine(c, min(args.kappa1))
    ok = lm - 1e-6 <= lp['lam'] <= lp_ + 1e-6
    rows.append({'c': float(c), 'cap': cap_of(c), 'binds': binds, 'kappa1_checked': list(args.kappa1),
                 'lambda_highs_by_kappa1': {str(k1): v['lam'] for k1, v in per_k.items()},
                 'lambda_star': lam_mo, 'lambda_minus': lm, 'lambda_plus': lp_,
                 'lambda_highs': lp['lam'], 'highs_in_interval': bool(ok),
                 'production': {TECH[k]: float(lp['q'][:, k].sum()) for k in range(len(e))},
                 'fossil_days': float(lp['q'][:, 0].sum() / capxi),
                 'emissions': lp['emissions'], 'profit': lp['profit']})
    assert ok, f"c={c}: HiGHS dual {lp['lam']} outside the merit-order interval [{lm}, {lp_}]"
lam_lo = np.array([r['lambda_minus'] for r in rows])
cs = np.array([r['c'] for r in rows])
good = np.array([r['binds'] for r in rows]) & (lam_lo >= 2 * args.floor)
c_lo, c_hi = (float(cs[good].min()), float(cs[good].max())) if good.any() else (None, None)
contiguous = bool(good.any() and np.all(good[(cs >= c_lo) & (cs <= c_hi)]))
summary = {'T': T, 'P_first_last': [float(P[0]), float(P[-1])], 'P_path': P.tolist(),
           'penalty': 'fixed fine kappa1 (Assumption 1(iii))', 'kappa1': list(args.kappa1), 'tech': {'e': e.tolist(), 'cost': cost.tolist()},
           'C_xi': capxi, 'e_fuel': e_fuel, 'floor_job3': args.floor, 'c_lo': c_lo, 'c_hi': c_hi,
           'range_contiguous': contiguous,
           'intervals': int(sum(r['lambda_plus'] - r['lambda_minus'] > 1e-9 for r in rows)), 'rows': rows}
with open(os.path.join(args.out, 'lp_job4.json'), 'w') as f:
    json.dump(summary, f, indent=1)
with open(os.path.join(args.out, 'lp_job4.csv'), 'w') as f:
    f.write('c,binds,lambda_star,lambda_minus,lambda_plus,lambda_highs,fossil,gas,emissions,cap,profit\n')
    for r in rows:
        f.write(f"{r['c']:.2f},{int(r['binds'])},{r['lambda_star']:.6f},{r['lambda_minus']:.6f},{r['lambda_plus']:.6f},"
                f"{r['lambda_highs']:.6f},{r['production']['fossil']:.3f},{r['production']['gas']:.3f},"
                f"{r['emissions']:.3f},{r['cap']:.3f},{r['profit']:.3f}\n")

ps.apply()
fig, ax = ps.line_panel()
ax.fill_between(cs, lam_lo, [r['lambda_plus'] for r in rows], color='C3', alpha=0.3, lw=0, label=r'$[\lambda^-, \lambda^+]$')
ax.plot(cs, [r['lambda_highs'] for r in rows], '.', color='k', ms=2.5, label='HiGHS dual')
if c_lo is not None:
    ax.axvspan(c_lo, c_hi, color='0.9', zorder=0, label='job-3 range')
ax.set_xlabel('Coverage ratio $c$'); ax.set_ylabel(r'$\lambda^\star$ (USD/tCO$_2$)')
ax.legend(fontsize=6, frameon=False)
ps.save(fig, os.path.join(args.out, 'lambda_star'))
print(f"P_t: {P[0]:.2f} -> {P[-1]:.2f} | {len(rows)} values of c | cap binds for {sum(r['binds'] for r in rows)} | "
      f"dual intervals: {summary['intervals']} | HiGHS inside the merit-order interval: all", flush=True)
for r in rows[::10] + [rows[-1]]:
    print(f"  c={r['c']:.2f}: lambda* = {r['lambda_star']:.3f}  [{r['lambda_minus']:.3f}, {r['lambda_plus']:.3f}]  "
          f"HiGHS {r['lambda_highs']:.3f} | fossil days {r['fossil_days']:.1f}, gas {r['production']['gas']:.1f} | "
          f"emissions/cap {r['emissions'] / r['cap']:.4f} | profit {r['profit']:.0f}", flush=True)
print(f"fine kappa1 in {args.kappa1}: no shortfall at any c, the same value and lambda* at every kappa1", flush=True)
print(f"job-3 range (cap binds, lambda- >= {2 * args.floor:g}): c in [{c_lo}, {c_hi}] (contiguous {contiguous})", flush=True)
print(f"-> {args.out}", flush=True)
