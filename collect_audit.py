"""Assemble the PMFG audit cells into tables and figures.

    python3 collect_audit.py results_floor20/choice results_floor20/bau ...

Reads every <run_dir>/audit/<scenario>/b<budget>/k<NNN>/result.json and answers
two questions:

  1. What is the TRUE exploitability of the BE mixture at each policy scenario?
     The training loop measures against a deviator from the same conditional
     family; this measures against a free specialist.
  2. What does generality cost? E_PMFG - E_BE at equal gradient budget.

Scenario 4 (high_floor_mid_alloc) is defined relative to each run's own context
box, so it is a DIFFERENT market in the floor20 / floor30 / floor40 studies. The
tables mark it, and the cross-run figure omits it.
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
                         SURFACE, ZERO_REF, LW_MIX, _style, _save)

SCEN_ORDER = ('current_kets', 'low_alloc', 'high_tax', 'no_market_maker',
              'high_floor_mid_alloc')
SCEN_LABEL = {'current_kets': 'Current K-ETS',
              'low_alloc': 'Low allocation',
              'high_tax': 'High tax',
              'no_market_maker': 'No market maker',
              'high_floor_mid_alloc': 'High floor, mid alloc*'}
RELATIVE = {'high_floor_mid_alloc'}      # not comparable across runs
MODE_LS = {'choice': '-', 'bau': (0, (5, 2.4))}


def load(run_dir):
    cells = {}
    for f in sorted(glob.glob(os.path.join(run_dir, 'audit', '*', 'b*', 'k*',
                                           'result.json'))):
        with open(f) as fh:
            r = json.load(fh)
        r['_loss'] = os.path.join(os.path.dirname(f), 'loss.npz')
        cells.setdefault(r['scenario'], []).append(r)
    for v in cells.values():
        v.sort(key=lambda r: (r['iters'], r['checkpoint']))
    return {'dir': run_dir.rstrip('/'), 'cells': cells}


runs = [load(d) for d in sys.argv[1:]]
runs = [r for r in runs if r['cells']]
if not runs:
    raise SystemExit("no audit cells found; run run_audit_pmfg.sh first\n"
                     + __doc__)

for R in runs:
    print("\n" + "=" * 96)
    print(f"{R['dir']}")
    print("=" * 96)
    print(f"\nTRUE EXPLOITABILITY AT EACH POLICY SCENARIO  (population weighted)")
    print(f"{'scenario':<24}{'ckpt':>6}{'budget':>8}{'E_PMFG':>11}{'E_BE':>11}"
          f"{'price of gen':>15}{'context':>34}")
    print("-" * 109)
    for s in SCEN_ORDER:
        for r in R['cells'].get(s, []):
            ctx = ", ".join(f"{v:g}" for v in r['context'])
            print(f"{SCEN_LABEL[s]:<24}{r['checkpoint']:>6}{r['iters']:>8}"
                  f"{r['weighted_pmfg']:>11.2%}{r['weighted_be']:>11.2%}"
                  f"{r['price_of_generality']:>15.2%}{'(' + ctx + ')':>34}")

    print(f"\nPER TYPE, at the largest budget on file")
    print(f"{'scenario':<24}{'type':<15}{'E_PMFG':>11}{'E_BE':>11}"
          f"{'price of gen':>15}{'V_br PMFG':>12}")
    print("-" * 88)
    for s in SCEN_ORDER:
        v = R['cells'].get(s, [])
        if not v:
            continue
        r = v[-1]
        for j, m in enumerate(r['per_type']):
            lab = SCEN_LABEL[s] if j == 0 else ""
            print(f"{lab:<24}{TYPE_NAMES[j]:<15}"
                  f"{m['pmfg']['gap_relative']:>11.2%}"
                  f"{m['be']['gap_relative']:>11.2%}"
                  f"{m['price_of_generality']:>15.2%}"
                  f"{m['pmfg']['V_br_relative']:>12.2%}")

    neg = [(r['scenario'], j) for v in R['cells'].values() for r in v
           for j in range(3)
           if r['per_type'][j]['pmfg']['gap']
           < -r['per_type'][j]['pmfg']['gap_se']]
    tot = sum(len(v) for v in R['cells'].values()) * 3
    print(f"\nUNDER-TRAINED AUDIT BR: {len(neg)}/{tot} (cell, type) pairs have "
          f"E_PMFG < -SE")
    if neg:
        print("  a negative gap is impossible for an exact best response — the "
              "audit budget is too small at: "
              + ", ".join(f"{s}/{TYPE_NAMES[j]}" for s, j in neg[:6]))

# ── figure: exploitability and the price of generality, by scenario ─────────
comparable = [s for s in SCEN_ORDER if s not in RELATIVE]
fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.6), facecolor=SURFACE)
x = np.arange(len(comparable))
w = 0.8 / max(len(runs), 1)

# colour = floor study, hatch = mode: six runs would otherwise cycle through
# three colours and choice/bau of different studies become indistinguishable.
studies = sorted({os.path.basename(os.path.dirname(R['dir'].rstrip('/')))
                  for R in runs})
for i, R in enumerate(runs):
    mode = 'bau' if R['dir'].endswith('bau') else 'choice'
    study = os.path.basename(os.path.dirname(R['dir'].rstrip('/')))
    col = TYPE_COLORS[studies.index(study) % len(TYPE_COLORS)]
    hatch = '///' if mode == 'bau' else None
    ep = [R['cells'][s][-1]['weighted_pmfg'] * 100 if R['cells'].get(s) else np.nan
          for s in comparable]
    pg = [R['cells'][s][-1]['price_of_generality'] * 100 if R['cells'].get(s) else np.nan
          for s in comparable]
    axes[0].bar(x - 0.4 + w * (i + 0.5), ep, w * 0.92, color=col,
                edgecolor='#ffffff', linewidth=2.0, hatch=hatch,
                label=f"{study.replace('results_', '')} {mode}")
    axes[1].bar(x - 0.4 + w * (i + 0.5), pg, w * 0.92, color=col,
                edgecolor='#ffffff', linewidth=2.0, hatch=hatch)

for ax, ttl, yl in ((axes[0], 'PMFG exploitability of the BE mixture', '%'),
                    (axes[1], 'Price of generality  '
                              r'$\mathcal{E}_{PMFG}-\mathcal{E}_{BE}$', '%')):
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in comparable], rotation=12)
    ax.axhline(0, color=ZERO_REF, lw=1.0)
    ax.yaxis.grid(True, color='#e6e5e1', linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax, ttl, yl, None)

fig.legend(loc='lower center', ncol=min(len(runs), 3), frameon=False,
           fontsize=9.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("PMFG audit — the four policy scenarios "
             "(the relative-floor scenario is omitted: it is a different "
             "market in each run)",
             color=INK, fontsize=12.5, x=0.005, ha="left", y=0.99)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
print(f"\n-> {_save(fig, 'figs_audit/audit_scenarios.pdf')} (+ .png)")
