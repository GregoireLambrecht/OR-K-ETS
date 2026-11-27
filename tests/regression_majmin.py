"""Regression check of the major–minor evaluation against a finished run.

Rebuilds the frozen evaluation blocks of a run, re-evaluates its LAST FP
iteration from the saved networks (best response policy_K against the mixture
of slots 0..K-1) and compares with the numbers the solver wrote to
progress.json: per-type exploitabilities, V_br, V_mix and the cleared prices.
Any change to the environment, the clearing, the networks or the evaluation
shows up here.

    python3 tests/regression_majmin.py --run-dir results_majmin/<run> [--blocks prior ref]
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
ap.add_argument('--blocks', nargs='+', default=None, help="default: 'prior ref' (rung 3), 'ref' (rung 2)")
ap.add_argument('--floor-max', type=float, default=30.0, help="used only if the run did not record it")
ap.add_argument('--rtol', type=float, default=2e-3)
args = ap.parse_args()

from majmin_eval_common import read_progress, set_env_from_progress
prog = read_progress(args.run_dir)
set_env_from_progress(prog, prog.get('floor_max', args.floor_max))
os.environ['KETS_CLEAR_ITERS'] = str(prog.get('clear_iters', 40))

import numpy as np
import jax
from envs.environment import N_TYPES
from envs.environment_majmin import MINORS
from utils import CTX_REF
from utils_majmin import load_net, make_sim, make_blocks, make_evaluator, make_buffers
from majmin_eval_common import make_env

rung, T, n = int(prog['rung']), int(prog['T_grid'][0]), int(prog['n_sim'])
ctx_ref = tuple(prog.get('context', CTX_REF))
K = len(prog['records'])
blocks_to_check = args.blocks or (['prior', 'ref'] if rung == 3 else ['ref'])
env = make_env(T)
blocks, _ = make_blocks(env, rung, ctx_ref, T, n, int(prog['S']), int(prog.get('S_fixed', 100)), int(prog['M']))
nets = [[load_net(args.run_dir, i, j) for j in range(N_TYPES)] for i in range(K + 1)]
bufs, stats = make_buffers(nets[:K])
sim = make_sim(env, tuple(stats[j] for j in MINORS), stats[1])
evaluate = make_evaluator(env, sim, ctx_ref, bool(prog.get('major_price_taker', False)))
hist_minor = {j: [tr[j] for tr in nets[:K]] for j in MINORS}
rec = prog['records'][-1]
print(f"regression | {args.run_dir} | rung {rung} T={T} n={n} K={K} | devices {jax.devices()}", flush=True)

worst = 0.0
for nm in blocks_to_check:
    ref = rec['blocks'][nm] if rung == 3 else rec
    per_type, wrel, clr, _ = evaluate(blocks[nm], K, nets[K], tuple(bufs[j] for j in MINORS), bufs[1], hist_minor)
    rows = []
    for j in range(N_TYPES):
        for q in ('V_br', 'V_mix'):
            rows.append((f'type{j} {q}', ref['per_type'][j][q], per_type[j][q]))
        rows.append((f'type{j} gap_rel', ref['per_type'][j]['gap_relative'], per_type[j]['gap_relative']))
    rows.append(('weighted', ref['weighted_relative'], wrel))
    for q in ('A_cleared_mean', 'A0_cleared', 'A_on_floor'):
        rows.append((q, ref['clearing'][q], clr[q]))
    print(f"[{nm}]")
    for lab, a, b in rows:
        # values relative to their own size; exploitabilities (already relative) in absolute terms
        err = abs(b - a) if ('gap_rel' in lab or lab in ('weighted', 'A_on_floor')) else abs(b - a) / max(abs(a), 1.0)
        worst = max(worst, err)
        print(f"   {lab:16s} stored {a:+14.6g}  now {b:+14.6g}  err {err:.2e}{'  <-- MISMATCH' if err > args.rtol else ''}")
ok = worst <= args.rtol
print(f"REGRESSION {'PASSED' if ok else 'FAILED'}: worst error {worst:.2e} (tolerance {args.rtol:g})", flush=True)
sys.exit(0 if ok else 1)
