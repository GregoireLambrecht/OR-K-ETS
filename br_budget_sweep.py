"""Best-response budget sweep on a finished rung-2 (PMFG, market clearing) run.

For each chosen FP iteration k, rebuild the exact mixture that iteration's BR
faced (triples 0..k-1), retrain the three best responses with several step
budgets and seeds, and evaluate them on the SAME frozen verification set as
the FP loop (VERIFY_SEED), so the gaps are directly comparable with the
progress.json values.

Seed 0 reuses the original BR key of iteration k (fold_in(k_iter_base, k-1)),
so seed 0 at the original budget should reproduce the logged gap.

    python3 br_budget_sweep.py --run-dir results_mc_pmfg/current_kets_T30_K30_seed42 \
        --k 28 --iters 4000,6000,8000,10000 --seeds 3

Writes <run-dir>/br_sweep/k<k>.json (rewritten after every BR, so partial
results survive a timeout).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import TypePolicyNN, stack_policies
from utils import (load_triple, make_verification_set, make_batched_return_fn,
                   make_batched_hist_fn, make_br_stage_fn_r2, train_best_response_r2,
                   exploitability_r2, make_price_fn_batched)

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
ap.add_argument('--k', type=int, required=True, help='FP iteration (1-based, as logged)')
ap.add_argument('--iters', default='4000,6000,8000,10000')
ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--lr-min', type=float, default=1e-4)
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--verify-samples', type=int, default=200)
ap.add_argument('--eval-samples', type=int, default=32)
ap.add_argument('--eval-chunk', type=int, default=50)
args = ap.parse_args()

with open(os.path.join(args.run_dir, 'progress.json')) as f:
    prog = json.load(f)
ctx = tuple(prog['context'])
run_seed = int(prog['seed'])
T = int(prog['T_grid'][-1])
orig = next(r for r in prog['records'] if r['iteration'] == args.k)

env = ExogenousMarketEnvJAX(
    kappa=GAMMA_REF[1], T=T,
    generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
    market_impact_func=market_impact_base,
    generate_eps0_func=supply_noise_base,
    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE,
    agent_per_policy=int(prog['agent_per_policy']),
)
env = env.set_context(jnp.asarray(ctx, dtype=jnp.float32)).set_T(T)
ver = make_verification_set(env, T, args.verify_samples, args.eval_samples)

history = [load_triple(args.run_dir, i) for i in range(args.k)]   # triples 0..k-1
stacked = tuple(stack_policies([h[j] for h in history])[0] for j in range(N_TYPES))
_, model_static = eqx.partition(history[0][0], eqx.is_array)
statics = tuple(stack_policies([history[0][j]])[1] for j in range(N_TYPES))
price_fn = make_price_fn_batched(env, statics)
S, ch = args.verify_samples, args.eval_chunk
A_ver = jnp.concatenate([price_fn(stacked, ver['P'][a:a + ch], ver['eps0'][a:a + ch],
                                  ver['idio_pop']) for a in range(0, S, ch)])
ret_fns = [make_batched_return_fn(env, j) for j in range(N_TYPES)]
hist_fns = [make_batched_hist_fn(env, j, model_static) for j in range(N_TYPES)]
mm_warm = eqx.tree_deserialise_leaves(os.path.join(args.run_dir, 'mm_warmstart_pmfg.eqx'),
                                      TypePolicyNN(key=jax.random.PRNGKey(0)))

# same key derivation as fictitious_play_rung2
_, k_iter_base = jax.random.split(jax.random.PRNGKey(run_seed))
budgets = [int(x) for x in args.iters.split(',')]
shares = env.type_shares

out_dir = os.path.join(args.run_dir, 'br_sweep')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'k{args.k:03d}.json')
res = {'k': args.k, 'context': list(ctx), 'T': T, 'budgets': budgets,
       'original': {'iters': prog['iters'],
                    'per_type': [{kk: p[kk] for kk in ('gap', 'gap_se', 'gap_relative', 'V_br', 'V_mix')}
                                 for p in orig['per_type']],
                    'weighted_relative': orig['weighted_relative']},
       'runs': []}
print(f"k={args.k}  mixture of {len(history)} triples  ctx {ctx}  T={T}", flush=True)
print("original: " + "  ".join(f"t{j} {p['gap_relative']*100:+.2f}%"
                               for j, p in enumerate(orig['per_type'])), flush=True)

for n in budgets:
    opt = optax.adam(optax.cosine_decay_schedule(
        init_value=args.lr, decay_steps=n, alpha=args.lr_min / args.lr))
    stage_fns = {T: make_br_stage_fn_r2(env, statics, opt, n, args.batch, model_static)}
    for s in range(args.seeds):
        key = jax.random.fold_in(k_iter_base, args.k - 1)
        if s > 0:
            key = jax.random.fold_in(key, 1000 + s)
        t0 = time.time()
        triple, loss_hist = train_best_response_r2(stacked, key, stage_fns, opt, mm_warm,
                                                   (T,), (n,), (args.lr,), verbose=False)
        secs = time.time() - t0
        L = np.asarray(loss_hist[0])                       # (n, 3)
        per = [exploitability_r2(env, j, triple[j], history, A_ver, ver['P'],
                                 ver['idio_eval'][j], ret_fns[j], hist_fns[j], model_static)
               for j in range(N_TYPES)]
        wrel = float(sum(shares[j] * per[j]['gap_relative'] for j in range(N_TYPES)))
        tail = max(1, n // 20)
        res['runs'].append({
            'iters': n, 'seed': s, 'seconds': secs, 'weighted_relative': wrel,
            'loss_end': L[-tail:].mean(0).tolist(),
            'loss_curve': L[::max(1, n // 200)].tolist(),
            'per_type': [{kk: p[kk] for kk in ('gap', 'gap_se', 'gap_relative', 'V_br', 'V_mix')}
                         for p in per]})
        print(f"iters {n:6d} seed {s}  {secs:6.0f}s  " +
              "  ".join(f"t{j} E {p['gap_relative']*100:+7.2f}% (loss {L[-tail:, j].mean():+.4f})"
                        for j, p in enumerate(per)) + f"  wtd {wrel*100:+.2f}%", flush=True)
        with open(out_path, 'w') as f:
            json.dump(res, f)
print(f"-> {out_path}", flush=True)
