"""Mixture price-path convergence at the audit scenarios.

The training loop records ||A^k - A^{k-1}||_2 (mean over a frozen block) for
its own evaluation blocks only. The PMFG audit does not save price paths, so
this recomputes the same quantity at each audit scenario context, for every
checkpoint 1..K, on ONE frozen block per scenario:

    A^k   = mixture price path under triples 0..k-1
    l2[k] = mean_s ||A^{k+1}_s - A^k_s||_2        recorded at iteration k

Checkpoint k here means the same as in progress.json and audit_pmfg.py: the
mixture is triples 0..k-1.

    python3 audit_paths.py --run-dir results_floor20/choice

Writes <run_dir>/audit/<scenario>/paths.json for each scenario.
"""

import os
import sys
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _floor_max_of_run(argv):
    """Same pre-import step as audit_pmfg.py: the FiLM normaliser must use the
    box the run was trained on, and base_params reads it at import."""
    run_dir = None
    for i, a in enumerate(argv):
        if a == '--run-dir' and i + 1 < len(argv):
            run_dir = argv[i + 1]
        elif a.startswith('--run-dir='):
            run_dir = a.split('=', 1)[1]
    if run_dir is None:
        return None
    path = os.path.join(run_dir, 'progress.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        box = json.load(f).get('context_box')
    return float(box[2][1]) if box else None


_RUN_FLOOR_MAX = _floor_max_of_run(sys.argv)
if _RUN_FLOOR_MAX is not None:
    os.environ['KETS_FLOOR_MAX'] = repr(_RUN_FLOOR_MAX)

import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import stack_policies
from utils import (_hms, _atomic_json, chunked, draw_common_noise, draw_idio,
                   load_triple_r3, bind_env, make_price_fn_r3,
                   CTX_NAMES, CONTEXT_DIM, VERIFY_SEED_R3, _G_HI)
from audit_pmfg import audit_scenarios

# one fixed offset per scenario, so the block is the same in every process
# (audit_pmfg.py uses hash(scenario), which Python randomises per run)
SCEN_SEED = {'current_kets': 1, 'low_alloc': 5, 'high_tax': 6,
             'no_market_maker': 3, 'high_floor_mid_alloc': 4}


def main():
    p = argparse.ArgumentParser(description='mixture price-path convergence '
                                            'at the audit scenarios')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--scenarios', nargs='+', default=list(SCEN_SEED))
    p.add_argument('--samples', type=int, default=250,
                   help='common-noise draws in the frozen block')
    p.add_argument('--eval-chunk', type=int, default=50)
    args = p.parse_args()

    with open(os.path.join(args.run_dir, 'progress.json')) as f:
        prog = json.load(f)
    n_avail = 0
    while os.path.exists(os.path.join(args.run_dir,
                                      f'policy_{n_avail}_type0.eqx')):
        n_avail += 1
    box = prog.get('context_box')
    floor_max = float(box[2][1]) if box else FLOOR_MAX
    if abs(float(_G_HI[2]) - floor_max) > 1e-6:
        raise SystemExit(f"FiLM normaliser box top {float(_G_HI[2])} != run "
                         f"box top {floor_max}")
    scen = audit_scenarios(floor_max)

    T = int(prog['T_grid'][-1])
    mode = prog.get('mode', 'choice')
    agents = int(prog.get('agent_per_policy', 20))
    env = ExogenousMarketEnvJAX(
        kappa=GAMMA_REF[1], T=T,
        generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
        market_impact_func=market_impact_base,
        generate_eps0_func=white_noise_A_base,
        generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
        A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=agents)
    if mode == 'bau':
        bp, bq = list(PRIVATE_GENERATOR), list(BIG_PUBLIC_GENERATOR)
        bp[8], bq[8] = TEC_PRIVATE, TEC_PUBLIC
        env = eqx.tree_at(lambda e: e.agent_params, env,
                          jnp.array([bp, bq, MARKET_MAKER]))

    print("=" * 74, flush=True)
    print(f"PATH CONVERGENCE  |  {args.run_dir}  |  {n_avail} triples, T={T}")
    print(f"floor box top {floor_max}   block S = {args.samples}")
    print("=" * 74, flush=True)

    history = [load_triple_r3(args.run_dir, i) for i in range(n_avail)]
    statics = tuple(stack_policies([history[0][j]])[1] for j in range(N_TYPES))
    price_fn = make_price_fn_r3(env, statics)

    for name in args.scenarios:
        c = scen[name]
        ca = jnp.asarray(c, dtype=jnp.float32)
        env_c = bind_env(env, ca)
        kc, kp = jax.random.split(jax.random.PRNGKey(
            VERIFY_SEED_R3 + 7919 * SCEN_SEED[name]), 2)
        P, eps0 = draw_common_noise(env_c, T, args.samples, kc)
        idio_pop = draw_idio(env_c, T, env_c.type_counts, kp)
        ctx_rows = jnp.broadcast_to(ca, (args.samples, CONTEXT_DIM))

        t0 = time.time()
        A_prev, l2, a_mean = None, [], []
        for k in range(1, n_avail + 1):
            stacked = tuple(stack_policies([h[j] for h in history[:k]])[0]
                            for j in range(N_TYPES))
            A = chunked(lambda a, b: price_fn(stacked, ctx_rows[a:b], P[a:b],
                                              eps0[a:b], idio_pop),
                        args.samples, args.eval_chunk)
            A = np.asarray(A)
            a_mean.append(A.mean(axis=0).tolist())
            if A_prev is not None:
                l2.append(float(np.mean(np.linalg.norm(A - A_prev, axis=1))))
            A_prev = A
        # l2[i] is the move from mixture i+1 (triples 0..i) to mixture i+2, so
        # it belongs to iteration i+1 exactly like the training records
        out = {
            'run_dir': args.run_dir, 'scenario': name,
            'context': [float(x) for x in c],
            'context_names': list(CTX_NAMES), 'floor_max': floor_max,
            'S': args.samples, 'T': T,
            'iterations': list(range(1, n_avail)),
            'path_l2': l2,
            'A_mean_final': a_mean[-1],
        }
        out_dir = os.path.join(args.run_dir, 'audit', name)
        os.makedirs(out_dir, exist_ok=True)
        _atomic_json(os.path.join(out_dir, 'paths.json'), out)
        print(f"  {name:<22s} {_hms(time.time() - t0)}   "
              f"l2: {l2[0]:.4f} -> {l2[-1]:.4f}   -> {out_dir}/paths.json",
              flush=True)


if __name__ == '__main__':
    main()
