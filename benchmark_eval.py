"""Exact validation of a fictitious-play run against the LP benchmark (benchmark_lp.tex,
validation protocol), on a grid of scenarios (init_alloc x kappa^1 x mm_share).

    python3 benchmark_eval.py --run-dir results_majmin/<benchmark run> [--res 8 --every 5]

The run must satisfy Assumption 1 (KETS_DETERMINISTIC=1, KETS_PENALTY=fixed,
KETS_NO_MAJOR=1, floor 0, KETS_CAP_F_MULT large enough); the settings are read back
from its progress.json. For every checkpoint m (mixture of slots 0..m, k = m + 1
components) and every grid scenario gamma:
  2. ONE deterministic market with one particle per component and type (equal
     weights, so no particle error): prices A^K_t, fill ratios rho^K_t, and the
     average rewards J^j (each component rolled out on the frozen path);
  3. the exact exploitability E^j = (best-response LP value) - J^j, relative to s_j,
     and the population value sum_j n_j E^j / s_j;
  6. the distance to the exact equilibrium: a*(gamma) from the planner,
     max_t |A^K_t - a*| relative to a* (to kappa^1 if a* = 0), |J^j - V^j_*| / s_j,
     the aggregate emissions against Theta.
At the last FP iteration K it also compares the run's own (trained, lower-bound)
estimate, policy_K against the mixture of slots 0..K-1, with the exact value on the
same market (protocol step 5).
Writes <run-dir>/benchmark/benchmark.npz and summary.json.
"""
import os
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
ap.add_argument('--out', default=None)
ap.add_argument('--res', type=int, default=8, help='grid points per axis for init_alloc and kappa^1')
ap.add_argument('--mm', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                help='mm_share values (multiples of 1/agent_per_policy = 0.05)')
ap.add_argument('--every', type=int, default=5, help='checkpoints every this many iterations (and K)')
args = ap.parse_args()

from majmin_eval_common import read_progress, set_env_from_progress
prog = read_progress(args.run_dir)
st = prog.get('setting')
assert st and st['deterministic'] and st['penalty'] == 'fixed' and st['no_major'], \
    'not an LP-benchmark run (progress.json "setting")'
set_env_from_progress(prog, prog['floor_max'])
os.environ['KETS_CLEAR_ITERS'] = str(prog.get('clear_iters', 40))

import json
import numpy as np
import jax
import jax.numpy as jnp
from envs.environment import N_TYPES
from envs.environment_majmin import MINORS, MAJOR
from envs.models_majmin import component
from utils import norm_gamma, _atomic_json
from utils_majmin import load_net, make_sim, make_buffers
from majmin_eval_common import make_env
from benchmark_lp import market_data, planner, best_response_lp, p_days, equilibrium_paths

T, K = int(prog['T_grid'][0]), len(prog['records'])
box = prog['gamma_box']
out_dir = args.out or os.path.join(args.run_dir, 'benchmark')
os.makedirs(out_dir, exist_ok=True)
env = make_env(T)
floor = float(box['Afloor'][0])
assert box['Afloor'][0] == box['Afloor'][1] == 0.0, 'Assumption 1(iv): floor 0'
# the cap allows init_alloc * T full fossil days; an integer number makes a* non-unique
# (Phi flat between two shutdown prices), so the grid uses half-integers: (n + 1/2) / T
al = (np.floor(np.linspace(*box['init_alloc'], args.res) * T - 0.5).clip(min=np.ceil(box['init_alloc'][0] * T - 0.5)) + 0.5) / T
ka = np.linspace(*box['kappa'], args.res)
G = np.array([(a, k, floor, mm) for mm in args.mm for a in al for k in ka], np.float32)
N = len(G)
ckpts = sorted(set(list(range(args.every, K + 1, args.every)) + [K]))
print(f"LP benchmark | {args.run_dir} | T={T} K={K} | grid {args.res}x{args.res}x{len(args.mm)} = {N} "
      f"scenarios | checkpoints {ckpts} | devices {jax.devices()}", flush=True)

nets = [[load_net(args.run_dir, i, j) for j in range(N_TYPES)] for i in range(K + 1)]
P = jnp.asarray(env.generate_P(jax.random.PRNGKey(0), T), jnp.float32)   # deterministic
Gj = jnp.asarray(G)

# ── the planner, once per scenario ───────────────────────────────────────────
t0 = time.time()
mkts = [market_data(env, tuple(g)) for g in G]
plans = [planner(m) for m in mkts]
a_star = np.array([pl['a_star'] for pl in plans])
regime = np.array([pl['regime'] for pl in plans])
unique = np.array([pl['unique'] for pl in plans])
Theta = np.array([m['Theta'] for m in mkts])
V_star = np.array([[pl['V_star'].get(j, np.nan) for j in MINORS] for pl in plans])
scales = np.array([float(env.reward_scale(j)) for j in MINORS])
wts = np.array([[next((d['w'] for d in m['types'] if d['j'] == j), 0.0) for j in MINORS] for m in mkts])
print(f"planner: {time.time() - t0:.1f}s | regimes " +
      ", ".join(f"{r} {np.mean(regime == r):.0%}" for r in ('cap_slack', 'interior', 'fine')) +
      f" | non-unique a*: {np.sum(~unique)}", flush=True)


def simulate(k, slots):
    """One market per scenario with the mixture of `slots` (k components, one particle each)."""
    bufs, stats = make_buffers([nets[i] for i in slots])
    mstat = tuple(stats[j] for j in MINORS)
    sim = make_sim(env, mstat, stats[MAJOR])
    mbuf = tuple(bufs[j] for j in MINORS)
    zeros_n = jnp.zeros((k, T), jnp.float32)

    @jax.jit
    def run(mbuf_, jbuf_):
        return jax.vmap(lambda g: sim(jbuf_, mbuf_, jnp.int32(k), g, P, jnp.zeros(T), jnp.zeros(T),
                                      (zeros_n, zeros_n), None, 0, False))(Gj)
    A, rho, H, _, ys, (Xf, _) = run(mbuf, bufs[MAJOR])

    J = np.zeros((N, len(MINORS)))
    for a, j in enumerate(MINORS):
        @jax.jit
        def per_u(buf, u, A_, rho_, H_):
            net = component(buf, mstat[a], u)
            return jax.vmap(lambda g, aa, rr, hh: env.bind(g).rollout_minor(
                net, j, aa, rr, hh, P, jnp.zeros((1, T)), norm_gamma(g))[0])(Gj, A_, rho_, H_)
        J[:, a] = np.mean([np.asarray(per_u(mbuf[a], jnp.int32(u), A, rho, H)) for u in range(k)], axis=0)
    idx = env.dic_indexes
    em = lambda X: np.asarray(jnp.sum(X[..., idx['xi']] * X[..., idx['e']], axis=-1) + X[..., idx['eps']])
    E_type = np.stack([em(Xf[a]).mean(axis=1) for a in range(len(MINORS))], axis=1)     # (N, 2)
    traj = {'theta': np.asarray(ys['theta_minor']), 'q': np.asarray(ys['q_minor'])}     # (N,T,2), (N,T,2,2)
    return np.asarray(A)[:, 1:], np.asarray(rho)[:, 1:], J, E_type, mbuf, mstat, (A, rho, H), traj


def exact_gaps(A, rho, J):
    """Exact exploitability per type (N, 2), by the best-response LP on each market."""
    BR = np.full((N, len(MINORS)), np.nan)
    for s in range(N):
        for d in mkts[s]['types']:
            a = MINORS.index(d['j'])
            BR[s, a] = best_response_lp(mkts[s], d, A[s].astype(np.float64), rho[s, :, 0].astype(np.float64),
                                        rho[s, :, 1].astype(np.float64))
    return BR, BR - J


res = {k_: [] for k_ in ('A', 'rho', 'J', 'BR', 'E', 'E_rel', 'E_pop', 'price_err', 'price_err_mean',
                         'J_err_rel', 'emis', 'emis_rel_err')}
for m in ckpts:
    t1 = time.time()
    k = m + 1
    A, rho, J, E_type, _, _, _, traj = simulate(k, list(range(k)))
    BR, E = exact_gaps(A, rho, J)
    E_rel = E / scales[None, :]
    E_pop = np.nansum(wts * E_rel, axis=1)
    ref = np.where(a_star > 1e-9, a_star, G[:, 1])
    perr = np.abs(A - a_star[:, None]).max(axis=1) / ref
    perr_m = np.abs(A - a_star[:, None]).mean(axis=1) / ref
    J_err = np.abs(J - V_star) / scales[None, :]
    emis = (wts * E_type).sum(axis=1)
    emis_err = (emis - Theta) / Theta
    for k_, v in zip(res, (A, rho, J, BR, E, E_rel, E_pop, perr, perr_m, J_err, emis, emis_err)):
        res[k_].append(v)
    print(f"[m={m:3d}] exact exploitability: pop mean {E_pop.mean():+.3%} max {E_pop.max():+.3%} | "
          f"per type mean {np.nanmean(E_rel[:, 0]):+.3%} / {np.nanmean(E_rel[:, 1]):+.3%} | "
          f"price error max_t|A-a*|/a*: mean {perr.mean():.3f} median {np.median(perr):.3f} | "
          f"|J-V*|/s mean {np.nanmean(J_err):.3%} | emissions/Theta-1 (interior) "
          f"{np.mean(emis_err[regime == 'interior']) if np.any(regime == 'interior') else float('nan'):+.3%}  "
          f"[{time.time() - t1:.0f}s]", flush=True)
    assert np.all(np.isfinite(E_rel[wts > 0])), 'NaN exploitability'
    assert np.all(E[wts > 0] >= -1e-3 * np.abs(BR[wts > 0]) - 1.0), \
        'exact exploitability negative beyond tolerance: the LP and the simulation disagree'

# ── protocol step 5: the run's trained estimate vs the exact value at iteration K ──
A_K, rho_K, J_K, _, mbuf, mstat, (Aj, rhoj, Hj), _ = simulate(K, list(range(K)))
BR_K, E_K = exact_gaps(A_K, rho_K, J_K)
est = np.zeros((N, len(MINORS)))
for a, j in enumerate(MINORS):
    net = nets[K][j]
    f = jax.jit(jax.vmap(lambda g, aa, rr, hh: env.bind(g).rollout_minor(
        net, j, aa, rr, hh, P, jnp.zeros((1, T)), norm_gamma(g))[0]))
    est[:, a] = np.asarray(f(Gj, Aj, rhoj, Hj)) - J_K[:, a]
print(f"[iteration K={K}] trained estimate vs exact (relative, mean over the grid): "
      f"{np.nanmean(est[:, 0] / scales[0]):+.3%} vs {np.nanmean(E_K[:, 0] / scales[0]):+.3%} (type 0), "
      f"{np.nanmean(est[:, 1] / scales[1]):+.3%} vs {np.nanmean(E_K[:, 1] / scales[1]):+.3%} (type 2)",
      flush=True)

# ── trajectories: the LP equilibrium (Theorem 1 (i)) and the FP mixture at the last checkpoint ──
lp_q = np.full((N, len(MINORS), T, 2), np.nan)
lp_E = np.full((N, len(MINORS), T + 1), np.nan)
lp_th = np.full((N, len(MINORS), T + 1), np.nan)
for s_, mk in enumerate(mkts):
    for j, pth in equilibrium_paths(mk).items():
        a = MINORS.index(j)
        lp_q[s_, a], lp_E[s_, a], lp_th[s_, a] = pth['q'], pth['E'], pth['theta']
ap_ = np.asarray(env.agent_params)
e_ft = np.stack([ap_[j, 1:3] for j in MINORS])                                   # (2 types, fossil/gas)
fp_q = traj['q']                                                                 # (N, T, 2 types, 2 techs)
fp_E = np.concatenate([np.zeros((N, 1, len(MINORS))), np.cumsum((fp_q * e_ft[None, None]).sum(-1), axis=1)], 1)
fp_th = np.concatenate([np.broadcast_to(np.array([float(env.bind(jnp.asarray(g)).type_initial_state(j)[1])
                                                   for j in MINORS])[None], (1, len(MINORS)))
                        for g in G])[:, None, :]
fp_th = np.concatenate([fp_th, traj['theta']], axis=1)                           # (N, T+1, 2)
arr = {k_: np.stack(v) for k_, v in res.items()}
np.savez_compressed(os.path.join(out_dir, 'benchmark.npz'), G=G, ckpts=np.array(ckpts), a_star=a_star,
                    regime=regime, unique=unique, Theta=Theta, V_star=V_star, scales=scales, wts=wts,
                    p=np.asarray(P), est_K=est, lp_q=lp_q, lp_E=lp_E, lp_theta=lp_th,
                    fp_q=fp_q, fp_E=fp_E, fp_theta=fp_th, traj_m=ckpts[-1], exact_K=E_K, BR_K=BR_K, J_K=J_K, **arr)
last = -1
summ = {'run': args.run_dir, 'T': T, 'K': K, 'grid': {'init_alloc': al.tolist(), 'kappa1': ka.tolist(),
                                                      'mm_share': args.mm},
        'regimes': {r: int(np.sum(regime == r)) for r in ('cap_slack', 'interior', 'fine')},
        'non_unique_a_star': int(np.sum(~unique)),
        'by_checkpoint': [{'m': int(m), 'E_pop_mean': float(arr['E_pop'][i].mean()),
                           'E_pop_max': float(arr['E_pop'][i].max()),
                           'E_rel_mean': [float(np.nanmean(arr['E_rel'][i][:, a])) for a in range(2)],
                           'price_err_mean': float(arr['price_err'][i].mean()),
                           'price_err_median': float(np.median(arr['price_err'][i])),
                           'J_err_rel_mean': [float(np.nanmean(arr['J_err_rel'][i][:, a])) for a in range(2)]}
                          for i, m in enumerate(ckpts)],
        'final_by_regime': {r: {'E_pop_mean': float(arr['E_pop'][last][regime == r].mean()),
                                'price_err_mean': float(arr['price_err'][last][regime == r].mean())}
                            for r in ('cap_slack', 'interior', 'fine') if np.any(regime == r)},
        'estimate_vs_exact_K': {'estimate_rel_mean': [float(np.nanmean(est[:, a] / scales[a])) for a in range(2)],
                                'exact_rel_mean': [float(np.nanmean(E_K[:, a] / scales[a])) for a in range(2)]}}
_atomic_json(os.path.join(out_dir, 'summary.json'), summ)
print(f"-> {out_dir}", flush=True)
