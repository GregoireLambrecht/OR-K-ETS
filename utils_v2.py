"""Rung 2 (fixed scenario, common noise) for market clearing VERSION 2.

Same fictitious play as utils.fictitious_play_rung2, with:
  * the v2 clearing (blocking corridor, two-sided rationing; no price-elastic supply): every
    price evaluation returns (A, rho) and the deviator executes with rho;
  * per-agent population sampling (N_SIM agents per type, stratified mixture
    components), the mixture held in a fixed K_MAX buffer passed with its
    current size k as a traced integer, so the price and best-response
    functions compile ONCE per solve and an FP iteration has constant cost;
  * n_restarts independent fresh best responses per type, best kept by
    training loss (utils._train_br_restarts), every network saved.
Exploitability keeps the mixture's value EXACT: every component is rolled out
on the frozen block (MARKET_CLEARING.md §1.8).
"""
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from envs.environment import N_TYPES
from envs.base_params import GAMMA_REF
from envs.params_v2 import (A_REF, A_CEIL_V2, SUPPLY_C, FLOOR_MODE, N_SIM, K_MAX,
                            BETA_MIN_V2, CLEAR_ITERS_V2)
from envs.models_v2 import TypePolicyNNV2, make_buffer, buffer_set
from envs.models import stack_policies
from utils import (_hms, _atomic_json, save_triple, draw_common_noise, draw_idio,
                   chunked_tree, exploitability_r2, _train_br_restarts,
                   save_br_candidates, VERIFY_SEED)

COMP_SEED = 31337        # frozen-block component assignment, folded with k


def load_triple_v2(save_dir, k):
    like = TypePolicyNNV2(key=jax.random.PRNGKey(0))
    return tuple(eqx.tree_deserialise_leaves(
        os.path.join(save_dir, f'policy_{k}_type{j}.eqx'), like) for j in range(N_TYPES))


def make_verification_set_v2(env, T, S, M, n, seed=VERIFY_SEED):
    """S frozen common-noise draws; population idiosyncratic draws (n, T) per
    type shared by the S markets (as in v1); paired evaluation draws (M, T)."""
    kc, kp, ke = jax.random.split(jax.random.PRNGKey(seed), 3)
    P, eps0 = draw_common_noise(env, T, S, kc)
    return {'P': P, 'eps0': eps0, 'S': S, 'M': M,
            'idio_pop': draw_idio(env, T, (n, n, n), kp),
            'idio_eval': draw_idio(env, T, (M, M, M), ke)}


def block_comp_keys(S, k):
    """Component-assignment keys of the frozen block at mixture size k."""
    return jax.random.split(jax.random.fold_in(jax.random.PRNGKey(COMP_SEED), k), S)


def make_price_fn_v2(env_T, statics, return_diag=False):
    """(buffers, k, comp_keys (S,), P (S,T+1), eps0 (S,T), idio_pop) -> A, rho [, diag]."""
    @eqx.filter_jit
    def fn(buffers, k, comp_keys, P, eps0, idio_pop):
        return jax.vmap(lambda ck, p, e: env_T.ensemble_price_path_v2(
            buffers, statics, k, ck, p, e, idio_pop, return_diag=return_diag)
        )(comp_keys, P, eps0)
    return fn


def make_batched_return_fn_v2(env_T, j):
    """policy, A (S,T+1), rho (S,T+1), P (S,T+1), idio (M,T) -> (S, M)."""
    @eqx.filter_jit
    def fn(policy, A, rho, P, idio):
        return jax.vmap(lambda a, r, p: env_T.rollout_agent_batch_v2(
            policy, j, a, r, p, idio)[1])(A, rho, P)
    return fn


def make_batched_hist_fn_v2(env_T, j, static):
    """Stacked history (L leading), A, rho, P, idio -> (L, S, M). Exact J^j."""
    @eqx.filter_jit
    def fn(stacked_params, A, rho, P, idio):
        def per_policy(prm):
            model = eqx.combine(prm, static)
            return jax.vmap(lambda a, r, p: env_T.rollout_agent_batch_v2(
                model, j, a, r, p, idio)[1])(A, rho, P)
        return jax.vmap(per_policy)(stacked_params)
    return fn


def clearing_metrics_v2(env, A, rho, diag):
    """MARKET_CLEARING.md §5 over a frozen block. A: (S, T+1), rho: (S, T+1, 2)."""
    w = np.asarray(env.context_weights())
    cap = float(np.sum(w * np.asarray(env.agent_params[:, 5])))
    A, rho = np.asarray(A), np.asarray(rho)
    Ac = A[:, 1:]
    d = {k: np.asarray(v) for k, v in diag.items()}
    fl, ce = d['on_floor'].astype(bool), d['on_ceil'].astype(bool)
    bound = fl | ce
    off = ~bound[:, 1:] & ~bound[:, :-1]
    drift = np.diff(Ac, axis=1)[off] if off.any() else np.zeros(1)
    G_floor = np.where(fl, d['G'], 0.0).sum(axis=1)
    return {
        'cap_per_step': cap,
        'A_on_floor': float(fl.mean()),
        'A_on_ceil': float(ce.mean()),
        'clear_flat_frac': float(d['flat'].mean()),
        'rho_sell_mean_on_floor': float(d['rho_sell'][fl].mean()) if fl.any() else 1.0,
        'rho_sell_min': float(d['rho_sell'].min()),
        'rho_buy_min': float(d['rho_buy'].min()),
        'blocked_volume': float(G_floor.mean()),
        'blocked_sell_type': [float(x) for x in d['blocked_sell_type'].sum(axis=1).mean(axis=0)],
        'blocked_buy_type': [float(x) for x in d['blocked_buy_type'].sum(axis=1).mean(axis=0)],
        'support_cost': float(np.asarray(env.Afloor) * G_floor.mean()),
        'clear_residual_mean_rel': float(d['resid'].mean() / cap),
        'clear_residual_max_rel': float(d['resid'].max() / cap),
        'cons_err_max_rel': float(np.abs(d['cons_err']).max() / cap),
        'price_drift_mean': float(drift.mean()),
        'price_drift_std': float(drift.std()),
        'A0_cleared': float(Ac[:, 0].mean()),
        'A_cleared_mean': float(Ac.mean()),
        'supply_abs_mean': float(np.abs(d['S']).mean()),
        'supply_exec_mean': float(d['S_exec'].mean()),
        'R_mean': [float(x) for x in d['R_mean'].mean(axis=(0, 1))],
        'R_median': [float(x) for x in np.median(d['R_median'], axis=(0, 1))],
        'beta_mean': [float(x) for x in d['beta_mean'].mean(axis=(0, 1))],
        'f_mean': [float(x) for x in d['f_mean'].mean(axis=(0, 1))],
        'market_short_frac': float((d['agg_gap_T'] > 0).mean()),
        'gap_T_mean': float(d['agg_gap_T'].mean()),
        'gap_T_type': [float(x) for x in d['gap_T_type'].mean(axis=0)],
        'short_T_type': [float(x) for x in d['short_T_type'].mean(axis=0)],
        'A_path_mean': [float(x) for x in A.mean(axis=0)],
        'A_path_p10': [float(x) for x in np.quantile(A, 0.1, axis=0)],
        'A_path_p90': [float(x) for x in np.quantile(A, 0.9, axis=0)],
        'rho_sell_path_mean': [float(x) for x in rho[..., 0].mean(axis=0)],
        'rho_buy_path_mean': [float(x) for x in rho[..., 1].mean(axis=0)],
    }


def make_br_stage_fn_v2(env_T, statics, optimizer, nb_iterations, B, static, n,
                        n_restarts=1):
    """One jitted best-response stage, all types (x n_restarts) at once.
    stacked_mix = (buffers, k): compiled once for the whole solve.
    Fresh common noise, component assignment and idiosyncratic draws per step."""
    T_int = int(env_T.T)
    js = jnp.tile(jnp.arange(N_TYPES), n_restarts)

    def sample_noise(key):
        kP, kE, kI, kC = jax.random.split(key, 4)
        P = jax.vmap(env_T.generate_P, in_axes=(0, None))(jax.random.split(kP, B), T_int)
        eps0 = jax.vmap(env_T.generate_eps0, in_axes=(0, None))(jax.random.split(kE, B), T_int)
        ki = jax.random.split(kI, N_TYPES)
        idio_pop = tuple(
            jax.vmap(lambda kk, j=j: jax.vmap(
                env_T.generate_idiosyncratic_noise, in_axes=(0, None, None))(
                    jax.random.split(kk, n), env_T.agent_params[j, 7], T_int)
            )(jax.random.split(ki[j], B))
            for j in range(N_TYPES))
        return P, eps0, idio_pop, jax.random.split(kC, B)

    def prices(mix, P, eps0, idio_pop, ck):
        buffers, k = mix
        return jax.vmap(lambda c, p, e, i0, i1, i2: env_T.ensemble_price_path_v2(
            buffers, statics, k, c, p, e, (i0, i1, i2)))(ck, P, eps0, *idio_pop)

    def loss_given_noise(p_stack, key, A, rho, P):
        def per_type(p, j, kk):
            model = eqx.combine(p, static)
            idio = jax.vmap(env_T.generate_idiosyncratic_noise, in_axes=(0, None, None))(
                jax.random.split(kk, B), env_T.agent_params[j, 7], T_int)
            R = jax.vmap(lambda a, r, pp, ii: env_T.rollout_agent_batch_v2(
                model, j, a, r, pp, ii[None, :])[1][0])(A, rho, P, idio)
            return -jnp.mean(R) / env_T.reward_scale(j)
        per = jax.vmap(per_type)(p_stack, js, jax.random.split(key, js.shape[0]))
        return jnp.sum(per), per

    @jax.jit
    def stage(p_stack, s, loop_key, mix):
        def one(carry, it):
            p_, s_ = carry
            k_noise, k_grad = jax.random.split(jax.random.fold_in(loop_key, it))
            P, eps0, idio_pop, ck = sample_noise(k_noise)
            A, rho = jax.lax.stop_gradient(prices(mix, P, eps0, idio_pop, ck))
            (_, per), g = jax.value_and_grad(loss_given_noise, has_aux=True)(
                p_, k_grad, A, rho, P)
            upd, s_new = optimizer.update(g, s_, p_)
            return (eqx.apply_updates(p_, upd), s_new), per
        (p_f, s_f), losses = jax.lax.scan(one, (p_stack, s), jnp.arange(nb_iterations))
        return p_f, s_f, losses

    return stage


def fictitious_play_rung2_v2(env, gamma_ref=GAMMA_REF, K=30, T=30, iters=8000,
                             lr=1e-3, lr_min=1e-4, B=256, S=200, M=32, n=N_SIM,
                             eval_chunk=50, key=None, save_dir=None,
                             mm_warmstart=None, br_restarts=2, verbose=True, meta=None):
    """Rung 2 FP with v2 clearing and per-agent sampling. Single horizon T,
    cosine lr. Resumable from progress.json + policy_<k>_type<j>.eqx."""
    import json
    assert K <= K_MAX, f"K={K} > K_MAX={K_MAX}"
    if key is None:
        key = jax.random.PRNGKey(0)
    k_init_base, k_iter_base = jax.random.split(key)
    env_f = env.set_context(jnp.asarray(gamma_ref, dtype=jnp.float32)).set_T(T)

    ver = make_verification_set_v2(env_f, T, S, M, n)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    progress_path = os.path.join(save_dir, 'progress.json') if save_dir else None

    records, start_k = [], 0
    history = None
    if progress_path and os.path.exists(progress_path):
        with open(progress_path) as f:
            records = json.load(f).get('records', [])
        n_avail = 0
        while os.path.exists(os.path.join(save_dir, f'policy_{n_avail}_type0.eqx')):
            n_avail += 1
        start_k = min(len(records), max(0, n_avail - 1))
        records = records[:start_k]
        history = [load_triple_v2(save_dir, i) for i in range(start_k + 1)]
        if verbose:
            print(f"=== RESUMING at iteration {start_k+1} ({len(history)} triples) ===", flush=True)
    if history is None:
        ik = jax.random.split(k_init_base, N_TYPES)
        history = [tuple(TypePolicyNNV2(key=ik[j]) for j in range(N_TYPES))]
        if save_dir:
            save_triple(save_dir, 0, history[0])
    if start_k >= K:
        return history, records

    mm_warm = None
    if mm_warmstart:
        mm_warm = eqx.tree_deserialise_leaves(mm_warmstart, TypePolicyNNV2(key=jax.random.PRNGKey(0)))

    # fixed mixture buffers (one per type), slots 0..k-1 filled
    bufs, statics = [], []
    for j in range(N_TYPES):
        b, st = make_buffer(history[0][j], K_MAX)
        for i, tr in enumerate(history):
            b = buffer_set(b, i, tr[j])
        bufs.append(b); statics.append(st)
    bufs, statics = tuple(bufs), tuple(statics)
    model_static = statics[0]

    price_fn = make_price_fn_v2(env_f, statics, return_diag=True)
    ret_fns = [make_batched_return_fn_v2(env_f, j) for j in range(N_TYPES)]
    hist_fns = [make_batched_hist_fn_v2(env_f, j, model_static) for j in range(N_TYPES)]
    optimizer = optax.adam(optax.cosine_decay_schedule(
        init_value=lr, decay_steps=iters, alpha=lr_min / lr))
    stage_fns = {T: make_br_stage_fn_v2(env_f, statics, optimizer, iters, B, model_static,
                                        n, n_restarts=br_restarts)}
    shares = env_f.type_shares
    if verbose:
        w = np.asarray(env_f.context_weights())
        cap = float(np.sum(w * np.asarray(env_f.agent_params[:, 5])))
        print(f"v2 clearing: floor_mode={FLOOR_MODE}  sigma_S(gamma) = {SUPPLY_C:g} x Theta / sqrt(365) = "
              f"{float(env_f.sigma_S()):.4g} (Theta {float(env_f.total_allocation()):.4g}, per-step std "
              f"{float(env_f.sigma_S() * jnp.sqrt(env_f.dt)):.4g}, T={T})  "
              f"corridor [{float(env_f.Afloor):.2f}, {A_CEIL_V2:g}] USD  beta_min={BETA_MIN_V2}  "
              f"bisection={CLEAR_ITERS_V2}/side  n={n}/type  K_MAX={K_MAX}  cap_total={cap:.4g}",
              flush=True)

    t_start, A_prev = time.time(), None
    for k in range(start_k, K):
        t_iter = time.time()
        kk = len(history)
        if verbose:
            done = k - start_k
            eta = "" if done == 0 else f"   ETA {_hms((time.time()-t_start)/done*(K-k))}"
            print(f"\n{'='*72}\nRUNG 2 v2 — FP ITERATION {k+1}/{K}   [mixture {kk}]   "
                  f"elapsed {_hms(time.time()-t_start)}{eta}\n{'='*72}", flush=True)
        mix = (bufs, jnp.int32(kk))
        ck = block_comp_keys(S, kk)
        A_ver, rho_ver, cdiag = chunked_tree(lambda a, b: price_fn(
            bufs, jnp.int32(kk), ck[a:b], ver['P'][a:b], ver['eps0'][a:b], ver['idio_pop']),
            S, eval_chunk)
        clr = clearing_metrics_v2(env_f, A_ver, rho_ver, cdiag)
        if verbose:
            a = np.asarray(A_ver)[:, 1:]
            print(f"  cleared price over {S} frozen draws: mean {a.mean():.3f}  A_0 {clr['A0_cleared']:.3f}  "
                  f"final median {np.median(a[:, -1]):.3f} [p10 {np.quantile(a[:, -1], .1):.2f}, "
                  f"p90 {np.quantile(a[:, -1], .9):.2f}]", flush=True)
            print(f"  clearing: resid {clr['clear_residual_mean_rel']:.2e} of cap (max "
                  f"{clr['clear_residual_max_rel']:.2e})  cons err {clr['cons_err_max_rel']:.2e}  "
                  f"floor {clr['A_on_floor']:.1%}  ceil {clr['A_on_ceil']:.1%}  flat "
                  f"{clr['clear_flat_frac']:.1%}  rho_sell on floor {clr['rho_sell_mean_on_floor']:.3f} "
                  f"(min {clr['rho_sell_min']:.3f})  rho_buy min {clr['rho_buy_min']:.3f}  "
                  f"blocked {clr['blocked_volume']:.3g} {np.round(clr['blocked_sell_type'], 2)}  "
                  f"support cost {clr['support_cost']:.3g}  "
                  f"drift {clr['price_drift_mean']:+.3f} +/- {clr['price_drift_std']:.3f}  "
                  f"market short {clr['market_short_frac']:.1%}  indiv short "
                  f"{np.round(clr['short_T_type'], 2)}", flush=True)
            print(f"  schedules: beta mean {np.round(clr['beta_mean'], 3)}  R mean "
                  f"{np.round(clr['R_mean'], 2)}  f mean {np.round(clr['f_mean'], 2)}  "
                  f"|S| mean {clr['supply_abs_mean']:.3g} (exec mean {clr['supply_exec_mean']:+.3g})",
                  flush=True)

        t_br = time.time()
        new_triple, loss_hist, br_info = _train_br_restarts(
            lambda kk_: TypePolicyNNV2(key=kk_), mix, jax.random.fold_in(k_iter_base, k),
            stage_fns, optimizer, mm_warm, (T,), (iters,), (lr,),
            n_restarts=br_restarts, verbose=verbose)
        br_secs = time.time() - t_br
        loss_file = None
        if save_dir:
            os.makedirs(os.path.join(save_dir, 'losses'), exist_ok=True)
            loss_file = os.path.join('losses', f'br_loss_{k+1:03d}.npz')
            np.savez_compressed(os.path.join(save_dir, loss_file),
                                **{f'stage0_T{T}': np.asarray(loss_hist[0], dtype=np.float32),
                                   f'all_stage0_T{T}': np.asarray(br_info['loss_all'][0], dtype=np.float32)})
            save_br_candidates(save_dir, k + 1, br_info)
        if verbose:
            print(f"    trained in {_hms(br_secs)}", flush=True)

        rho_j = jnp.asarray(rho_ver)
        per_type = [
            exploitability_r2(env_f, j, new_triple[j], history, A_ver, ver['P'],
                              ver['idio_eval'][j],
                              lambda pol, A, P, idio, j=j: ret_fns[j](pol, A, rho_j, P, idio),
                              lambda st, A, P, idio, j=j: hist_fns[j](st, A, rho_j, P, idio),
                              model_static)
            for j in range(N_TYPES)]
        weighted = sum(shares[j] * per_type[j]['gap'] for j in range(N_TYPES))
        weighted_rel = sum(shares[j] * per_type[j]['gap_relative'] for j in range(N_TYPES))
        if verbose:
            names = ("private     ", "large public", "market maker")
            print("  exploitability (mean over the frozen common-noise set):", flush=True)
            for j, m in enumerate(per_type):
                flag = "  <-- NEGATIVE: BR under-trained" if m['gap'] < -m['gap_se'] else ""
                print(f"    {names[j]}  E = {m['gap']:+12.1f} +/- {m['gap_se']:8.1f}  "
                      f"({m['gap_relative']:+8.4%})  V_br {m['V_br']:+11.5g}{flag}", flush=True)
            print(f"    weighted {weighted:+.2f}  ({weighted_rel:+.4%})", flush=True)

        path_l2 = float('nan')
        if A_prev is not None:
            path_l2 = float(jnp.mean(jnp.linalg.norm(A_ver - A_prev, axis=1)))
            if verbose:
                print(f"  mean_s ||A^k - A^k-1||_2 = {path_l2:.6f}", flush=True)
        A_prev = A_ver

        rec = {'iteration': k + 1, 'mixture_size': kk, 'per_type': per_type,
               'weighted': weighted, 'weighted_relative': weighted_rel, 'path_l2': path_l2,
               'A_mean': float(np.mean(np.asarray(A_ver))), 'clearing': clr,
               'A_on_floor': clr['A_on_floor'], 'br_seconds': br_secs,
               'br_loss_file': loss_file, 'br_restarts': br_restarts,
               'br_chosen': br_info['chosen'],
               'br_loss_end': np.asarray(br_info['loss_end']).tolist(),
               'iter_seconds': time.time() - t_iter}
        records.append(rec)
        history.append(new_triple)
        bufs = tuple(buffer_set(bufs[j], kk, new_triple[j]) for j in range(N_TYPES))

        if save_dir:
            save_triple(save_dir, k + 1, new_triple)
            _atomic_json(progress_path, {
                'rung': 2, 'version': 'mc_v2', 'gamma_ref': [float(x) for x in gamma_ref],
                'T_grid': [T], 'iters': [iters], 'lrs': [lr], 'lr_schedule': 'cosine',
                'lr_min': lr_min, 'B': B, 'S': S, 'M': M, 'K': K, 'n_sim': n, 'K_MAX': K_MAX,
                'floor_mode': FLOOR_MODE, 'A_CEIL': A_CEIL_V2, 'supply_c': SUPPLY_C, 'sigma_S': float(env_f.sigma_S()),
                'Theta': float(env_f.total_allocation()),
                'beta_min': BETA_MIN_V2, 'clear_iters': CLEAR_ITERS_V2,
                'pop_sampling': 'agent', 'br_restarts': br_restarts,
                'agent_per_policy': env.agent_per_policy,
                'type_shares': list(shares), 'verify_seed': VERIFY_SEED,
                'comp_seed': COMP_SEED, 'mm_warmstart': mm_warmstart,
                **(meta or {}), 'records': records})
        if verbose:
            print(f"  checkpoint saved  [iteration {_hms(time.time()-t_iter)}]", flush=True)
    return history, records
