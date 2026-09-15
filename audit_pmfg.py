"""PMFG audit of a BE-MFG mixture — one (scenario, checkpoint) cell.

What the training loop reports at a fixed context is measured against a deviator
drawn from the SAME conditional family as the mixture: one FiLM network that has
to serve the whole context box at once. That is a restricted deviation class, so
E_BE is a lower bound on a lower bound.

This replaces the deviator with a free PMFG policy — an unconditional
TypePolicyNN trained at that one context and nothing else. What comes out is the
real PMFG exploitability of the BE-MFG mixture, and

    E_PMFG(c) - E_BE(c)

is the price of generality: what a specialist gains over the generalist when
both are given the same gradient budget.

The audit scenarios are POLICY narratives, not diagnostics. The one-at-a-time
attribution (alloc_hi/lo, kappa_hi/lo, floor_lo/hi, mm_thin/thick) is already
done by the frozen blocks inside the training loop; these four are the settings
a reader of the paper cares about.

Pairing
-------
Checkpoint i is the record the FP loop wrote as `iteration: i`: the mixture is
triples 0..i-1 and the best response it scored is policy_i. The audit trains
against that same mixture, prices the same frozen block, and re-scores policy_i
on those same price paths — so E_PMFG and E_BE come out of one job, row-paired,
sharing every draw.

    python3 audit_pmfg.py --run-dir results_floor20/choice \
        --scenario current_kets --checkpoint 50 --iters 12000
"""

import os
import sys
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _floor_max_of_run(argv):
    """Box top of the run under audit, read from its progress.json BEFORE
    envs/utils are imported.

    base_params reads KETS_FLOOR_MAX once at import and utils._G_HI (the FiLM
    normaliser) is built from it, so the mixture must be conditioned with the
    box it was trained on. Relying on the shell to export the right value is
    how the first floor30/floor40 high_floor_mid_alloc cells were normalised
    with the floor20 box; the run itself is the authority."""
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
import optax

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import TypePolicyNN, stack_policies
from utils import (_hms, _atomic_json, chunked, draw_common_noise, draw_idio,
                   pretrain_market_maker, new_policy, load_triple_r3,
                   bind_env, bind_stacked, make_price_fn_r3,
                   make_batched_return_fn_r3, make_batched_hist_fn_r3,
                   CTX_NAMES, CONTEXT_DIM, VERIFY_SEED_R3, _G_HI)

TYPE_LABEL = ("private     ", "large public", "market maker")
AUDIT_SEED = 90909          # never VERIFY_SEED_R3: the audit BR must not train
                            # on the draws it is scored against


def audit_scenarios(floor_max=None):
    """The four policy scenarios, in CTX_NAMES order.

    `high_floor_mid_alloc` is defined RELATIVE to the top of the context box, so
    it is 15 / 25 / 35 for the floor20 / floor30 / floor40 runs. That makes it a
    consistent question within a run ("a floor near the top of what this policy
    was trained on") but NOT the same market across runs — do not read scenario 4
    across the three floor studies as a like-for-like comparison.
    """
    fm = float(FLOOR_MAX if floor_max is None else floor_max)
    return {
        # name                    init_alloc  kappa  Afloor  financial_inst
        'current_kets':          (0.9,        3.0,   6.0,    0.4),
        'low_alloc':             (0.1,        3.0,   6.0,    0.4),
        'high_tax':              (0.9,        5.0,   6.0,    0.4),
        'no_market_maker':       (0.9,        3.0,   6.0,    0.0),
        'high_floor_mid_alloc':  (0.5,        3.0,   fm - 5.0, 0.4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Best response at ONE fixed context, with an unconditional policy
# ═════════════════════════════════════════════════════════════════════════════

def make_br_stage_fn_pmfg(env_c, statics, optimizer, nb_iterations, B, static):
    """One jitted BR stage at a fixed context, all three types at once.

    Two differences from the conditional stage in utils.py, and no others: the
    context is not sampled (the env arrives already bound, and the mixture bound
    with it), and the responding policy is unconditional so it is used directly
    instead of through bind_policy. The stop_gradient on the price path, the one
    scan of T steps per gradient step and the three types stacked under one Adam
    are unchanged — so a step here costs the same as a step there and the
    budgets are comparable.
    """
    T_int = int(env_c.T)
    js = jnp.arange(N_TYPES)
    counts = env_c.type_counts

    def sample_market(key):
        kP, kE, kI = jax.random.split(key, 3)
        P = jax.vmap(env_c.generate_P, in_axes=(0, None))(
            jax.random.split(kP, B), T_int)
        eps0 = jax.vmap(env_c.generate_eps0, in_axes=(0, None))(
            jax.random.split(kE, B), T_int)
        ki = jax.random.split(kI, N_TYPES)
        idio_pop = tuple(
            jax.vmap(lambda k, j=j: jax.vmap(
                env_c.generate_idiosyncratic_noise, in_axes=(0, None, None))(
                    jax.random.split(k, counts[j]),
                    env_c.agent_params[j, 7], T_int)
            )(jax.random.split(ki[j], B))
            for j in range(N_TYPES))
        return P, eps0, idio_pop

    def prices(mix_bound, P, eps0, idio_pop):
        def one(p, e, i0, i1, i2):
            return env_c.ensemble_price_path(mix_bound, statics, p, e,
                                             (i0, i1, i2))
        return jax.vmap(one)(P, eps0, *idio_pop)

    def loss_given_market(p_stack, key, A, P):
        def per_type(p, j, kk):
            model = eqx.combine(p, static)
            idio = jax.vmap(env_c.generate_idiosyncratic_noise,
                            in_axes=(0, None, None))(
                jax.random.split(kk, B), env_c.agent_params[j, 7], T_int)

            def one(a, pp, ii):
                _, R = env_c.rollout_agent_batch(model, j, a, pp, ii[None, :])
                return R[0]

            R = jax.vmap(one)(A, P, idio)
            return -jnp.mean(R) / env_c.reward_scale(j)

        per = jax.vmap(per_type, in_axes=(0, 0, 0))(
            p_stack, js, jax.random.split(key, N_TYPES))
        return jnp.sum(per), per

    @jax.jit
    def stage(p_stack, s, loop_key, mix_bound):
        def one(carry, n):
            p_, s_ = carry
            k_noise, k_grad = jax.random.split(jax.random.fold_in(loop_key, n))
            P, eps0, idio_pop = sample_market(k_noise)
            A = jax.lax.stop_gradient(prices(mix_bound, P, eps0, idio_pop))
            (_, per), g = jax.value_and_grad(loss_given_market, has_aux=True)(
                p_, k_grad, A, P)
            upd, s_new = optimizer.update(g, s_, p_)
            return (eqx.apply_updates(p_, upd), s_new), per

        (p_f, s_f), losses = jax.lax.scan(one, (p_stack, s),
                                          jnp.arange(nb_iterations))
        return p_f, s_f, losses

    return stage


def make_return_fn_pmfg(env_c, j):
    """Returns of one unconditional policy over the rows of a block -> (S, M)."""
    @eqx.filter_jit
    def fn(policy, A, P, idio):
        def one(a, p):
            _, R = env_c.rollout_agent_batch(policy, j, a, p, idio)
            return R
        return jax.vmap(one)(A, P)
    return fn


def gap_stats(R_br, R_mix, scale):
    """Paired per row, exactly as the training loop's exploitability does it."""
    d_row = jnp.mean(R_br - R_mix, axis=1)
    return d_row, {
        'gap': float(jnp.mean(d_row)),
        'gap_se': float(jnp.std(d_row, ddof=1) / jnp.sqrt(d_row.shape[0])),
        'gap_relative': float(jnp.mean(d_row)) / scale,
        'V_br': float(jnp.mean(R_br)),
        'V_br_relative': float(jnp.mean(R_br)) / scale,
        'gap_p10': float(jnp.quantile(d_row, 0.1)),
        'gap_p90': float(jnp.quantile(d_row, 0.9)),
    }


# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='PMFG audit of a BE-MFG mixture')
    p.add_argument('--run-dir', required=True,
                   help='a solved run, e.g. results_floor20/choice')
    p.add_argument('--scenario', required=True)
    p.add_argument('--checkpoint', type=int, default=-1,
                   help='FP iteration i (mixture = triples 0..i-1, BE BR = '
                        'policy_i); -1 = the last one on disk')
    p.add_argument('--iters', type=int, default=12000,
                   help='BR gradient steps; match the BE run for the '
                        'like-for-like read')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-schedule', type=str, default='cosine',
                   choices=['cosine', 'const'])
    p.add_argument('--lr-min', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--samples', type=int, default=250,
                   help='S: frozen noise draws for the audit block')
    p.add_argument('--eval-samples', type=int, default=32)
    p.add_argument('--eval-chunk', type=int, default=50)
    p.add_argument('--mm-steps', type=int, default=800,
                   help='heuristic warm-start steps for the market maker; 0 '
                        'disables it (invites the zero-gradient trap)')
    p.add_argument('--seed', type=int, default=AUDIT_SEED)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    prog_path = os.path.join(args.run_dir, 'progress.json')
    if not os.path.exists(prog_path):
        raise SystemExit(f"{prog_path} not found — nothing to audit")
    with open(prog_path) as f:
        prog = json.load(f)

    n_avail = 0
    while os.path.exists(os.path.join(args.run_dir,
                                      f'policy_{n_avail}_type0.eqx')):
        n_avail += 1
    if n_avail < 2:
        raise SystemExit(f"{args.run_dir}: only {n_avail} triples on disk")
    ck = args.checkpoint if args.checkpoint > 0 else n_avail - 1
    if not 1 <= ck <= n_avail - 1:
        raise SystemExit(f"--checkpoint must be in 1..{n_avail - 1}")

    # the box this run was trained on decides the relative scenario
    box = prog.get('context_box')
    floor_max = float(box[2][1]) if box else FLOOR_MAX
    if abs(float(_G_HI[2]) - floor_max) > 1e-6:
        raise SystemExit(f"FiLM normaliser box top {float(_G_HI[2])} != run box top "
                         f"{floor_max}: KETS_FLOOR_MAX was set to a different "
                         f"value than the run was trained with")
    scen = audit_scenarios(floor_max)
    if args.scenario not in scen:
        raise SystemExit(f"--scenario must be one of {list(scen)}")
    c = scen[args.scenario]

    T = int(prog['T_grid'][-1])
    B = args.batch or prog['B']
    mode = prog.get('mode', 'choice')
    agents = int(prog.get('agent_per_policy', 20))
    out_dir = args.out or os.path.join(args.run_dir, 'audit', args.scenario,
                                       f'b{args.iters}', f'k{ck:03d}')
    os.makedirs(out_dir, exist_ok=True)

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
    ca = jnp.asarray(c, dtype=jnp.float32)
    env_c = bind_env(env, ca)

    print("=" * 74, flush=True)
    print(f"PMFG AUDIT  |  {args.run_dir}  |  {args.scenario}")
    print(f"context   {dict(zip(CTX_NAMES, [round(float(x), 4) for x in c]))}")
    print(f"          (floor box top for this run = {floor_max})")
    print(f"checkpoint {ck}: mixture = triples 0..{ck-1} ({ck} triples), "
          f"BE deviator = policy_{ck}")
    print(f"BR budget {args.iters} steps, lr {args.lr:g} {args.lr_schedule}"
          + (f" -> {args.lr_min:g}" if args.lr_schedule == 'cosine' else "")
          + f", B = {B}   (the BE run used {prog['iters'][-1]} at B = {prog['B']})")
    print(f"block     S = {args.samples}, M = {args.eval_samples}")
    print(f"out       {out_dir}")
    print("=" * 74, flush=True)

    # ── the mixture the BE run actually faced at this checkpoint ────────────
    history = [load_triple_r3(args.run_dir, i) for i in range(ck)]
    be_triple = load_triple_r3(args.run_dir, ck)
    statics = tuple(stack_policies([history[0][j]])[1] for j in range(N_TYPES))
    _, film_static = eqx.partition(history[0][0], eqx.is_array)
    stacked = tuple(stack_policies([h[j] for h in history])[0]
                    for j in range(N_TYPES))
    mix_bound = tuple(bind_stacked(stacked[j], ca) for j in range(N_TYPES))

    # ── the frozen audit block: one context, S noise draws ──────────────────
    kc, kp, ke = jax.random.split(jax.random.PRNGKey(
        VERIFY_SEED_R3 + 7919 * (abs(hash(args.scenario)) % 97 + 1)), 3)
    P, eps0 = draw_common_noise(env_c, T, args.samples, kc)
    idio_pop = draw_idio(env_c, T, env_c.type_counts, kp)
    idio_eval = draw_idio(env_c, T, (args.eval_samples,) * N_TYPES, ke)
    ctx_rows = jnp.broadcast_to(ca, (args.samples, CONTEXT_DIM))

    # ── train the unconditional deviator ────────────────────────────────────
    if args.lr_schedule == 'cosine':
        optimizer = optax.adam(optax.cosine_decay_schedule(
            init_value=args.lr, decay_steps=args.iters,
            alpha=args.lr_min / args.lr))
    else:
        optimizer = optax.adam(args.lr)

    k_model, k_train, k_mm = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    mkeys = jax.random.split(k_model, N_TYPES)
    models = [TypePolicyNN(key=mkeys[j]) for j in range(N_TYPES)]

    if args.mm_steps > 0:
        # The market maker starts at theta_0 = 0 against a clip, so a fresh net
        # sits in a zero-gradient trap. Regress it onto the buy-then-sell
        # heuristic first, on this scenario's own price path.
        A_seed = make_price_fn_r3(env, statics)(
            stacked, ctx_rows[:1], P[:1], eps0[:1], idio_pop)[0]
        models[2], mm_loss = pretrain_market_maker(
            env_c, A_seed, P[0], idio_eval[2], k_mm, steps=args.mm_steps)
        print(f"market-maker warm start: {args.mm_steps} steps, "
              f"mse {float(mm_loss[0]):.5f} -> {float(mm_loss[-1]):.6f}",
              flush=True)

    parts = [eqx.partition(m, eqx.is_array) for m in models]
    static = parts[0][1]
    p_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                     *[q for q, _ in parts])
    opt_state = optimizer.init(p_stack)
    stage = make_br_stage_fn_pmfg(env_c, statics, optimizer, args.iters, B,
                                  static)

    t0 = time.time()
    p_stack, opt_state, losses = stage(p_stack, opt_state, k_train, mix_bound)
    losses = np.asarray(losses)
    br_secs = time.time() - t0
    print(f"  trained in {_hms(br_secs)}   "
          + "  ".join(f"t{j}: {losses[0, j]:+.5f}->{losses[-1, j]:+.5f}"
                      for j in range(N_TYPES)), flush=True)
    pmfg = tuple(eqx.combine(jax.tree_util.tree_map(lambda x: x[j], p_stack),
                             static) for j in range(N_TYPES))

    # ── score BOTH deviators on the same block and the same price paths ─────
    price_fn = make_price_fn_r3(env, statics)
    A = chunked(lambda a, b: price_fn(stacked, ctx_rows[a:b], P[a:b],
                                      eps0[a:b], idio_pop),
                args.samples, args.eval_chunk)
    ret_pmfg = [make_return_fn_pmfg(env_c, j) for j in range(N_TYPES)]
    ret_be = [make_batched_return_fn_r3(env, j) for j in range(N_TYPES)]
    hist_fn = [make_batched_hist_fn_r3(env, j, film_static)
               for j in range(N_TYPES)]

    per_type, rows_p, rows_b = [], [], []
    for j in range(N_TYPES):
        scale = float(env_c.reward_scale(j))
        idio = idio_eval[j]
        R_mix = jnp.mean(hist_fn[j](stacked[j], ctx_rows, A, P, idio), axis=1)
        dp, mp = gap_stats(ret_pmfg[j](pmfg[j], A, P, idio), R_mix, scale)
        db, mb = gap_stats(ret_be[j](be_triple[j], ctx_rows, A, P, idio),
                           R_mix, scale)
        rows_p.append(np.asarray(dp) / scale)
        rows_b.append(np.asarray(db) / scale)
        per_type.append({'type': TYPE_LABEL[j].strip(), 'scale': scale,
                         'V_mix': float(jnp.mean(R_mix)),
                         'pmfg': mp, 'be': mb,
                         'price_of_generality': (mp['gap_relative']
                                                 - mb['gap_relative'])})

    w = np.asarray(env_c.context_weights())
    wtd_p = float(np.sum(w * np.stack(rows_p, axis=1), axis=1).mean())
    wtd_b = float(np.sum(w * np.stack(rows_b, axis=1), axis=1).mean())

    print(f"\n  {'type':<14}{'E_PMFG':>12}{'E_BE':>12}{'price of gen':>15}"
          f"{'V_br PMFG':>12}{'V_br BE':>10}")
    print("  " + "-" * 73)
    for j in range(N_TYPES):
        m = per_type[j]
        flag = ("  <-- NEGATIVE: audit BR under-trained"
                if m['pmfg']['gap'] < -m['pmfg']['gap_se'] else "")
        print(f"  {TYPE_LABEL[j]:<14}{m['pmfg']['gap_relative']:>12.4%}"
              f"{m['be']['gap_relative']:>12.4%}"
              f"{m['price_of_generality']:>15.4%}"
              f"{m['pmfg']['V_br_relative']:>12.2%}"
              f"{m['be']['V_br_relative']:>10.2%}{flag}")
    print(f"  {'weighted':<14}{wtd_p:>12.4%}{wtd_b:>12.4%}"
          f"{wtd_p - wtd_b:>15.4%}   (shares {np.round(w, 3)})")

    for j in range(N_TYPES):
        eqx.tree_serialise_leaves(
            os.path.join(out_dir, f'policy_type{j}.eqx'), pmfg[j])
    np.savez_compressed(os.path.join(out_dir, 'loss.npz'),
                        loss=losses.astype(np.float32))
    _atomic_json(os.path.join(out_dir, 'result.json'), {
        'run_dir': args.run_dir, 'mode': mode, 'scenario': args.scenario,
        'context': [float(x) for x in c], 'context_names': list(CTX_NAMES),
        'floor_max': floor_max, 'checkpoint': ck, 'mixture_size': ck,
        'iters': args.iters, 'lr': args.lr, 'lr_schedule': args.lr_schedule,
        'lr_min': args.lr_min, 'B': B, 'seed': args.seed,
        'S': args.samples, 'M': args.eval_samples,
        'be_iters': prog['iters'][-1], 'be_B': prog['B'],
        'br_seconds': br_secs, 'shares': [float(x) for x in w],
        'per_type': per_type,
        'weighted_pmfg': wtd_p, 'weighted_be': wtd_b,
        'price_of_generality': wtd_p - wtd_b,
        'loss_first': [float(losses[0, j]) for j in range(N_TYPES)],
        'loss_last': [float(losses[-1, j]) for j in range(N_TYPES)],
    })
    print(f"\n-> {out_dir}  (policy_type*.eqx, loss.npz, result.json)",
          flush=True)


if __name__ == '__main__':
    main()
