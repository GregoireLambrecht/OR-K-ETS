"""Faster fictitious play for the major–minor market (same model and evaluation
as utils_majmin.fictitious_play_majmin). Four changes:

1. PRICE BANK for the minors' best responses (MAJOR_MINOR_CLEARING.md §8, note;
   MARKET_CLEARING.md phase 5). The population is frozen during a best
   response, so once per FP iteration N_bank markets are cleared WITHOUT
   gradient (gamma ~ Q, common noise, major component u ~ U{0..k-1} per market,
   stratified minor components) and their public paths (gamma, P, A, rho, H)
   are stored. Every minor gradient step samples B of them; the deviator's own
   idiosyncratic noise stays fresh. The major's best response cannot use it
   (prices depend on the deviating major), so it re-clears as before.

2. WARM-STARTED best responses. From iteration WARM_FROM on, each player's best
   response starts from its best response of the previous iteration and trains
   for a smaller budget (iters_warm / iters0_warm). If the warm best response
   FAILS — its exploitability on the main frozen block (prior for rung 3, ref
   for rung 2) is negative beyond one standard error — that type is retrained
   ONCE from a fresh seed with the full budget (the financial institution from
   its warm start), and the fresh network is kept.

3. LIGHTER CLEARING: the bisection length is KETS_CLEAR_ITERS (set by the
   caller, e.g. 24; float32 saturates there on a [floor, 200] corridor).

4. TWO GPUs: the minors' stage (bank + best responses) runs on device 0 and the
   major's stage on device 1, concurrently (both respond to the mixture of
   iteration k-1). Falls back to one device.
"""
import os
import json
import threading
import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from envs.environment import N_TYPES
from envs.environment_majmin import MINORS, MAJOR
from envs.models_majmin import USE_H
from envs.models_majmin import minor_policy, major_policy, make_buffer, buffer_set
from envs.params_v2 import K_MAX, FLOOR_MODE, SUPPLY_C, A_CEIL_V2, CLEAR_ITERS_V2
from utils import _hms, _atomic_json, norm_gamma, FIXED_SCENARIOS, VERIFY_SEED
from utils_majmin import (save_net, load_net, draw_markets, make_sim, block_metrics,
                          RUNG3_FIXED, COMP_SEED)

WARM_FROM = 2          # iteration (1-based) from which best responses are warm-started
STUCK_TOL = 1e-3      # |tail-mean training loss| below this = stuck at no trade (losses are O(0.1-1))


def fictitious_play_majmin_fast(env, rung, ctx_ref, K=30, T=30, n=30, iters=8000, iters0=4000,
                                iters_warm=2000, iters0_warm=1000, lr_warm=1e-4, n_bank=65536, bank_chunk=8192,
                                warm_from=WARM_FROM,
                                lr=1e-3, lr_min=1e-4, B=256, B0=64, S=200, S_fixed=100, M=32,
                                key=None, save_dir=None, mm_warmstart=None, verbose=True, meta=None):
    assert K < K_MAX
    bank_chunk = min(bank_chunk, n_bank)
    key = jax.random.PRNGKey(0) if key is None else key
    k_init, k_iter = jax.random.split(key)
    env = env.set_T(T)
    g_ref = jnp.asarray(ctx_ref, jnp.float32)
    if rung == 2:
        prior = lambda kk, b: jnp.broadcast_to(g_ref, (b, 4))
    else:
        prior = lambda kk, b: env.generate_context(kk, b)
    m0_ref = float(env.bind(g_ref).major_mass())
    s0 = float(env.major_reward_scale(m0_ref))
    s_minor = {j: float(env.reward_scale(j)) for j in MINORS}
    w_ref = np.asarray(env.bind(g_ref).context_weights())
    devs = jax.devices()
    d_min, d_maj = devs[0], devs[1 if len(devs) > 1 else 0]

    # ── frozen evaluation blocks (identical to utils_majmin) ─────────────────
    kv = jax.random.PRNGKey(VERIFY_SEED)
    blocks = {}
    if rung == 2:
        specs = [('ref', ctx_ref, S)]
    else:
        fx = dict(FIXED_SCENARIOS)
        specs = [('prior', None, S)] + [(nm, fx[nm], S_fixed) for nm in RUNG3_FIXED]
    for bi, (nm, c, Sb) in enumerate(specs):
        pr = prior if c is None else (lambda kk, b, c=c: jnp.broadcast_to(jnp.asarray(c, jnp.float32), (b, 4)))
        gam, P, eps0, imaj, imin, _ = draw_markets(env, jax.random.fold_in(kv, bi), Sb, T, n, pr)
        imin = tuple(x[0] for x in imin)
        kE = jax.random.fold_in(kv, 100 + bi)
        ieval = {j: jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
            jax.random.split(jax.random.fold_in(kE, j), M), env.agent_params[j, 7], T) for j in MINORS}
        blocks[nm] = dict(gam=gam, P=P, eps0=eps0, imaj=imaj, imin=imin, ieval=ieval, S=Sb,
                          ctx=None if c is None else list(c))
    main_block = 'ref' if rung == 2 else 'prior'

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    ppath = os.path.join(save_dir, 'progress.json') if save_dir else None

    records, start_k, nets = [], 0, None
    if ppath and os.path.exists(ppath):
        with open(ppath) as f:
            records = json.load(f).get('records', [])
        n_av = 0
        while os.path.exists(os.path.join(save_dir, f'policy_{n_av}_type0.eqx')):
            n_av += 1
        start_k = min(len(records), max(0, n_av - 1))
        records = records[:start_k]
        nets = [[load_net(save_dir, i, j) for j in range(N_TYPES)] for i in range(start_k + 1)]
        if verbose:
            print(f"=== RESUMING at iteration {start_k + 1} ===", flush=True)
    if nets is None:
        ik = jax.random.split(k_init, N_TYPES)
        nets = [[major_policy(ik[j]) if j == MAJOR else minor_policy(ik[j]) for j in range(N_TYPES)]]
        if save_dir:
            for j in range(N_TYPES):
                save_net(save_dir, 0, j, nets[0][j])
    mm_warm = None
    if mm_warmstart:
        mm_warm = eqx.tree_deserialise_leaves(mm_warmstart, minor_policy(jax.random.PRNGKey(0)))

    bufs, stats = [], []
    for j in range(N_TYPES):
        b, st = make_buffer(nets[0][j], K_MAX)
        for i, tr in enumerate(nets):
            b = buffer_set(b, i, tr[j])
        bufs.append(b); stats.append(st)
    minor_static = tuple(stats[j] for j in MINORS)
    maj_static = stats[MAJOR]
    sim = make_sim(env, minor_static, maj_static)
    js = jnp.array(MINORS)
    scales = jnp.array([s_minor[j] for j in MINORS], jnp.float32)

    # ── the price bank (no gradient) ─────────────────────────────────────────
    @jax.jit
    def bank_chunk_fn(kk, minor_buf, maj_buf, k):
        kn, ku = jax.random.split(kk)
        gam, P, eps0, imaj, imin, ck = draw_markets(env, kn, bank_chunk, T, n, prior)
        u = jax.random.randint(ku, (bank_chunk,), 0, k)
        A, rho, H, _, _, _ = jax.vmap(
            lambda g, p, e, im, i0, i2, c, uu: sim(maj_buf, minor_buf, k, g, p, e, im, (i0, i2), c, uu, False)
        )(gam, P, eps0, imaj, imin[0], imin[1], ck, u)
        out = {'gam': gam, 'P': P, 'A': A, 'rho': rho}
        if USE_H:                       # public history only stored if it is a network input
            out['H'] = H
        return out

    def build_bank(kk, minor_buf, maj_buf, k):
        parts = [bank_chunk_fn(jax.random.fold_in(kk, c), minor_buf, maj_buf, k)
                 for c in range(max(1, n_bank // bank_chunk))]
        return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *parts)

    # ── minors' stage on the bank ────────────────────────────────────────────
    def make_opt(n_it, warm):
        """Fresh training: cosine lr -> lr_min. Warm-started fine-tuning: CONSTANT lr_warm
        (a high restart of the cosine knocked good warm starts out of their basin)."""
        if warm:
            return optax.adam(lr_warm)
        return optax.adam(optax.cosine_decay_schedule(lr, n_it, alpha=lr_min / lr))

    def make_minor_stage(n_it, warm=False):
        opt = make_opt(n_it, warm)

        @jax.jit
        def stage(p_stack, key_, bank):
            st = opt.init(p_stack)
            N = bank['A'].shape[0]

            def one(carry, it):
                p_, s_ = carry
                ks, kg = jax.random.split(jax.random.fold_in(key_, it))
                sel = jax.random.randint(ks, (B,), 0, N)
                H = bank['H'][sel] if USE_H else jnp.zeros((B, T, 3), jnp.float32)
                A, rho, P, gam = (bank['A'][sel], bank['rho'][sel],
                                     bank['P'][sel], bank['gam'][sel])

                def loss(pst):
                    def per(p, j, scale, kk):
                        net = eqx.combine(p, minor_static[0])
                        idio = jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
                            jax.random.split(kk, B), env.agent_params[j, 7], T)
                        R = jax.vmap(lambda a, r, h, pp, ii, g: env.bind(g).rollout_minor(
                            net, j, a, r, h, pp, ii[None, :], norm_gamma(g))[0])(A, rho, H, P, idio, gam)
                        return -jnp.mean(R) / scale
                    per_t = jax.vmap(per)(pst, js, scales, jax.random.split(kg, len(MINORS)))
                    return jnp.sum(per_t), per_t
                (_, per_t), g = jax.value_and_grad(loss, has_aux=True)(p_)
                upd, s_new = opt.update(g, s_, p_)
                return (eqx.apply_updates(p_, upd), s_new), per_t
            (pf, _), L = jax.lax.scan(one, (p_stack, st), jnp.arange(n_it))
            return pf, L
        return stage

    # ── major's stage (re-clears the market) ─────────────────────────────────
    def make_major_stage(n_it, warm=False):
        opt0 = make_opt(n_it, warm)

        @jax.jit
        def stage(p, key_, minor_buf, maj_buf, k):
            st = opt0.init(p)

            def one(carry, it):
                p_, s_ = carry
                gam, P, eps0, imaj, imin, ck = draw_markets(env, jax.random.fold_in(key_, it), B0, T, n, prior)

                def loss(pp):
                    net = eqx.combine(pp, maj_static)
                    r0 = jax.vmap(lambda g, P_, e, im, i0, i2, c: sim(
                        net, minor_buf, k, g, P_, e, im, (i0, i2), c, 0, True,
                        myopic=True, ckpt=True)[3])(gam, P, eps0, imaj, imin[0], imin[1], ck)
                    return -jnp.mean(r0) / s0
                l, g = jax.value_and_grad(loss)(p_)
                upd, s_new = opt0.update(g, s_, p_)
                return (eqx.apply_updates(p_, upd), s_new), l
            (pf, _), L = jax.lax.scan(one, (p, st), jnp.arange(n_it))
            return pf, L
        return stage

    minor_fresh, minor_warm = make_minor_stage(iters), make_minor_stage(iters_warm, warm=True)
    major_fresh, major_warm = make_major_stage(iters0), make_major_stage(iters0_warm, warm=True)

    # ── evaluation (identical to utils_majmin) ───────────────────────────────
    @eqx.filter_jit
    def eval_sim(maj_src, use_br, minor_buf, maj_buf, k, b, ck, u):
        src = maj_src if use_br else maj_buf
        return jax.vmap(lambda g, p, e, im, c, uu: sim(
            src, minor_buf, k, g, p, e, im, b['imin'], c, uu, use_br))(
            b['gam'], b['P'], b['eps0'], b['imaj'], ck, u)

    @eqx.filter_jit
    def minor_returns(net, j, A, rho, H, P, gam, idio):
        return jax.vmap(lambda a, r, h, pp, g: env.bind(g).rollout_minor(
            net, j, a, r, h, pp, idio, norm_gamma(g)))(A, rho, H, P, gam)

    def evaluate(b, k, new, minor_buf, maj_buf, hist_minor):
        Sb = b['S']
        ck = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(COMP_SEED), k), Sb)
        u = jnp.arange(Sb) % k
        bj = {kk: b[kk] for kk in ('gam', 'P', 'eps0', 'imaj', 'imin')}
        A, rho, H, r0, ys, (Xf, X0f) = eval_sim(None, False, minor_buf, maj_buf, jnp.int32(k), bj, ck, u)
        _, _, _, r0_br, _, _ = eval_sim(new[MAJOR], True, minor_buf, maj_buf, jnp.int32(k), bj, ck, u)
        gaps, short_t = jax.vmap(lambda g, xf, x0: env.bind(g).minor_gaps(xf, x0))(b['gam'], Xf, X0f)
        per_type = [None] * N_TYPES
        for j in MINORS:
            Rb = minor_returns(new[j], j, A, rho, H, b['P'], b['gam'], b['ieval'][j])
            Rh = jnp.stack([minor_returns(h, j, A, rho, H, b['P'], b['gam'], b['ieval'][j])
                            for h in hist_minor[j]]).mean(0)
            d = jnp.mean(Rb - Rh, axis=1)
            per_type[j] = {'gap': float(d.mean()), 'gap_se': float(d.std() / np.sqrt(Sb)),
                           'gap_relative': float(d.mean()) / s_minor[j],
                           'V_br': float(Rb.mean()), 'V_mix': float(Rh.mean())}
        d0 = r0_br - r0
        per_type[MAJOR] = {'gap': float(d0.mean()), 'gap_se': float(d0.std() / np.sqrt(Sb)),
                           'gap_relative': float(d0.mean()) / s0,
                           'V_br': float(r0_br.mean()), 'V_mix': float(r0.mean())}
        wrel = float(sum(w_ref[j] * per_type[j]['gap_relative'] for j in range(N_TYPES)))
        clr = block_metrics(env, ctx_ref if b['ctx'] is None else b['ctx'], A, rho, ys, gaps, short_t, r0)
        return per_type, wrel, clr, A

    force = os.environ.get('KETS_FORCE_RESTART') == '1'      # test only: exercise the restart path

    def failed(m):
        return force or m['gap'] < -m['gap_se']

    def split_minor(p_stack):
        return {j: eqx.combine(jax.tree_util.tree_map(lambda x: x[a], p_stack), minor_static[0])
                for a, j in enumerate(MINORS)}

    def stack_minor(nets_by_type):
        parts = [eqx.partition(nets_by_type[j], eqx.is_array)[0] for j in MINORS]
        return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *parts)

    names = ("private (minor)", "large public (MAJOR)", "fin. inst. (minor)")
    if verbose:
        print(f"major-minor FAST FP rung {rung}: K={K} T={T} n={n}; minors {iters} steps fresh / "
              f"{iters_warm} warm (constant lr {lr_warm:g}) on a bank of {n_bank} markets (B={B}); major {iters0} fresh / "
              f"{iters0_warm} warm x B0={B0}; warm from iteration {warm_from if warm_from < 10**8 else "never (fresh BRs)"}, one fresh restart on "
              f"failure; bisection {CLEAR_ITERS_V2}/side; devices minor={d_min} major={d_maj}; "
              f"floor_mode={FLOOR_MODE} c={SUPPLY_C:g} ceiling {A_CEIL_V2:g}", flush=True)
    A_prev = {nm: None for nm in blocks}
    prev_new = None if start_k == 0 else nets[-1]
    prev_stuck = {j: False for j in range(N_TYPES)}
    t0 = time.time()
    for k in range(start_k, K):
        ti = time.time()
        kk = len(nets)
        warm = (k + 1) >= warm_from and prev_new is not None
        if verbose:
            done = k - start_k
            eta = "" if done == 0 else f"   ETA {_hms((time.time() - t0) / done * (K - k))}"
            print(f"\n{'=' * 72}\nMAJOR-MINOR FAST rung {rung} — FP ITERATION {k + 1}/{K}  [mixture {kk}]  "
                  f"{'warm' if warm else 'fresh'}  elapsed {_hms(time.time() - t0)}{eta}\n{'=' * 72}", flush=True)
        ka, kb, kn, kbank, kr = jax.random.split(jax.random.fold_in(k_iter, k), 5)
        nk = jax.random.split(kn, N_TYPES)
        mb0 = jax.device_put(tuple(bufs[j] for j in MINORS), d_min)
        jb0 = jax.device_put(bufs[MAJOR], d_min)
        mb1 = jax.device_put(tuple(bufs[j] for j in MINORS), d_maj)
        jb1 = jax.device_put(bufs[MAJOR], d_maj)
        kk32_0 = jax.device_put(jnp.int32(kk), d_min)
        kk32_1 = jax.device_put(jnp.int32(kk), d_maj)

        # initial networks. Never warm-start from a STUCK network (training loss
        # ended at 0 = no trade, the zero-gradient trap): such a type starts from
        # its fresh initialisation instead (the FI from its warm-start regression).
        fresh_minor = {j: minor_policy(nk[j]) for j in MINORS}
        if mm_warm is not None:
            fresh_minor[2] = mm_warm
        if warm:
            m_init = {j: (fresh_minor[j] if prev_stuck[j] else prev_new[j]) for j in MINORS}
            maj_init = major_policy(nk[MAJOR]) if prev_stuck[MAJOR] else prev_new[MAJOR]
        else:
            m_init = fresh_minor
            maj_init = major_policy(nk[MAJOR])
        tb = time.time()
        # (b) major on device d_maj — dispatched first, runs asynchronously
        p0 = jax.device_put(eqx.partition(maj_init, eqx.is_array)[0], d_maj)
        maj_out = (major_warm if warm else major_fresh)(p0, jax.device_put(kb, d_maj), mb1, jb1, kk32_1)
        t_major_done = [None]

        def _wait_major():
            jax.block_until_ready(maj_out)
            t_major_done[0] = time.time()
        th = threading.Thread(target=_wait_major); th.start()
        # (a) minors on device d_min: price bank, then best responses on the bank
        bank = build_bank(jax.device_put(kbank, d_min), mb0, jb0, kk32_0)
        jax.block_until_ready(bank)
        t_bank = time.time() - tb
        ps = jax.device_put(stack_minor(m_init), d_min)
        min_out = (minor_warm if warm else minor_fresh)(ps, jax.device_put(ka, d_min), bank)
        p_stack, Lm = min_out
        Lm = np.asarray(Lm)
        t_minor = time.time() - tb
        p0, L0 = maj_out
        L0 = np.asarray(L0)
        th.join()
        t_major = t_major_done[0] - tb
        t_both = time.time() - tb
        new = [None] * N_TYPES
        sm = split_minor(jax.device_get(p_stack))
        for j in MINORS:
            new[j] = sm[j]
        new[MAJOR] = eqx.combine(jax.device_get(p0), maj_static)
        if verbose:
            tail = lambda L: L[-max(1, len(L) // 20):].mean(0)
            print(f"  BR phase {_hms(t_both)} (bank {_hms(t_bank)}, bank+minors {_hms(t_minor)}, major {_hms(t_major)}): loss t0 "
                  f"{Lm[0, 0]:+.4f}->{tail(Lm)[0]:+.4f}  t2 {Lm[0, 1]:+.4f}->{tail(Lm)[1]:+.4f}   "
                  f"major {L0[0]:+.4f}->{tail(L0):+.4f}", flush=True)

        tailm = Lm[-max(1, len(Lm) // 20):].mean(0)
        stuck_now = {MINORS[0]: abs(float(tailm[0])) < STUCK_TOL,
                     MINORS[1]: abs(float(tailm[1])) < STUCK_TOL,
                     MAJOR: abs(float(L0[-max(1, len(L0) // 20):].mean())) < STUCK_TOL}
        minor_buf = tuple(bufs[j] for j in MINORS)
        hist_minor = {j: [tr[j] for tr in nets] for j in MINORS}
        restarted = []
        if warm:
            per_type, _, _, _ = evaluate(blocks[main_block], kk, new, minor_buf, bufs[MAJOR], hist_minor)
            # failure = negative exploitability beyond one s.e., or stuck at no trade
            bad_minor = [j for j in MINORS if failed(per_type[j]) or stuck_now[j]]
            bad_major = failed(per_type[MAJOR]) or stuck_now[MAJOR]
            if bad_minor or bad_major:
                tr = time.time()
                if bad_major:
                    nkr = jax.random.split(kr, N_TYPES)
                    p0r = jax.device_put(eqx.partition(major_policy(nkr[MAJOR]), eqx.is_array)[0], d_maj)
                    maj_r = major_fresh(p0r, jax.device_put(jax.random.fold_in(kr, 1), d_maj), mb1, jb1, kk32_1)
                if bad_minor:
                    nkr = jax.random.split(jax.random.fold_in(kr, 2), N_TYPES)
                    mf = {j: minor_policy(nkr[j]) for j in MINORS}
                    if mm_warm is not None:
                        mf[2] = mm_warm
                    pr_, Lr = minor_fresh(jax.device_put(stack_minor(mf), d_min),
                                          jax.device_put(jax.random.fold_in(kr, 3), d_min), bank)
                    smr = split_minor(jax.device_get(pr_))
                    Lr = np.asarray(Lr)
                    tr_ = Lr[-max(1, len(Lr) // 20):].mean(0)
                    for a, j in enumerate(MINORS):
                        if j in bad_minor:
                            new[j] = smr[j]
                            stuck_now[j] = abs(float(tr_[a])) < STUCK_TOL
                if bad_major:
                    new[MAJOR] = eqx.combine(jax.device_get(maj_r[0]), maj_static)
                    L0r = np.asarray(maj_r[1])
                    stuck_now[MAJOR] = abs(float(L0r[-max(1, len(L0r) // 20):].mean())) < STUCK_TOL
                restarted = bad_minor + ([MAJOR] if bad_major else [])
                if verbose:
                    print(f"  RESTART (fresh seed, full budget) for types {restarted}  "
                          f"[{_hms(time.time() - tr)}]", flush=True)
        del bank
        t_br = time.time() - tb
        if save_dir:
            os.makedirs(os.path.join(save_dir, 'losses'), exist_ok=True)
            np.savez_compressed(os.path.join(save_dir, 'losses', f'br_loss_{k + 1:03d}.npz'),
                                minors=Lm.astype(np.float32), major=L0.astype(np.float32))

        rec_blocks = {}
        for nm, b in blocks.items():
            per_type, wrel, clr, A = evaluate(b, kk, new, minor_buf, bufs[MAJOR], hist_minor)
            pl2 = float('nan') if A_prev[nm] is None else float(jnp.mean(jnp.linalg.norm(A - A_prev[nm], axis=1)))
            A_prev[nm] = A
            rec_blocks[nm] = {'per_type': per_type, 'weighted_relative': wrel, 'path_l2': pl2,
                              'clearing': clr, 'A_on_floor': clr['A_on_floor']}
            if verbose:
                print(f"  [{nm}] price mean {clr['A_cleared_mean']:.2f}  A_0 {clr['A0_cleared']:.2f}  "
                      f"floor {clr['A_on_floor']:.1%} ceil {clr['A_on_ceil']:.1%} flat {clr['clear_flat_frac']:.1%}  "
                      f"resid {clr['clear_residual_max_rel']:.1e} cons {clr['cons_err_max_rel']:.1e}  "
                      f"impact {clr['major_impact_mean']:.3g} (guard {clr['ift_guard_frac']:.1%})  "
                      f"markdown {clr['major_markdown']:+.2f}  major f {clr['major_f_mean']:+.2f}  "
                      f"short {clr['market_short_frac']:.0%}", flush=True)
                for j in range(N_TYPES):
                    m = per_type[j]
                    flag = "  <-- NEGATIVE" if failed(m) else ""
                    print(f"      {names[j]:22s} E = {m['gap']:+11.1f} +/- {m['gap_se']:8.1f}  "
                          f"({m['gap_relative']:+.4%}){flag}", flush=True)
                print(f"      weighted {wrel:+.4%}   ||A^k-A^k-1|| {pl2:.4g}", flush=True)

        nets.append(new)
        prev_new = new
        prev_stuck = stuck_now
        for j in range(N_TYPES):
            bufs[j] = buffer_set(bufs[j], kk, new[j])
        rec = {'iteration': k + 1, 'mixture_size': kk, 'warm': bool(warm), 'restarted': restarted,
               'stuck': [j for j in range(N_TYPES) if stuck_now[j]],
               'br_seconds': t_br, 'bank_seconds': t_bank, 'minor_seconds': t_minor, 'major_seconds': t_major, 'iter_seconds': time.time() - ti,
               'major_loss_end': float(L0[-max(1, len(L0) // 20):].mean())}
        if rung == 2:
            rec.update(rec_blocks['ref'])
        else:
            rec['blocks'] = rec_blocks
        records.append(rec)
        if save_dir:
            for j in range(N_TYPES):
                save_net(save_dir, k + 1, j, new[j])
            _atomic_json(ppath, {
                'rung': rung, 'version': 'major_minor_fast_v1', 'context': list(ctx_ref),
                'fixed_scenarios': {nm: (b['ctx'] if b['ctx'] is not None else None) for nm, b in blocks.items()},
                'K': K, 'T_grid': [T], 'n_sim': n, 'iters': [iters], 'iters_major': iters0,
                'iters_warm': iters_warm, 'iters_major_warm': iters0_warm, 'warm_from': warm_from,
                'lr_warm': lr_warm, 'warm_schedule': 'constant',
                'n_bank': n_bank, 'clear_iters': CLEAR_ITERS_V2, 'devices': [str(d_min), str(d_maj)],
                'B': B, 'B0': B0, 'S': S, 'S_fixed': S_fixed, 'M': M, 'lr': lr, 'lr_min': lr_min,
                'floor_mode': FLOOR_MODE, 'supply_c': SUPPLY_C, 'A_CEIL': A_CEIL_V2,
                'N0': 1, 'major_type': MAJOR, 'minor_types': list(MINORS),
                'major_gradient': 'myopic_ift', 'public_history_input': bool(USE_H), 'm0_ref': m0_ref, 's0': s0,
                'type_shares': [float(x) for x in w_ref], **(meta or {}), 'records': records})
        if verbose:
            print(f"  checkpoint saved  [iteration {_hms(time.time() - ti)}]", flush=True)
    return nets, records
