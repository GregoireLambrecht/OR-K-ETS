"""Fictitious play for the major–minor market (MAJOR_MINOR_CLEARING.md, Algorithm 3).

One code path for both rungs: rung 2 is rung 3 with the prior Q a point mass at
the reference scenario (the networks still take gamma through FiLM; it is just
constant). Per FP iteration k:

  (a) minors' best responses (types 0 and 2), price-taking: B markets per step,
      gamma ~ Q, common noise, the major following a component u ~ U{0..k-1}
      drawn PER MARKET, minors following stratified components PER AGENT; the
      market is cleared without gradient and the deviators roll out against the
      frozen (A, rho, H);
  (b) the major's best response: B0 markets per step, the deviating major INSIDE
      the clearing, gradient through the clearing by the IFT custom JVP, myopic
      regime (stop_gradient on the minors' schedules and states);
  (c) the three networks go into slot k of their buffers (fixed K_MAX buffers,
      k a traced integer: compile once per solve);
  (d) evaluation on frozen blocks:
      minors E^j = V^j(new BR) - J^j on the same frozen (A, rho, H), J^j exact
      over the minor components; major J^0 = stratified over u (market s uses
      u = s mod k), V^0 = the same markets re-simulated with the BR major inside
      the clearing, E^0 = V^0 - J^0 (paired).
"""
import os
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from envs.environment import N_TYPES
from envs.environment_majmin import MINORS, MAJOR
from envs.models_majmin import (minor_policy, major_policy, make_buffer, buffer_set,
                                component, SchedulePolicy)
from envs.params_v2 import K_MAX, FLOOR_MODE, SUPPLY_C, A_CEIL_V2
from utils import (_hms, _atomic_json, norm_gamma, FIXED_SCENARIOS, VERIFY_SEED)

COMP_SEED = 31337
RUNG3_FIXED = ('ref', 'alloc_lo', 'kappa_sub', 'floor_hi', 'mm_thick')


def save_net(save_dir, k, j, net):
    eqx.tree_serialise_leaves(os.path.join(save_dir, f'policy_{k}_type{j}.eqx'), net)


def load_net(save_dir, k, j):
    like = major_policy(jax.random.PRNGKey(0)) if j == MAJOR else minor_policy(jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(os.path.join(save_dir, f'policy_{k}_type{j}.eqx'), like)


# ── noise ─────────────────────────────────────────────────────────────────────

def draw_markets(env, key, S, T, n, prior):
    """S markets: gamma (S,4), P (S,T+1), eps0 (S,T), raw major emission noise
    (S,T) (type-1 sigma; x m0 per market), minor population noise (S, n, T) per type,
    component keys (S,)."""
    kg, kP, kE, kM, kI, kC = jax.random.split(key, 6)
    gam = prior(kg, S)
    P = jax.vmap(env.generate_P, in_axes=(0, None))(jax.random.split(kP, S), T)
    eps0 = jax.vmap(env.generate_eps0, in_axes=(0, None))(jax.random.split(kE, S), T)
    imaj = jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
        jax.random.split(kM, S), env.agent_params[MAJOR, 7], T)
    ki = jax.random.split(kI, len(MINORS))
    imin = tuple(jax.vmap(lambda kk, j=j: jax.vmap(
        env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
            jax.random.split(kk, n), env.agent_params[j, 7], T))(jax.random.split(ki[a], S))
        for a, j in enumerate(MINORS))
    return gam, P, eps0, imaj, imin, jax.random.split(kC, S)


# ── one market, vmapped ──────────────────────────────────────────────────────

def make_sim(env, mstatic, majstatic):
    """sim(major, maj_is_net, minor_buf, k, g, P, eps0, imaj_raw, imin, ck, u, myopic, ckpt)."""
    def sim(major_src, minor_buf, k, g, P, eps0, imaj_raw, imin, ck, u, use_br,
            myopic=False, ckpt=False):
        envg = env.bind(g)
        ctx = norm_gamma(g)
        if use_br:
            major_fn = lambda inp, c: major_src(inp, c)
        else:
            net = component(major_src, majstatic, u)
            major_fn = lambda inp, c: net(inp, c)
        imaj = imaj_raw * envg.major_mass()
        return envg.simulate(major_fn, minor_buf, mstatic, k, ck, ctx, P, eps0, imin, imaj,
                             myopic=myopic, checkpoint=ckpt)
    return sim


def block_metrics(env, g_ref, A, rho, ys, gaps_agg, short_t, r0):
    """Clearing and market-power diagnostics over a block (numpy)."""
    envr = env.bind(jnp.asarray(g_ref, jnp.float32))
    cap = float(envr.cap_total())
    A, rho = np.asarray(A), np.asarray(rho)
    Ac = A[:, 1:]
    d = {k: np.asarray(v) for k, v in ys.items()}
    fl, ce = d['on_floor'].astype(bool), d['on_ceil'].astype(bool)
    bound = fl | ce
    off = ~bound[:, 1:] & ~bound[:, :-1]
    drift = np.diff(Ac, axis=1)[off] if off.any() else np.zeros(1)
    mv = d['moving'].astype(bool)
    G_floor = np.where(fl, d['G'], 0.0).sum(axis=1)
    return {
        'cap_per_step': cap,
        'A_on_floor': float(fl.mean()), 'A_on_ceil': float(ce.mean()),
        'clear_flat_frac': float(d['flat'].mean()),
        'rho_sell_mean_on_floor': float(rho[:, 1:, 0][fl].mean()) if fl.any() else 1.0,
        'rho_sell_min': float(rho[..., 0].min()), 'rho_buy_min': float(rho[..., 1].min()),
        'blocked_volume': float(G_floor.mean()),
        'support_cost': float(G_floor.mean() * float(envr.Afloor)),
        'clear_residual_max_rel': float(d['resid'].max() / cap),
        'cons_err_max_rel': float(np.abs(d['cons_err']).max() / cap),
        'price_drift_mean': float(drift.mean()), 'price_drift_std': float(drift.std()),
        'A0_cleared': float(Ac[:, 0].mean()), 'A_cleared_mean': float(Ac.mean()),
        'major_impact_mean': float(d['impact'][mv].mean()) if mv.any() else 0.0,
        'major_impact_max': float(d['impact'].max()),
        'ift_guard_frac': float(d['guard'][mv].mean()) if mv.any() else 0.0,
        'major_markdown': float(np.median(d['R0'] - Ac)),
        'major_beta_mean': float(d['beta0'].mean()),
        'major_f_mean': float(d['f0'].mean()),
        'major_f_profile': [float(x) for x in d['f0'].mean(axis=0)],
        'minor_f_mean': [float(x) for x in d['f_minor'].mean(axis=(0, 1))],
        'market_short_frac': float((np.asarray(gaps_agg) > 0).mean()),
        'short_frac_type': [float(x) for x in np.asarray(short_t).mean(axis=0)],
        'J_major_mean': float(np.asarray(r0).mean()),
        'A_path_mean': [float(x) for x in A.mean(axis=0)],
        'A_path_p10': [float(x) for x in np.quantile(A, 0.1, axis=0)],
        'A_path_p90': [float(x) for x in np.quantile(A, 0.9, axis=0)],
        'rho_sell_path_mean': [float(x) for x in rho[..., 0].mean(axis=0)],
    }


# ── fictitious play ──────────────────────────────────────────────────────────

def fictitious_play_majmin(env, rung, ctx_ref, K=30, T=30, n=30, iters=8000, iters0=4000,
                           lr=1e-3, lr_min=1e-4, B=256, B0=64, S=200, S_fixed=100, M=32,
                           chunk=50, key=None, save_dir=None, mm_warmstart=None,
                           verbose=True, meta=None):
    assert K < K_MAX
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

    # frozen evaluation blocks
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
        imin = tuple(x[0] for x in imin)   # population noise shared across the block's markets
        kE = jax.random.fold_in(kv, 100 + bi)
        ieval = {j: jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
            jax.random.split(jax.random.fold_in(kE, j), M), env.agent_params[j, 7], T) for j in MINORS}
        blocks[nm] = dict(gam=gam, P=P, eps0=eps0, imaj=imaj, imin=imin, ieval=ieval, S=Sb,
                          ctx=None if c is None else list(c))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    ppath = os.path.join(save_dir, 'progress.json') if save_dir else None

    # history / resume
    records, start_k = [], 0
    nets = None
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

    def build_buffers(nets):
        bufs, stats = [], []
        for j in range(N_TYPES):
            b, st = make_buffer(nets[0][j], K_MAX)
            for i, tr in enumerate(nets):
                b = buffer_set(b, i, tr[j])
            bufs.append(b); stats.append(st)
        return bufs, stats
    bufs, stats = build_buffers(nets)
    minor_static = tuple(stats[j] for j in MINORS)
    maj_static = stats[MAJOR]
    sim = make_sim(env, minor_static, maj_static)

    # ── (a) minors' BR stage ─────────────────────────────────────────────────
    opt = optax.adam(optax.cosine_decay_schedule(lr, iters, alpha=lr_min / lr))
    opt0 = optax.adam(optax.cosine_decay_schedule(lr, iters0, alpha=lr_min / lr))
    js = jnp.array(MINORS)
    scales = jnp.array([s_minor[j] for j in MINORS], jnp.float32)

    @jax.jit
    def minor_stage(p_stack, st, loop_key, minor_buf, maj_buf, k):
        def one(carry, it):
            p_, s_ = carry
            kn, kg, ku = jax.random.split(jax.random.fold_in(loop_key, it), 3)
            gam, P, eps0, imaj, imin, ck = draw_markets(env, kn, B, T, n, prior)
            u = jax.random.randint(ku, (B,), 0, k)
            A, rho, H, _, _, _ = jax.vmap(
                lambda g, p, e, im, i0, i2, c, uu: sim(maj_buf, minor_buf, k, g, p, e, im,
                                                       (i0, i2), c, uu, False)
            )(gam, P, eps0, imaj, imin[0], imin[1], ck, u)
            A, rho, H = jax.lax.stop_gradient((A, rho, H))

            def loss(pst):
                def per(p, j, scale, kk):
                    net = eqx.combine(p, minor_static[0])
                    idio = jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
                        jax.random.split(kk, B), env.agent_params[j, 7], T)
                    R = jax.vmap(lambda a, r, h, pp, ii, g: env.bind(g).rollout_minor(
                        net, j, a, r, h, pp, ii[None, :], norm_gamma(g))[0])(
                        A, rho, H, P, idio, gam)
                    return -jnp.mean(R) / scale
                per_t = jax.vmap(per)(pst, js, scales, jax.random.split(kg, len(MINORS)))
                return jnp.sum(per_t), per_t
            (_, per_t), g = jax.value_and_grad(loss, has_aux=True)(p_)
            upd, s_new = opt.update(g, s_, p_)
            return (eqx.apply_updates(p_, upd), s_new), per_t
        (pf, sf), L = jax.lax.scan(one, (p_stack, st), jnp.arange(iters))
        return pf, L

    # ── (b) major's BR stage ─────────────────────────────────────────────────
    @jax.jit
    def major_stage(p, st, loop_key, minor_buf, maj_buf, k):
        def one(carry, it):
            p_, s_ = carry
            kn = jax.random.fold_in(loop_key, it)
            gam, P, eps0, imaj, imin, ck = draw_markets(env, kn, B0, T, n, prior)

            def loss(pp):
                net = eqx.combine(pp, maj_static)
                r0 = jax.vmap(lambda g, P_, e, im, i0, i2, c: sim(
                    net, minor_buf, k, g, P_, e, im, (i0, i2), c, 0, True,
                    myopic=True, ckpt=True)[3])(gam, P, eps0, imaj, imin[0], imin[1], ck)
                return -jnp.mean(r0) / s0
            l, g = jax.value_and_grad(loss)(p_)
            upd, s_new = opt0.update(g, s_, p_)
            return (eqx.apply_updates(p_, upd), s_new), l
        (pf, sf), L = jax.lax.scan(one, (p, st), jnp.arange(iters0))
        return pf, L

    # ── evaluation ───────────────────────────────────────────────────────────
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
            Rb = minor_returns(new[j], j, A, rho, H, b['P'], b['gam'], b['ieval'][j])   # (S, M)
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

    names = ("private (minor)", "large public (MAJOR)", "fin. inst. (minor)")
    if verbose:
        print(f"major-minor FP rung {rung}: K={K} T={T} n={n}/minor type, minors {iters} steps x B={B}, "
              f"major {iters0} steps x B0={B0} (myopic IFT gradient), floor_mode={FLOOR_MODE}, "
              f"supply c={SUPPLY_C:g}, ceiling {A_CEIL_V2:g}, m0_ref={m0_ref:.3f}, s0={s0:.4g}", flush=True)
    A_prev = {nm: None for nm in blocks}
    t0 = time.time()
    for k in range(start_k, K):
        ti = time.time()
        kk = len(nets)
        if verbose:
            done = k - start_k
            eta = "" if done == 0 else f"   ETA {_hms((time.time() - t0) / done * (K - k))}"
            print(f"\n{'=' * 72}\nMAJOR-MINOR rung {rung} — FP ITERATION {k + 1}/{K}  [mixture {kk}]  "
                  f"elapsed {_hms(time.time() - t0)}{eta}\n{'=' * 72}", flush=True)
        minor_buf = tuple(bufs[j] for j in MINORS)
        kit = jax.random.fold_in(k_iter, k)
        ka, kb, kn = jax.random.split(kit, 3)
        # (a) minors
        nk = jax.random.split(kn, N_TYPES)
        m_init = [minor_policy(nk[j]) for j in MINORS]
        if mm_warm is not None:
            m_init[1] = mm_warm                           # financial institution
        parts = [eqx.partition(m, eqx.is_array)[0] for m in m_init]
        p_stack = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *parts)
        tb = time.time()
        p_stack, Lm = minor_stage(p_stack, opt.init(p_stack), ka, minor_buf, bufs[MAJOR], jnp.int32(kk))
        Lm = np.asarray(Lm)
        new = [None] * N_TYPES
        for a, j in enumerate(MINORS):
            new[j] = eqx.combine(jax.tree_util.tree_map(lambda x: x[a], p_stack), minor_static[0])
        t_minor = time.time() - tb
        # (b) major
        tb = time.time()
        p0, _ = eqx.partition(major_policy(nk[MAJOR]), eqx.is_array)
        p0, L0 = major_stage(p0, opt0.init(p0), kb, minor_buf, bufs[MAJOR], jnp.int32(kk))
        L0 = np.asarray(L0)
        new[MAJOR] = eqx.combine(p0, maj_static)
        t_major = time.time() - tb
        if verbose:
            tail = lambda L: L[-max(1, len(L) // 20):].mean(0)
            print(f"  minors BR {_hms(t_minor)}: loss t0 {Lm[0, 0]:+.4f}->{tail(Lm)[0]:+.4f}  "
                  f"t2 {Lm[0, 1]:+.4f}->{tail(Lm)[1]:+.4f}   major BR {_hms(t_major)}: "
                  f"loss {L0[0]:+.4f}->{tail(L0):+.4f}", flush=True)
        if save_dir:
            os.makedirs(os.path.join(save_dir, 'losses'), exist_ok=True)
            np.savez_compressed(os.path.join(save_dir, 'losses', f'br_loss_{k + 1:03d}.npz'),
                                minors=Lm.astype(np.float32), major=L0.astype(np.float32))

        # (d) evaluation against the CURRENT mixture (before the update)
        hist_minor = {j: [tr[j] for tr in nets] for j in MINORS}
        rec_blocks = {}
        for nm, b in blocks.items():
            per_type, wrel, clr, A = evaluate(b, kk, new, minor_buf, bufs[MAJOR], hist_minor)
            pl2 = float('nan') if A_prev[nm] is None else float(
                jnp.mean(jnp.linalg.norm(A - A_prev[nm], axis=1)))
            A_prev[nm] = A
            rec_blocks[nm] = {'per_type': per_type, 'weighted_relative': wrel,
                              'path_l2': pl2, 'clearing': clr, 'A_on_floor': clr['A_on_floor']}
            if verbose:
                print(f"  [{nm}] price mean {clr['A_cleared_mean']:.2f}  A_0 {clr['A0_cleared']:.2f}  "
                      f"floor {clr['A_on_floor']:.1%} ceil {clr['A_on_ceil']:.1%} flat {clr['clear_flat_frac']:.1%}  "
                      f"resid {clr['clear_residual_max_rel']:.1e} cons {clr['cons_err_max_rel']:.1e}  "
                      f"impact {clr['major_impact_mean']:.3g} (guard {clr['ift_guard_frac']:.1%})  "
                      f"markdown {clr['major_markdown']:+.2f}  major f {clr['major_f_mean']:+.2f}  "
                      f"short {clr['market_short_frac']:.0%}", flush=True)
                for j in range(N_TYPES):
                    m = per_type[j]
                    flag = "  <-- NEGATIVE" if m['gap'] < -m['gap_se'] else ""
                    print(f"      {names[j]:22s} E = {m['gap']:+11.1f} +/- {m['gap_se']:8.1f}  "
                          f"({m['gap_relative']:+.4%}){flag}", flush=True)
                print(f"      weighted {wrel:+.4%}   ||A^k-A^k-1|| {pl2:.4g}", flush=True)

        # (c) update the mixtures
        nets.append(new)
        for j in range(N_TYPES):
            bufs[j] = buffer_set(bufs[j], kk, new[j])
        rec = {'iteration': k + 1, 'mixture_size': kk, 'minor_seconds': t_minor,
               'major_seconds': t_major, 'iter_seconds': time.time() - ti,
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
                'rung': rung, 'version': 'major_minor_v1', 'context': list(ctx_ref),
                'fixed_scenarios': {nm: (b['ctx'] if b['ctx'] is not None else None)
                                    for nm, b in blocks.items()},
                'K': K, 'T_grid': [T], 'n_sim': n, 'iters': [iters], 'iters_major': iters0,
                'B': B, 'B0': B0, 'S': S, 'S_fixed': S_fixed, 'M': M, 'lr': lr, 'lr_min': lr_min,
                'floor_mode': FLOOR_MODE, 'supply_c': SUPPLY_C, 'A_CEIL': A_CEIL_V2,
                'N0': 1, 'major_type': MAJOR, 'minor_types': list(MINORS),
                'major_gradient': 'myopic_ift', 'm0_ref': m0_ref, 's0': s0,
                'type_shares': [float(x) for x in w_ref], **(meta or {}), 'records': records})
        if verbose:
            print(f"  checkpoint saved  [iteration {_hms(time.time() - ti)}]", flush=True)
    return nets, records
