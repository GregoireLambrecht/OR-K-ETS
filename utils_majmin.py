"""Fictitious play for the major–minor market (MAJOR_MINOR_CLEARING.md, Algorithm 3).

One code path for both rungs: rung 2 is rung 3 with the prior Q a point mass at
the reference scenario (the networks still take gamma through FiLM; it is just
constant). Per FP iteration k:

  (a) minors' best responses (types 0 and 2), price-taking, on a PRICE BANK:
      the population is frozen during a best response, so once per iteration
      N_bank markets are cleared WITHOUT gradient (gamma ~ Q, common noise, the
      major following a component u ~ U{0..k-1} per market, minors following
      stratified components per agent) and their public paths (gamma, P, A,
      rho, H) are stored. Every gradient step samples B of them; the deviator's
      own idiosyncratic noise stays fresh;
  (b) the major's best response: B0 fresh markets per step, the deviating major
      INSIDE the clearing, gradient through the clearing by the IFT custom JVP,
      myopic regime (stop_gradient on the minors' schedules and states). It
      runs concurrently with (a) on a second GPU when there is one;
  (c) best responses are warm-started from the previous iteration's from
      iteration `warm_from` on (smaller budget, constant lr), else trained
      fresh (cosine lr). Every best response is CHECKED on the main frozen
      block (prior for rung 3, ref for rung 2): a type FAILS if its
      exploitability is negative beyond one standard error, if its training
      loss is stuck at 0 (no trade; not checked without a major, where no trade
      can be the best response), or, for the major, if its training loss
      ended above where it started (diverged). A failed type is retrained once
      from a fresh seed with the full budget; if it still fails, its previous
      best response is reused for this iteration (recorded as 'rejected');
  (d) the three networks go into slot k of their buffers (fixed K_MAX buffers,
      k a traced integer: compile once per solve);
  (e) evaluation on frozen blocks:
      minors E^j = V^j(new BR) - J^j on the same frozen (A, rho, H), J^j exact
      over the minor components; major J^0 = stratified over u (market s uses
      u = s mod k), V^0 = the same markets re-simulated with the BR major inside
      the clearing, E^0 = V^0 - J^0 (paired).
The bisection length of the clearing is KETS_CLEAR_ITERS (set by the caller).
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
from envs.models_majmin import (minor_policy, major_policy, make_buffer, buffer_set,
                                component, SchedulePolicy)
from envs.params_v2 import K_MAX, FLOOR_MODE, SUPPLY_C, A_CEIL_V2, CLEAR_ITERS_V2
from envs.base_params import LAYOUT, C_MIN, S_MIN, NO_MAJOR, ALLOC_MIN, ALLOC_MAX, KAPPA_MIN, KAPPA_MAX, FLOOR_MIN, FLOOR_MAX, MM_MIN, MM_MAX
from utils import _hms, _atomic_json, norm_gamma, VERIFY_SEED, CTX_REF

COMP_SEED = 31337
WARM_FROM = 2          # iteration (1-based) from which best responses are warm-started
STUCK_TOL = 1e-3       # |tail-mean training loss| below this = stuck at no trade (losses are O(0.1-1))
DIVERGE_MARGIN = 0.25  # major BR diverged: tail loss > head loss + margin * |head loss|


def save_net(save_dir, k, j, net):
    eqx.tree_serialise_leaves(os.path.join(save_dir, f'policy_{k}_type{j}.eqx'), net)


def load_net(save_dir, k, j):
    like = major_policy(jax.random.PRNGKey(0)) if j == MAJOR else minor_policy(jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(os.path.join(save_dir, f'policy_{k}_type{j}.eqx'), like)


def rung3_scenarios():
    """Fixed evaluation scenarios of rung 3, all INSIDE the training box: the
    reference (clipped to the box) and the reference moved to one edge per axis."""
    if LAYOUT == 'v5':                 # (c, s, Afloor, mm_share, kappa2)
        from envs.base_params import C_MAX, S_MAX
        lo = (C_MIN, S_MIN, FLOOR_MIN, MM_MIN, KAPPA_MIN)
        hi = (C_MAX, S_MAX, FLOOR_MAX, MM_MAX, KAPPA_MAX)
        ref = tuple(float(min(max(c, l), h)) for c, l, h in zip(CTX_REF, lo, hi))

        def at5(i, v):
            r = list(ref)
            r[i] = float(v)
            return tuple(r)
        return (('ref', ref), ('c_lo', at5(0, C_MIN)), ('s_lo', at5(1, S_MIN)),
                ('floor_hi', at5(2, FLOOR_MAX)), ('mm_thick', at5(3, MM_MAX)), ('kappa_lo', at5(4, KAPPA_MIN)))
    lo = (ALLOC_MIN, KAPPA_MIN, FLOOR_MIN, MM_MIN)
    hi = (ALLOC_MAX, KAPPA_MAX, FLOOR_MAX, MM_MAX)
    ref = tuple(float(min(max(c, l), h)) for c, l, h in zip(CTX_REF, lo, hi))

    def at(i, v):
        r = list(ref)
        r[i] = float(v)
        return tuple(r)
    return (('ref', ref), ('alloc_lo', at(0, ALLOC_MIN)), ('kappa_lo', at(1, KAPPA_MIN)),
            ('floor_hi', at(2, FLOOR_MAX)), ('mm_thick', at(3, MM_MAX)))


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


def make_prior(env, rung, ctx_ref):
    if rung == 2:
        g_ref = jnp.asarray(ctx_ref, jnp.float32)
        return lambda kk, b: jnp.broadcast_to(g_ref, (b, g_ref.shape[0]))
    return lambda kk, b: env.generate_context(kk, b)


# ── one market, vmapped ──────────────────────────────────────────────────────

def make_sim(env, mstatic, majstatic):
    """sim(major_src, minor_buf, k, g, P, eps0, imaj_raw, imin, ck, u, use_br, myopic, ckpt):
    the major is the network `major_src` if use_br, else component u of the buffer `major_src`."""
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


# ── frozen evaluation blocks and the evaluation ──────────────────────────────

def make_blocks(env, rung, ctx_ref, T, n, S, S_fixed, M):
    """Frozen evaluation blocks, keyed by VERIFY_SEED. Block bi draws from
    fold_in(VERIFY_SEED, bi): the order (prior first, then ref) must not change."""
    prior = make_prior(env, rung, ctx_ref)
    kv = jax.random.PRNGKey(VERIFY_SEED)
    if rung == 2:
        specs = [('ref', ctx_ref, S)]
    else:
        specs = [('prior', None, S)] + [(nm, c, S_fixed) for nm, c in rung3_scenarios()]
    blocks = {}
    for bi, (nm, c, Sb) in enumerate(specs):
        pr = prior if c is None else (lambda kk, b, c=c: jnp.broadcast_to(jnp.asarray(c, jnp.float32), (b, len(c))))
        gam, P, eps0, imaj, imin, _ = draw_markets(env, jax.random.fold_in(kv, bi), Sb, T, n, pr)
        imin = tuple(x[0] for x in imin)
        kE = jax.random.fold_in(kv, 100 + bi)
        ieval = {j: jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
            jax.random.split(jax.random.fold_in(kE, j), M), env.agent_params[j, 7], T) for j in MINORS}
        blocks[nm] = dict(gam=gam, P=P, eps0=eps0, imaj=imaj, imin=imin, ieval=ieval, S=Sb,
                          ctx=None if c is None else list(c))
    return blocks, ('ref' if rung == 2 else 'prior')


def make_evaluator(env, sim, ctx_ref, major_price_taker=False):
    """evaluate(b, k, new, minor_buf, maj_buf, hist_minor) -> (per_type, weighted, clearing, A):
    exploitability of the networks `new` against the mixture of the first k
    buffer slots (hist_minor: the minors' networks of those slots). With
    major_price_taker (job 2 of RUNS_SETTING.md) the major's best response is valued
    like a minor's, on the mixture market's frozen paths (rollout_major), not re-cleared."""
    g_ref = jnp.asarray(ctx_ref, jnp.float32)
    s0 = float(env.major_reward_scale(float(env.bind(g_ref).major_mass())))
    s_minor = {j: float(env.reward_scale(j)) for j in MINORS}
    w_ref = np.asarray(env.bind(g_ref).context_weights())

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

    @eqx.filter_jit
    def major_frozen(net, A, rho, H, P, imaj_raw, gam):
        return jax.vmap(lambda a, r, h, pp, im, g: env.bind(g).rollout_major(
            net, a, r, h, pp, im * env.bind(g).major_mass(), norm_gamma(g)))(A, rho, H, P, imaj_raw, gam)

    def evaluate(b, k, new, minor_buf, maj_buf, hist_minor):
        Sb = b['S']
        ck = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(COMP_SEED), k), Sb)
        u = jnp.arange(Sb) % k
        bj = {kk: b[kk] for kk in ('gam', 'P', 'eps0', 'imaj', 'imin')}
        A, rho, H, r0, ys, (Xf, X0f) = eval_sim(None, False, minor_buf, maj_buf, jnp.int32(k), bj, ck, u)
        if NO_MAJOR:                    # no major (LP benchmark): nothing to deviate
            r0_br = r0
        elif major_price_taker:         # the price-taking deviation, on the frozen paths
            r0_br = major_frozen(new[MAJOR], A, rho, H, b['P'], b['imaj'], b['gam'])
        else:
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
    return evaluate


def make_buffers(nets):
    """K_MAX buffers per type holding nets[i][j] in slot i."""
    bufs, stats = [], []
    for j in range(N_TYPES):
        b, st = make_buffer(nets[0][j], K_MAX)
        for i, tr in enumerate(nets):
            b = buffer_set(b, i, tr[j])
        bufs.append(b); stats.append(st)
    return bufs, stats


# ── fictitious play ──────────────────────────────────────────────────────────

def _tail(L):
    return L[-max(1, len(L) // 20):].mean(0)


def _head(L):
    return L[:max(1, len(L) // 20)].mean(0)


def fictitious_play_majmin(env, rung, ctx_ref, K=30, T=30, n=30, iters=8000, iters0=4000,
                           iters_warm=2000, iters0_warm=1000, lr_warm=1e-4, n_bank=65536, bank_chunk=8192,
                           warm_from=WARM_FROM,
                           lr=1e-3, lr_min=1e-4, B=256, B0=64, S=200, S_fixed=100, M=32,
                           key=None, save_dir=None, mm_warmstart=None, verbose=True, meta=None,
                           major_price_taker=False):
    assert K < K_MAX
    bank_chunk = min(bank_chunk, n_bank)
    key = jax.random.PRNGKey(0) if key is None else key
    k_init, k_iter = jax.random.split(key)
    env = env.set_T(T)
    g_ref = jnp.asarray(ctx_ref, jnp.float32)
    prior = make_prior(env, rung, ctx_ref)
    m0_ref = float(env.bind(g_ref).major_mass())
    s0 = float(env.major_reward_scale(m0_ref))
    s_minor = {j: float(env.reward_scale(j)) for j in MINORS}
    w_ref = np.asarray(env.bind(g_ref).context_weights())
    devs = jax.devices()
    d_min, d_maj = devs[0], devs[1 if len(devs) > 1 else 0]

    blocks, main_block = make_blocks(env, rung, ctx_ref, T, n, S, S_fixed, M)

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

    bufs, stats = make_buffers(nets)
    minor_static = tuple(stats[j] for j in MINORS)
    maj_static = stats[MAJOR]
    sim = make_sim(env, minor_static, maj_static)
    evaluate = make_evaluator(env, sim, ctx_ref, major_price_taker)
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

    # ── the major as a PRICE TAKER (job 2): trained on the bank's frozen paths ──
    def make_major_pt_stage(n_it, warm=False):
        opt0 = make_opt(n_it, warm)

        @jax.jit
        def stage(p, key_, bank):
            st = opt0.init(p)
            N = bank['A'].shape[0]

            def one(carry, it):
                p_, s_ = carry
                ks, ki = jax.random.split(jax.random.fold_in(key_, it))
                sel = jax.random.randint(ks, (B,), 0, N)
                H = bank['H'][sel] if USE_H else jnp.zeros((B, T, 3), jnp.float32)
                A, rho, P, gam = bank['A'][sel], bank['rho'][sel], bank['P'][sel], bank['gam'][sel]
                im = jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
                    jax.random.split(ki, B), env.agent_params[MAJOR, 7], T)

                def loss(pp):
                    net = eqx.combine(pp, maj_static)
                    R = jax.vmap(lambda a, r, h, pp_, i, g: env.bind(g).rollout_major(
                        net, a, r, h, pp_, i * env.bind(g).major_mass(), norm_gamma(g)))(A, rho, H, P, im, gam)
                    return -jnp.mean(R) / s0
                l, g = jax.value_and_grad(loss)(p_)
                upd, s_new = opt0.update(g, s_, p_)
                return (eqx.apply_updates(p_, upd), s_new), l
            (pf, _), L = jax.lax.scan(one, (p, st), jnp.arange(n_it))
            return pf, L
        return stage

    minor_fresh, minor_warm = make_minor_stage(iters), make_minor_stage(iters_warm, warm=True)
    if major_price_taker:
        major_fresh, major_warm = make_major_pt_stage(iters0), make_major_pt_stage(iters0_warm, warm=True)
    else:
        major_fresh, major_warm = make_major_stage(iters0), make_major_stage(iters0_warm, warm=True)

    force = os.environ.get('KETS_FORCE_RESTART') == '1'      # test only: exercise the restart path

    def failed(m):
        return force or m['gap'] < -m['gap_se']

    def diverged(L0):
        """Major training loss (= -reward / s0) ended clearly above where it started (the
        T=365 failures went from about -1 to +0.9; the margin keeps a warm start's noise out)."""
        h = float(_head(L0))
        return bool(float(_tail(L0)) > h + DIVERGE_MARGIN * abs(h))

    def split_minor(p_stack):
        return {j: eqx.combine(jax.tree_util.tree_map(lambda x: x[a], p_stack), minor_static[0])
                for a, j in enumerate(MINORS)}

    def stack_minor(nets_by_type):
        parts = [eqx.partition(nets_by_type[j], eqx.is_array)[0] for j in MINORS]
        return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *parts)

    names = ("private (minor)", "large public (MAJOR)", "fin. inst. (minor)")
    if verbose:
        print(f"major-minor FP rung {rung}: K={K} T={T} n={n}; minors {iters} steps fresh / "
              f"{iters_warm} warm (constant lr {lr_warm:g}) on a bank of {n_bank} markets (B={B}); major {iters0} fresh / "
              f"{iters0_warm} warm x B0={B0}; warm from iteration {warm_from if warm_from < 10**8 else "never (fresh BRs)"}, "
              f"checked on block '{main_block}', one fresh restart on failure; bisection {CLEAR_ITERS_V2}/side; "
              f"devices minor={d_min} major={d_maj}; floor_mode={FLOOR_MODE} c={SUPPLY_C:g} ceiling {A_CEIL_V2:g}",
              flush=True)
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
            print(f"\n{'=' * 72}\nMAJOR-MINOR rung {rung} — FP ITERATION {k + 1}/{K}  [mixture {kk}]  "
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
        t_major_done = [None]
        def _launch_major(out_):
            def _wait_major():
                jax.block_until_ready(out_)
                t_major_done[0] = time.time()
            th_ = threading.Thread(target=_wait_major); th_.start()
            return th_
        if NO_MAJOR:                    # LP benchmark: the major has no mass, no best response
            maj_out = (p0, jnp.zeros((1,), jnp.float32))
            t_major_done[0] = time.time()
            th = threading.Thread(target=lambda: None); th.start()
        elif not major_price_taker:     # strategic major: re-clears, does not need the bank
            maj_out = (major_warm if warm else major_fresh)(p0, jax.device_put(kb, d_maj), mb1, jb1, kk32_1)
            th = _launch_major(maj_out)
        # (a) minors on device d_min: price bank, then best responses on the bank
        bank = build_bank(jax.device_put(kbank, d_min), mb0, jb0, kk32_0)
        jax.block_until_ready(bank)
        t_bank = time.time() - tb
        if major_price_taker and not NO_MAJOR:     # price-taking major: on the bank, device d_maj
            bank_maj = jax.device_put(bank, d_maj)
            maj_out = (major_warm if warm else major_fresh)(p0, jax.device_put(kb, d_maj), bank_maj)
            th = _launch_major(maj_out)
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
            print(f"  BR phase {_hms(t_both)} (bank {_hms(t_bank)}, bank+minors {_hms(t_minor)}, major {_hms(t_major)}): loss t0 "
                  f"{Lm[0, 0]:+.4f}->{_tail(Lm)[0]:+.4f}  t2 {Lm[0, 1]:+.4f}->{_tail(Lm)[1]:+.4f}   "
                  f"major {L0[0]:+.4f}->{_tail(L0):+.4f}", flush=True)

        tailm = _tail(Lm)
        # No stuck rule without a major (the LP benchmark): there, no trade IS the best
        # response of a financial institution at a flat price (no allowances, no production,
        # value ~ 0), so a zero training loss is not a failure.
        stuck_on = not NO_MAJOR
        stuck_now = {MINORS[0]: stuck_on and abs(float(tailm[0])) < STUCK_TOL,
                     MINORS[1]: stuck_on and abs(float(tailm[1])) < STUCK_TOL,
                     MAJOR: stuck_on and abs(float(_tail(L0))) < STUCK_TOL}
        minor_buf = tuple(bufs[j] for j in MINORS)
        hist_minor = {j: [tr[j] for tr in nets] for j in MINORS}

        def check(new_, L0_):
            """Types whose best response fails on the main block (see the module docstring)."""
            per_type, _, _, _ = evaluate(blocks[main_block], kk, new_, minor_buf, bufs[MAJOR], hist_minor)
            bad = [j for j in MINORS if failed(per_type[j]) or stuck_now[j]]
            if not NO_MAJOR and (failed(per_type[MAJOR]) or stuck_now[MAJOR] or diverged(L0_)):
                bad.append(MAJOR)
            return sorted(bad)

        restarted, rejected = check(new, L0), []
        L0_kept = L0
        if restarted:
            tr = time.time()
            if MAJOR in restarted:
                nkr = jax.random.split(kr, N_TYPES)
                p0r = jax.device_put(eqx.partition(major_policy(nkr[MAJOR]), eqx.is_array)[0], d_maj)
                maj_r = (major_fresh(p0r, jax.device_put(jax.random.fold_in(kr, 1), d_maj), bank_maj)
                         if major_price_taker else
                         major_fresh(p0r, jax.device_put(jax.random.fold_in(kr, 1), d_maj), mb1, jb1, kk32_1))
            bad_minor = [j for j in restarted if j in MINORS]
            if bad_minor:
                nkr = jax.random.split(jax.random.fold_in(kr, 2), N_TYPES)
                mf = {j: minor_policy(nkr[j]) for j in MINORS}
                if mm_warm is not None:
                    mf[2] = mm_warm
                pr_, Lr = minor_fresh(jax.device_put(stack_minor(mf), d_min),
                                      jax.device_put(jax.random.fold_in(kr, 3), d_min), bank)
                smr = split_minor(jax.device_get(pr_))
                tr_ = _tail(np.asarray(Lr))
                for a, j in enumerate(MINORS):
                    if j in bad_minor:
                        new[j] = smr[j]
                        stuck_now[j] = stuck_on and abs(float(tr_[a])) < STUCK_TOL
            if MAJOR in restarted:
                new[MAJOR] = eqx.combine(jax.device_get(maj_r[0]), maj_static)
                L0_kept = np.asarray(maj_r[1])
                stuck_now[MAJOR] = abs(float(_tail(L0_kept))) < STUCK_TOL
            # still failing after the restart: reuse the previous best response
            if prev_new is not None:
                rejected = [j for j in check(new, L0_kept) if j in restarted]
                for j in rejected:
                    new[j] = prev_new[j]
                    stuck_now[j] = prev_stuck[j]
            if verbose:
                print(f"  RESTART (fresh seed, full budget) for types {restarted}"
                      f"{f'; still failing, previous BR reused for {rejected}' if rejected else ''}  "
                      f"[{_hms(time.time() - tr)}]", flush=True)
        del bank
        if major_price_taker and not NO_MAJOR:
            del bank_maj
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
               'rejected': rejected, 'stuck': [j for j in range(N_TYPES) if stuck_now[j]],
               'br_seconds': t_br, 'bank_seconds': t_bank, 'minor_seconds': t_minor, 'major_seconds': t_major,
               'iter_seconds': time.time() - ti,
               'major_loss_end': float(_tail(L0_kept))}
        if rung == 2:
            rec.update(rec_blocks['ref'])
        else:
            rec['blocks'] = rec_blocks
        records.append(rec)
        if save_dir:
            for j in range(N_TYPES):
                save_net(save_dir, k + 1, j, new[j])
            _atomic_json(ppath, {
                'rung': rung, 'version': 'major_minor_v2', 'context': list(ctx_ref),
                'fixed_scenarios': {nm: (b['ctx'] if b['ctx'] is not None else None) for nm, b in blocks.items()},
                'K': K, 'T_grid': [T], 'n_sim': n, 'iters': [iters], 'iters_major': iters0,
                'iters_warm': iters_warm, 'iters_major_warm': iters0_warm, 'warm_from': warm_from,
                'lr_warm': lr_warm, 'warm_schedule': 'constant',
                'n_bank': n_bank, 'clear_iters': CLEAR_ITERS_V2, 'devices': [str(d_min), str(d_maj)],
                'B': B, 'B0': B0, 'S': S, 'S_fixed': S_fixed, 'M': M, 'lr': lr, 'lr_min': lr_min,
                'floor_mode': FLOOR_MODE, 'supply_c': SUPPLY_C, 'A_CEIL': A_CEIL_V2,
                'N0': 1, 'major_type': MAJOR, 'minor_types': list(MINORS),
                'major_gradient': 'price_taker_frozen_paths' if major_price_taker else 'myopic_ift',
                'major_price_taker': bool(major_price_taker), 'no_major': NO_MAJOR, 'layout': LAYOUT, 'public_history_input': bool(USE_H), 'm0_ref': m0_ref, 's0': s0,
                'type_shares': [float(x) for x in w_ref], **(meta or {}), 'records': records})
        if verbose:
            print(f"  checkpoint saved  [iteration {_hms(time.time() - ti)}]", flush=True)
    return nets, records
