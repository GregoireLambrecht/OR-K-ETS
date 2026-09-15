"""Solver and infrastructure for the BE-MFG study.

The merge of what were utils / utils_rung2 / utils_rung3: frozen-noise
construction, the jitted best-response stages, the exploitability metric and the
fictitious-play loops. The `_r2` suffix marks the common-noise variant of a
routine and `_r3` the context-conditional one; `fictitious_play_rung3` is the
loop the study actually runs.

The context is four-dimensional — (init_alloc, kappa, Afloor, financial_inst) —
see CTX_NAMES. BM_uniform is deliberately not part of it.


The three rungs, and why they exist
===================================
The three `fictitious_play_rung*` loops are not alternative solvers. They are a
DIAGNOSTIC LADDER, built because fictitious play did not converge on the full
problem and the question was which ingredient was responsible. Each rung adds
exactly one source of difficulty to the one below, so a failure can be attributed
rather than guessed at:

  Rung 1   one scenario, one noise draw.  Everything is frozen except the
           policies, so the aggregate price path is a DETERMINISTIC function of
           the mixture. If FP fails here, nothing stochastic is to blame — the
           fault is the best response, the mixture update or the metric.

  Rung 2   one scenario, stochastic common noise.  The scenario stays at
           GAMMA_REF while the common noise becomes random: training draws fresh
           common noise per market, and the metric is measured on S FROZEN
           verification draws so successive iterations remain comparable. The
           mean field is now a random measure. A failure that appears here and
           not at Rung 1 is caused by the common noise alone.

  Rung 3   a family of scenarios (the BE layer).  Each market draws its own
           context c ~ q along with its noise, and the policy is conditioned on
           that context, so ONE Nash solve covers the whole regulatory family
           rather than a single cell of it. The quantity that must decrease is
           E_{c~q}[E_j(c)], estimated on the frozen prior block; the fixed
           blocks say whether that average is hiding a corner of the box where
           the solve has failed.

Rung 3 is the object of the study. Rungs 1 and 2 are kept because they are the
instrument for localising a regression: if Rung 3 stops converging after a
change, re-running Rung 2 and then Rung 1 says whether the cause is the context
layer, the common noise, or the inner solve.

Common to all three: the mixture is the empirical measure over past best
responses, exploitability E_j = V_br - V_mix is a LOWER bound whose tightness is
set by the best-response budget, and every loop is resumable from `save_dir`.
"""

import os
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import TypePolicyNN, stack_policies, apply_stacked
from envs.models import TypePolicyNN, stack_policies
from envs.models import ContextActionNN, BoundContextNN, stack_policies


# =============================================================================
# from utils.py
# =============================================================================

# ═════════════════════════════════════════════════════════════════════════════
# Frozen noise
# ═════════════════════════════════════════════════════════════════════════════

def make_frozen_noise(env, T, M, seed=NOISE_SEED):
    """All the randomness of a Rung 1 run, drawn once for a given horizon.

      P_traj    (T+1,)          frozen good-price path        [common noise 1]
      eps0      (T,)            frozen allowance shock        [common noise 2]
      idio_pop  tuple (n_j, T)  population draws, REUSED by every FP block so
                                the aggregate price is a deterministic function
                                of the policies and blocks share common random
                                numbers
      idio_eval tuple (M, T)    evaluation draws, used by BOTH terms of the
                                exploitability so the pairing is exact

    idio_eval is deliberately separate from the draws used to train the best
    response (resampled every gradient step, see train_best_response_stage):
    training and evaluating on the same fixed draws would bias the gap
    optimistically.
    """
    T_int = int(T)
    k_P, k_e0, k_pop, k_eval = jax.random.split(jax.random.PRNGKey(seed), 4)

    P_traj = env.generate_P(k_P, T_int)
    eps0   = env.generate_eps0(k_e0, T_int)

    pop_keys  = jax.random.split(k_pop,  N_TYPES)
    eval_keys = jax.random.split(k_eval, N_TYPES)

    def draws(k, n, sigma):
        ks = jax.random.split(k, n)
        return jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
            ks, sigma, T_int
        )

    idio_pop = tuple(
        draws(pop_keys[j], env.type_counts[j], env.agent_params[j, 7])
        for j in range(N_TYPES)
    )
    idio_eval = tuple(
        draws(eval_keys[j], M, env.agent_params[j, 7])
        for j in range(N_TYPES)
    )

    return {'P_traj': P_traj, 'eps0': eps0,
            'idio_pop': idio_pop, 'idio_eval': idio_eval}


# ═════════════════════════════════════════════════════════════════════════════
# Jitted closures  (env captured, never traced)
# ═════════════════════════════════════════════════════════════════════════════

def make_price_fn(env_T, statics):
    """Aggregate allowance-price path under the FP mixture, at env_T's horizon."""
    @eqx.filter_jit
    def fn(stacked, P_traj, eps0, idio_pop):
        return env_T.ensemble_price_path(stacked, statics, P_traj, eps0, idio_pop)
    return fn


def make_return_fn(env_T, j):
    """Total realised return of a batch of type-j agents against a frozen path."""
    @eqx.filter_jit
    def fn(policy, A_traj, P_traj, idio):
        _, R = env_T.rollout_agent_batch(policy, j, A_traj, P_traj, idio)
        return R
    return fn


def make_hist_return_fn(env_T, j, static):
    """Returns of the WHOLE type-j policy history in one vmapped call.

    V_j^mix needs (k+1) x M rollouts per type per FP iteration; a python loop
    over the history dispatches one policy at a time, which at K = 200 and
    M = 256 is ~154k separately-dispatched rollouts. Stacking the history and
    vmapping keeps it a single launch. Still linear in k, with a much smaller
    constant.

    stacked_params: type-j parameters with leading dim L -> returns (L, M).
    """
    @eqx.filter_jit
    def fn(stacked_params, A_traj, P_traj, idio):
        def one(p):
            _, R = env_T.rollout_agent_batch(
                eqx.combine(p, static), j, A_traj, P_traj, idio
            )
            return R
        return jax.vmap(one)(stacked_params)
    return fn


# ═════════════════════════════════════════════════════════════════════════════
# Market-maker warm start
#
# The market maker starts at theta_0 = 0, so its admissible set at t=0 is
# [0, cap_f] — it cannot sell. unnormalize_action enforces that with jnp.clip,
# which has exactly zero gradient outside its range. A fresh network that
# happens to want f < 0 across the reachable states therefore gets f = 0 at
# every step, theta never leaves 0, the constraint never unbinds, and
# d f / d hat_a0 == 0 forever. The initialisation alone decides it: measured
# collapse rate was 60-70% across every schedule and learning rate tried.
#
# Fix: don't leave it to chance. Regress the network onto a hand-designed
# policy that buys first, so theta is positive before gradient descent starts
# and the constraint is already slack.
#
#   buy at maximum for the first `buy_frac` of the horizon,
#   then sell the inventory down proportionally to the time remaining.
#
# In the network's own normalised action units (hat_f in [-1, 1], f =
# clip(hat_f * cap_f, ...)), with t_n = t/T and th_n = theta/(T*cap_f):
#
#   sell rate  f = -theta / (T - t)
#   => hat_f    = -theta / ((T - t) * cap_f) = -th_n / (1 - t_n)
#
# so the target is a function of two of the network's inputs and a small MLP
# represents it easily.
# ═════════════════════════════════════════════════════════════════════════════

def mm_heuristic_hat_f(t_n, th_n, buy_frac=0.2):
    """Target f, in normalised action units, for the hand-designed policy."""
    sell = -jnp.minimum(th_n / jnp.maximum(1.0 - t_n, 1e-3), 1.0)
    return jnp.where(t_n < buy_frac, jnp.ones_like(sell), sell)


def mm_heuristic_rollout(env, A_traj, P_traj, idio, buy_frac=0.2):
    """Run the market maker under the heuristic and return the (normalised
    state, target) pairs it visits — on-policy, so the regression is fitted
    where the policy will actually be evaluated."""
    T_int = int(env.T)
    B     = idio.shape[0]
    X0    = jnp.broadcast_to(env.type_initial_state(2), (B, 16))

    def step(X, t_idx):
        nrm = jax.vmap(env.normalize_state)(X)
        tgt = mm_heuristic_hat_f(nrm[:, 0], nrm[:, 1], buy_frac)
        half  = jnp.full_like(tgt, 0.5)
        a_hat = jnp.stack([tgt, half, half], axis=1)
        a     = jax.vmap(env.unnormalize_action)(X, a_hat)
        nX = jax.vmap(env.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
            X, a, idio[:, t_idx], A_traj[t_idx + 1], P_traj[t_idx + 1])
        return nX, (nrm, tgt)

    _, (S, Y) = jax.lax.scan(step, X0, jnp.arange(T_int))
    return S.reshape(-1, S.shape[-1]), Y.reshape(-1)


def pretrain_market_maker(env, A_traj, P_traj, idio, key,
                          steps=800, lr=1e-3, buy_frac=0.2):
    """Regress a fresh TypePolicyNN onto the heuristic. Only the f head is
    supervised — xi and eta carry no signal for a market maker (cap_xi = 1e-5)
    and are left to the best-response training that follows."""
    S, Y = mm_heuristic_rollout(env, A_traj, P_traj, idio, buy_frac)
    model = TypePolicyNN(key=key)
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr)
    st  = opt.init(params)

    def loss_fn(p):
        pred = jax.vmap(eqx.combine(p, static))(S)[:, 0]
        return jnp.mean((pred - Y) ** 2)

    @jax.jit
    def loop(p, s):
        def one(carry, _):
            p_, s_ = carry
            l, g      = jax.value_and_grad(loss_fn)(p_)
            u, s_new  = opt.update(g, s_, p_)
            return (eqx.apply_updates(p_, u), s_new), l
        (p_f, _), ls = jax.lax.scan(one, (p, s), jnp.arange(steps))
        return p_f, ls

    params, losses = loop(params, st)
    return eqx.combine(params, static), np.asarray(losses)


# ═════════════════════════════════════════════════════════════════════════════
# Best response
# ═════════════════════════════════════════════════════════════════════════════

def make_br_stage_fn(env_T, optimizer, nb_iterations, B, static):
    """One jitted best-response stage training ALL THREE types at once.

    The three best responses are independent — they all respond to the same
    frozen A_traj and never interact — so they can be batched. The networks are
    architecturally identical, so their parameters stack along a leading type
    axis and the whole rollout vmaps over it.

    Why batch rather than put one type per GPU: each BR is an 8-64-64-64-3 MLP
    at batch 256, so the per-step matmuls are 256x64 and the wall clock is
    dominated by kernel-launch overhead across the T-step scan, not arithmetic.
    Stacking makes them 768x64 for the same launch count — the same ~3x a
    three-GPU pmap would give, on one GPU.

    Summing the three per-type losses and running ONE optimizer on the stacked
    parameters is exactly equivalent to three separate Adams: loss_i does not
    depend on p_j for i != j, so slice j of the gradient of the sum is exactly
    d(loss_j)/d(p_j), and Adam is elementwise. (It would stop being equivalent
    under global-norm clipping, which we do not use.)

    A_traj / P_traj are arguments, not closure constants, so the compiled
    function is reused across FP iterations.
    """
    T_int = int(env_T.T)
    js    = jnp.arange(N_TYPES)

    def per_type_loss(p, j, key, A_traj, P_traj):
        # j is a tracer here; every per-type constant is a dynamic gather.
        ks   = jax.random.split(key, B)
        idio = jax.vmap(env_T.generate_idiosyncratic_noise,
                        in_axes=(0, None, None))(
            ks, env_T.agent_params[j, 7], T_int)
        _, R = env_T.rollout_agent_batch(
            eqx.combine(p, static), j, A_traj, P_traj, idio)
        return -jnp.mean(R) / env_T.reward_scale(j)   # exact reward only (D5)

    def total_loss(p_stack, step_key, A_traj, P_traj):
        keys = jax.random.split(step_key, N_TYPES)
        per  = jax.vmap(per_type_loss, in_axes=(0, 0, 0, None, None))(
            p_stack, js, keys, A_traj, P_traj)
        return jnp.sum(per), per

    @jax.jit
    def stage(p_stack, s, loop_key, A_traj, P_traj):
        A_traj = jax.lax.stop_gradient(A_traj)   # the BR never moves the price

        def one(carry, n):
            p_, s_ = carry
            step_key      = jax.random.fold_in(loop_key, n)
            (_, per), grads = jax.value_and_grad(total_loss, has_aux=True)(
                p_, step_key, A_traj, P_traj)
            upd, s_new    = optimizer.update(grads, s_, p_)
            return (eqx.apply_updates(p_, upd), s_new), per

        (p_f, s_f), losses = jax.lax.scan(one, (p_stack, s),
                                          jnp.arange(nb_iterations))
        return p_f, s_f, losses          # losses: (nb_iterations, N_TYPES)

    return stage


def train_best_response(env, A_trajs, noises, key, stage_fns, optimizer,
                        T_grid=RUNG1_T_GRID, lrs=RUNG1_T_LR, verbose=True,
                        init_override=None):
    """Full multigrid best responses for ALL THREE types.

    init_override: {type_index: model} replacing that type's fresh random init.
    Used to warm-start the market maker from the heuristic (see above), which
    is a better starting point rather than a change to the game — the argmax
    being sought is unchanged.

    Returns (triple, loss_history) where triple is (pi_1, pi_2, pi_3).
    """
    k_model, k_train = jax.random.split(key)
    mkeys  = jax.random.split(k_model, N_TYPES)
    models = [TypePolicyNN(key=mkeys[j]) for j in range(N_TYPES)]
    for j, m in (init_override or {}).items():
        models[j] = m

    parts   = [eqx.partition(m, eqx.is_array) for m in models]
    static  = parts[0][1]
    p_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                     *[p for p, _ in parts])
    opt_state = optimizer.init(p_stack)

    loss_hist = []
    for i, T in enumerate(T_grid):
        # With inject_hyperparams the lr is data and can change between stages
        # without a recompile. A schedule-based optimizer has no hyperparams
        # dict — its lr is a function of the state's own step counter.
        if hasattr(opt_state, 'hyperparams'):
            opt_state.hyperparams['learning_rate'] = jnp.array(lrs[i])
        p_stack, opt_state, losses = stage_fns[T](
            p_stack, opt_state, jax.random.fold_in(k_train, i),
            A_trajs[T], noises[T]['P_traj'],
        )
        losses = np.asarray(losses)
        loss_hist.append(losses)
        if verbose:
            per = "  ".join(f"t{j}: {losses[0, j]:+.5f}->{losses[-1, j]:+.5f}"
                            for j in range(N_TYPES))
            print(f"    T={T:<4d}  {per}", flush=True)

    triple = tuple(
        eqx.combine(jax.tree_util.tree_map(lambda x: x[j], p_stack), static)
        for j in range(N_TYPES)
    )
    return triple, loss_hist


# ═════════════════════════════════════════════════════════════════════════════
# Exploitability  (plan.md §5)
# ═════════════════════════════════════════════════════════════════════════════

def exploitability(env_T, j, new_policy, history, A_traj, noise,
                   return_fn=None, hist_fn=None, static=None):
    """E_j = V_j^BR - V_j^mix.

    Both terms are single-agent rollouts of a type-j agent against the same
    frozen (A_traj, P*) with the same frozen idiosyncratic draws, so the CRN
    pairing is exact and the difference is purely strategic.

    V_j^mix is the value of the type-j *mixture*: the uniform average over all
    past best responses of that type — what the n_j agents per past BR earn
    inside the population.
    """
    if return_fn is None:
        return_fn = make_return_fn(env_T, j)

    idio   = noise['idio_eval'][j]
    P_traj = noise['P_traj']
    M      = idio.shape[0]

    R_br = return_fn(new_policy, A_traj, P_traj, idio)                    # (M,)

    if hist_fn is not None:
        stacked_hist, _ = stack_policies([h[j] for h in history])
        R_hist = hist_fn(stacked_hist, A_traj, P_traj, idio)              # (L, M)
    else:
        R_hist = jnp.stack([return_fn(h[j], A_traj, P_traj, idio)
                            for h in history])
    R_mix = jnp.mean(R_hist, axis=0)                                      # (M,)

    d     = R_br - R_mix          # paired per-draw difference
    gap   = float(jnp.mean(d))
    se    = float(jnp.std(d, ddof=1) / jnp.sqrt(M)) if M > 1 else 0.0
    scale = float(env_T.reward_scale(j))

    return {
        'gap':          gap,
        'gap_se':       se,
        'gap_relative': gap / scale,
        'V_br':         float(jnp.mean(R_br)),
        'V_mix':        float(jnp.mean(R_mix)),
        'scale':        scale,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Fictitious play
# ═════════════════════════════════════════════════════════════════════════════

def _hms(seconds):
    s = int(round(seconds))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def _atomic_json(path, obj):
    """Write via a temp file + rename, so a kill mid-write cannot leave a
    truncated progress.json that would break the resume."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fictitious_play_rung1(env, gamma_ref=GAMMA_REF, K=RUNG1_K,
                          T_grid=RUNG1_T_GRID, iters=RUNG1_T_ITERS,
                          lrs=RUNG1_T_LR, B=RUNG1_B, M=RUNG1_M,
                          window=RUNG1_WINDOW, key=None, save_dir=None,
                          plot_dir=None, verbose=True, meta=None,
                          mm_warmstart=None, lr_schedule=None, lr_min=None):
    """Rung 1 fictitious play — the deterministic base of the ladder.

    What is frozen
    --------------
    Everything except the policies. ONE scenario (`gamma_ref`), ONE common-noise
    realisation and ONE set of population idiosyncratic draws, all fixed for the
    whole run. The aggregate price path is therefore a deterministic function of
    the mixture, and two iterations differ only because the policies differ.

    What it is for
    --------------
    To answer "does fictitious play converge at all here?" with no stochastic
    confound. Because the noise cannot move, any failure to converge is
    attributable to the inner solve (an under-trained best response), the
    mixture update, or the metric — never to sampling. It is the control against
    which Rungs 2 and 3 are read: a pathology present at Rung 1 is not caused by
    common noise or by the context layer.

    The metric is E_j = V_br - V_mix on that single frozen draw, so it carries no
    Monte Carlo error at all — the only error is the best-response budget.

    Resumable: if save_dir already holds a progress.json, the run picks up at
    the next iteration with the saved mixture restored. Iteration keys are
    derived as fold_in(base_key, k) rather than by sequentially splitting, so a
    resumed run uses exactly the keys the uninterrupted run would have — no
    fragile RNG fast-forwarding.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Two key namespaces: one for the initial mixture, one folded per
    # iteration. fold_in requires a uint32 index, so a sentinel like -1 is not
    # available for the init; splitting first keeps the two collision-free and
    # keeps iteration k's key independent of how the run was segmented.
    k_init_base, k_iter_base = jax.random.split(key)

    env     = env.set_context(jnp.asarray(gamma_ref, dtype=jnp.float32))
    T_final = T_grid[-1]
    envs_T  = {T: env.set_T(T) for T in T_grid}
    env_f   = envs_T[T_final]

    # Frozen noise, one set per horizon (paths have different lengths).
    noises = {T: make_frozen_noise(envs_T[T], T, M) for T in T_grid}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    progress_path = os.path.join(save_dir, 'progress.json') if save_dir else None

    # ── resume, or start fresh ───────────────────────────────────────────────
    records, start_k = [], 0
    resumed = False
    if progress_path and os.path.exists(progress_path):
        with open(progress_path) as f:
            prev = json.load(f)
        records = prev.get('records', [])
        start_k = len(records)
        n_avail = 0
        while os.path.exists(os.path.join(save_dir, f'policy_{n_avail}_type0.eqx')):
            n_avail += 1
        # after finishing iteration k (0-indexed) there are k+2 triples on disk
        start_k = min(start_k, max(0, n_avail - 1))
        records = records[:start_k]
        lo = max(0, start_k + 1 - window)
        history = [load_triple(save_dir, i) for i in range(lo, start_k + 1)]
        resumed = True
        if verbose:
            print(f"=== RESUMING: {start_k} iteration(s) done, "
                  f"{n_avail} triple(s) on disk, mixture restored to "
                  f"{len(history)} triple(s) ===", flush=True)

    if not resumed:
        init_keys = jax.random.split(k_init_base, N_TYPES)
        history = [tuple(TypePolicyNN(key=init_keys[j]) for j in range(N_TYPES))]
        if save_dir:
            save_triple(save_dir, 0, history[0])

    if start_k >= K:
        if verbose:
            print(f"Nothing to do: {start_k} >= K = {K}", flush=True)
        return history, records

    # statics are identical for every TypePolicyNN, so build the jitted
    # closures once and reuse them across FP iterations.
    _, statics = zip(*[stack_policies([h[j] for h in history])
                       for j in range(N_TYPES)])
    price_fns  = {T: make_price_fn(envs_T[T], statics) for T in T_grid}
    return_fns = [make_return_fn(env_f, j) for j in range(N_TYPES)]

    # static is the non-array part of a TypePolicyNN — shared by all of them.
    _p0, model_static = eqx.partition(history[0][0], eqx.is_array)
    hist_fns  = [make_hist_return_fn(env_f, j, model_static)
                 for j in range(N_TYPES)]

    # One compiled best-response stage per horizon, training all three types
    # together, reused every FP iteration.
    #
    # lr_schedule='cosine' anneals within EACH best response, from lrs[0] down
    # to lr_min over that stage's iterations. The schedule's step counter lives
    # in the optimizer state, which train_best_response re-initialises for every
    # FP iteration, so each fresh network gets the full anneal rather than the
    # decay running out after iteration 1.
    if lr_schedule == 'cosine':
        if len(T_grid) != 1:
            raise ValueError("lr_schedule='cosine' expects a single horizon; "
                             f"got T_grid={T_grid}")
        lo = lr_min if lr_min is not None else lrs[0] / 10.0
        optimizer = optax.adam(optax.cosine_decay_schedule(
            init_value=lrs[0], decay_steps=iters[0], alpha=lo / lrs[0]))
        if verbose:
            print(f"lr: cosine anneal {lrs[0]:g} -> {lo:g} over {iters[0]} steps",
                  flush=True)
    else:
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lrs[0])
    stage_fns = {
        T: make_br_stage_fn(envs_T[T], optimizer, iters[i], B, model_static)
        for i, T in enumerate(T_grid)
    }

    # Load the market-maker warm start ONCE. Its targets are functions of
    # normalised inputs only, so a single fitted network serves every horizon.
    mm_warm = None
    if mm_warmstart:
        mm_warm = eqx.tree_deserialise_leaves(
            mm_warmstart, TypePolicyNN(key=jax.random.PRNGKey(0)))
        if verbose:
            print(f"market-maker warm start loaded from {mm_warmstart}", flush=True)
    elif verbose:
        print("market-maker warm start: OFF (fresh random init)", flush=True)

    shares  = env.type_shares
    floor   = float(env.Afloor)
    t_start = time.time()

    for k in range(start_k, K):
        t_iter = time.time()
        if verbose:
            done = k - start_k
            eta = ("" if done == 0 else
                   f"   ETA {_hms((time.time() - t_start) / done * (K - k))}")
            print(f"\n{'='*72}\nFP ITERATION {k+1}/{K}   "
                  f"[mixture {len(history)} triple(s)]   "
                  f"elapsed {_hms(time.time() - t_start)}{eta}\n{'='*72}",
                  flush=True)

        # ── aggregate price path under the current mixture, per horizon ──────
        stacked = tuple(stack_policies([h[j] for h in history])[0]
                        for j in range(N_TYPES))
        A_trajs = {
            T: price_fns[T](stacked, noises[T]['P_traj'],
                            noises[T]['eps0'], noises[T]['idio_pop'])
            for T in T_grid
        }
        A_final = A_trajs[T_final]

        if verbose:
            a = np.asarray(A_final)
            print(f"  price A_t : min {a.min():9.4f}  max {a.max():9.4f}  "
                  f"mean {a.mean():9.4f}  final {a[-1]:9.4f}  "
                  f"on-floor {float(np.mean(a <= floor + 1e-6)):5.1%}", flush=True)
            print(f"  best responses (exact reward, loss = -return/scale):",
                  flush=True)

        # ── three best responses, trained together ──────────────────────────
        # The market maker starts from the saved warm-start network (loaded
        # once, before the loop) rather than a fresh random init, which
        # collapsed into the theta_0 = 0 clip trap 60-70% of the time.
        t_br = time.time()
        new_triple, loss_hist = train_best_response(
            env, A_trajs, noises, jax.random.fold_in(k_iter_base, k),
            stage_fns, optimizer, T_grid=T_grid, lrs=lrs, verbose=verbose,
            init_override=({2: mm_warm} if mm_warm is not None else None),
        )
        br_secs = time.time() - t_br
        if verbose:
            print(f"    trained in {_hms(br_secs)}", flush=True)

        # ── per-type exploitability at the final horizon ─────────────────────
        per_type = [
            exploitability(env_f, j, new_triple[j], history, A_final,
                           noises[T_final], return_fn=return_fns[j],
                           hist_fn=hist_fns[j])
            for j in range(N_TYPES)
        ]
        weighted = sum(shares[j] * per_type[j]['gap'] for j in range(N_TYPES))
        # Population-weighted relative gap: each type's exploitability as a
        # fraction of its own reward scale, then averaged by population share.
        # Needed because type 2 has 7x type 1's capacity, so the absolute
        # weighted number is dominated by it.
        weighted_rel = sum(shares[j] * per_type[j]['gap_relative']
                           for j in range(N_TYPES))

        if verbose:
            print("  exploitability  E_j = V(new BR) - V(mixture), "
                  "same price path, paired noise:", flush=True)
            names = ("private     ", "large public", "market maker")
            for j, m in enumerate(per_type):
                flag = "  <-- NEGATIVE: BR under-trained" if m['gap'] < -m['gap_se'] else ""
                print(f"    {names[j]}  E = {m['gap']:+13.1f} +/- {m['gap_se']:8.1f}"
                      f"  ({m['gap_relative']:+8.4%} of scale)"
                      f"   V_br {m['V_br']:+11.5g}  V_mix {m['V_mix']:+11.5g}{flag}",
                      flush=True)
            print(f"    weighted {weighted:+.2f}  "
                  f"({weighted_rel:+.4%} population-weighted)", flush=True)

        # ── mixture path movement (deterministic: same frozen noise) ─────────
        windowed = (history + [new_triple])[-window:]
        stacked_next = tuple(stack_policies([h[j] for h in windowed])[0]
                             for j in range(N_TYPES))
        A_next = price_fns[T_final](
            stacked_next, noises[T_final]['P_traj'],
            noises[T_final]['eps0'], noises[T_final]['idio_pop']
        )
        path_l2 = float(jnp.linalg.norm(A_next - A_final))
        if verbose:
            print(f"  mixture movement  ||A_next - A_now||_2 = {path_l2:.6f}",
                  flush=True)

        rec = {
            'iteration':         k + 1,
            'mixture_size':      len(history),
            'per_type':          per_type,
            'weighted':          weighted,
            'weighted_relative': weighted_rel,
            'path_l2':           path_l2,
            'A_min':             float(A_final.min()),
            'A_max':             float(A_final.max()),
            'A_mean':            float(A_final.mean()),
            'A_final':           float(A_final[-1]),
            'A_on_floor':        float(np.mean(np.asarray(A_final) <= floor + 1e-6)),
            'br_seconds':        br_secs,
            'br_loss_first':     [[float(L[0, j]) for j in range(N_TYPES)]
                                  for L in loss_hist],
            'br_loss_last':      [[float(L[-1, j]) for j in range(N_TYPES)]
                                  for L in loss_hist],
        }

        # ── diagnostic sheet: mixture vs BR, states and actions, per type ────
        if plot_dir:
            from diagnostics import collect_iteration_diagnostics
            from helper_plot import plot_iteration, plot_convergence
            diag = collect_iteration_diagnostics(
                env_f, history, new_triple, noises[T_final], statics
            )
            if verbose:
                names = ("private     ", "large public", "market maker")
                print("  mixture behaviour (mean over horizon, "
                      "[10-90%] spread across agents):", flush=True)
                for j in range(N_TYPES):
                    f_m  = diag['mix']['action'][j][:, 0]
                    band = (diag['mix']['action_hi'][j][:, 0]
                            - diag['mix']['action_lo'][j][:, 0]).mean()
                    xi_m = diag['mix']['action'][j][:, 1].mean()
                    em_m = diag['mix']['emission'][j].mean()
                    st   = diag['mix']['state'][j]
                    comp = (st[-1, 8] * st[-1, 2] + st[-1, 9] * st[-1, 3]
                            + st[-1, 4] - st[-1, 1])
                    print(f"    {names[j]}  f {f_m.mean():+10.2f} "
                          f"(band {band:9.2f})   xi {xi_m:9.2f}   "
                          f"e {em_m:9.2f}   emissions-allowances_T {comp:+12.1f}",
                          flush=True)
            p = plot_iteration(
                diag, os.path.join(plot_dir, f'iter_{k+1:03d}.pdf'),
                title_suffix=(f"  —  iteration {k+1}, gamma = "
                              f"(alloc {gamma_ref[0]}, kappa {gamma_ref[1]}), "
                              f"T = {T_final}")
            )
            plot_convergence(
                records + [rec], os.path.join(plot_dir, 'convergence.pdf'),
                title_suffix=(f"  —  gamma = (alloc {gamma_ref[0]}, "
                              f"kappa {gamma_ref[1]})")
            )
            if verbose:
                print(f"  plots -> {p}", flush=True)

        rec['iter_seconds'] = time.time() - t_iter
        records.append(rec)
        history = windowed

        # ── checkpoint: policies first, then progress.json atomically, so a
        #    kill between the two can only ever leave MORE policies than
        #    progress claims — which the resume logic tolerates. ─────────────
        if save_dir:
            save_triple(save_dir, k + 1, new_triple)
            _atomic_json(progress_path, {
                'rung':        1,
                'gamma_ref':   [float(x) for x in gamma_ref],
                'T_grid':      list(T_grid),
                'iters':       list(iters),
                'lrs':         list(lrs),
                'B': B, 'M': M, 'window': window, 'K': K,
                'agent_per_policy': env.agent_per_policy,
                'type_counts': list(env.type_counts),
                'type_shares': list(shares),
                'noise_seed':  NOISE_SEED,
                'mm_warmstart': mm_warmstart,
                'lr_schedule': lr_schedule, 'lr_min': lr_min,
                **(meta or {}),
                'records':     records,
            })
            if verbose:
                print(f"  checkpoint -> {save_dir}/policy_{k+1}_type*.eqx  "
                      f"+ progress.json   [iteration took "
                      f"{_hms(rec['iter_seconds'])}]", flush=True)

    if verbose:
        print(f"\nTotal wall time: {_hms(time.time() - t_start)}", flush=True)
    return history, records


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint IO
# ═════════════════════════════════════════════════════════════════════════════

def save_triple(save_dir, k, triple):
    for j, model in enumerate(triple):
        eqx.tree_serialise_leaves(
            os.path.join(save_dir, f'policy_{k}_type{j}.eqx'), model
        )


def load_triple(save_dir, k):
    like = TypePolicyNN(key=jax.random.PRNGKey(0))
    return tuple(
        eqx.tree_deserialise_leaves(
            os.path.join(save_dir, f'policy_{k}_type{j}.eqx'), like
        )
        for j in range(N_TYPES)
    )


# =============================================================================
# from utils_rung2.py
# =============================================================================

VERIFY_SEED = 90210          # frozen evaluation set; never reused for training


# ═════════════════════════════════════════════════════════════════════════════
# Noise
# ═════════════════════════════════════════════════════════════════════════════

def draw_common_noise(env, T, S, key):
    """S independent common-noise realisations.

    Returns P (S, T+1) and eps0 (S, T). These are the two common channels of
    K-ETS.tex: the exogenous good-price path and the allowance-price shock.
    """
    T_int = int(T)
    kP, ke = jax.random.split(key)
    P    = jax.vmap(env.generate_P,     in_axes=(0, None))(jax.random.split(kP, S), T_int)
    eps0 = jax.vmap(env.generate_eps0,  in_axes=(0, None))(jax.random.split(ke, S), T_int)
    return P, eps0


def draw_idio(env, T, n_per_type, key):
    """Idiosyncratic draws, one array per type of shape (n_j, T)."""
    T_int = int(T)
    ks = jax.random.split(key, N_TYPES)
    out = []
    for j in range(N_TYPES):
        n = n_per_type[j] if isinstance(n_per_type, (list, tuple)) else n_per_type
        out.append(jax.vmap(env.generate_idiosyncratic_noise,
                            in_axes=(0, None, None))(
            jax.random.split(ks[j], n), env.agent_params[j, 7], T_int))
    return tuple(out)


def make_verification_set(env, T, S, M, seed=VERIFY_SEED):
    """The frozen evaluation set: S common-noise draws plus the idiosyncratic
    draws used by the ensemble and by the paired exploitability rollouts.

    idio_pop is shared across the S samples on purpose — the aggregate price
    path is then a deterministic function of (policies, common noise), so
    differencing two FP iterations on the same sample isolates policy movement.
    """
    kc, kp, ke = jax.random.split(jax.random.PRNGKey(seed), 3)
    P, eps0 = draw_common_noise(env, T, S, kc)
    return {
        'P': P, 'eps0': eps0, 'S': S, 'M': M,
        'idio_pop':  draw_idio(env, T, env.type_counts, kp),
        'idio_eval': draw_idio(env, T, (M, M, M), ke),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Jitted closures  (env captured, never traced — env.T must stay concrete)
# ═════════════════════════════════════════════════════════════════════════════

def make_price_fn_batched(env_T, statics):
    """Aggregate price path for a BATCH of common-noise draws.

    vmapped over the sample axis so the GPU sees one wide batch rather than S
    sequential rollouts; Rung 1's throughput was launch-bound, so this is where
    the factor of S is recovered. Memory is the trade — call it in chunks.
    """
    @eqx.filter_jit
    def fn(stacked, P, eps0, idio_pop):
        return jax.vmap(
            lambda p, e: env_T.ensemble_price_path(stacked, statics, p, e, idio_pop)
        )(P, eps0)
    return fn


def make_batched_return_fn(env_T, j):
    """Returns of one policy across (sample, draw): A (S,T+1), P (S,T+1),
    idio (M,T) -> (S, M)."""
    @eqx.filter_jit
    def fn(policy, A, P, idio):
        def one(a, p):
            _, R = env_T.rollout_agent_batch(policy, j, a, p, idio)
            return R
        return jax.vmap(one)(A, P)
    return fn


def make_batched_hist_fn(env_T, j, static):
    """Same, for the whole stacked policy history -> (L, S, M)."""
    @eqx.filter_jit
    def fn(stacked_params, A, P, idio):
        def per_policy(prm):
            model = eqx.combine(prm, static)
            def one(a, p):
                _, R = env_T.rollout_agent_batch(model, j, a, p, idio)
                return R
            return jax.vmap(one)(A, P)
        return jax.vmap(per_policy)(stacked_params)
    return fn


def chunked(fn, n, chunk):
    """Apply fn to slices [i:i+chunk] of the leading axis and concatenate."""
    outs = [fn(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    return jnp.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]


# ═════════════════════════════════════════════════════════════════════════════
# Best response  —  trained against a POOL of priced common-noise paths
# ═════════════════════════════════════════════════════════════════════════════

def make_br_stage_fn_r2(env_T, statics, optimizer, nb_iterations, B, static):
    """One jitted best-response stage, all three types at once.

    Every gradient step draws FRESH noise:
      * B independent common-noise realisations (good-price path + allowance
        shock), each giving its own aggregate price path under the mixture;
      * fresh idiosyncratic draws for the population inside each of those
        ensemble rollouts;
      * fresh idiosyncratic draws for the responding agent itself.

    So the BR is optimised against the distribution of common noise rather than
    any fixed sample. Cost is dominated by the T sequential scan steps, not by
    B — the B draws are vmapped into a single wide batch.

    The price path is computed outside value_and_grad and stop_gradient'd: the
    representative agent is infinitesimal and does not move the price, and
    keeping it out of the differentiated function means it is evaluated once per
    step instead of once per forward and once per backward.

    `stacked_mix` is an argument because the mixture grows every FP iteration.
    """
    T_int  = int(env_T.T)
    js     = jnp.arange(N_TYPES)
    counts = env_T.type_counts

    def sample_noise(key):
        """B common-noise draws + the population idiosyncratic draws they need.
        Returns P (B,T+1), eps0 (B,T), and per type (B, n_j, T)."""
        kP, kE, kI = jax.random.split(key, 3)
        P    = jax.vmap(env_T.generate_P,    in_axes=(0, None))(
            jax.random.split(kP, B), T_int)
        eps0 = jax.vmap(env_T.generate_eps0, in_axes=(0, None))(
            jax.random.split(kE, B), T_int)
        ki = jax.random.split(kI, N_TYPES)
        idio_pop = tuple(
            jax.vmap(lambda k, j=j: jax.vmap(
                env_T.generate_idiosyncratic_noise, in_axes=(0, None, None))(
                    jax.random.split(k, counts[j]),
                    env_T.agent_params[j, 7], T_int)
            )(jax.random.split(ki[j], B))
            for j in range(N_TYPES))
        return P, eps0, idio_pop

    def prices(stacked_mix, P, eps0, idio_pop):
        """B aggregate price paths, one per common-noise draw. (B, T+1)."""
        return jax.vmap(
            lambda p, e, i0, i1, i2: env_T.ensemble_price_path(
                stacked_mix, statics, p, e, (i0, i1, i2))
        )(P, eps0, *idio_pop)

    def loss_given_noise(p_stack, key, A, P):
        """A and P are constants here — the gradient only flows through the
        responding agent's own trajectory."""
        def per_type(p, j, kk):
            model = eqx.combine(p, static)
            idio  = jax.vmap(env_T.generate_idiosyncratic_noise,
                             in_axes=(0, None, None))(
                jax.random.split(kk, B), env_T.agent_params[j, 7], T_int)

            def one(a, pp, ii):                    # one agent per noise draw
                _, R = env_T.rollout_agent_batch(model, j, a, pp, ii[None, :])
                return R[0]

            R = jax.vmap(one)(A, P, idio)
            return -jnp.mean(R) / env_T.reward_scale(j)   # exact reward only

        per = jax.vmap(per_type, in_axes=(0, 0, 0))(
            p_stack, js, jax.random.split(key, N_TYPES))
        return jnp.sum(per), per

    @jax.jit
    def stage(p_stack, s, loop_key, stacked_mix):
        def one(carry, n):
            p_, s_ = carry
            k_noise, k_grad = jax.random.split(jax.random.fold_in(loop_key, n))
            P, eps0, idio_pop = sample_noise(k_noise)
            A = jax.lax.stop_gradient(prices(stacked_mix, P, eps0, idio_pop))
            (_, per), g = jax.value_and_grad(loss_given_noise, has_aux=True)(
                p_, k_grad, A, P)
            upd, s_new = optimizer.update(g, s_, p_)
            return (eqx.apply_updates(p_, upd), s_new), per

        (p_f, s_f), losses = jax.lax.scan(one, (p_stack, s),
                                          jnp.arange(nb_iterations))
        return p_f, s_f, losses

    return stage


def train_best_response_r2(stacked_mix, key, stage_fns, optimizer, mm_warm,
                           T_grid, iters, lrs, verbose=True):
    """Multigrid best responses for all three types, from fresh networks
    (market maker warm-started, see utils.pretrain / pretrain_mm.py)."""
    k_model, k_train = jax.random.split(key)
    mkeys  = jax.random.split(k_model, N_TYPES)
    models = [TypePolicyNN(key=mkeys[j]) for j in range(N_TYPES)]
    if mm_warm is not None:
        models[2] = mm_warm

    parts   = [eqx.partition(m, eqx.is_array) for m in models]
    static  = parts[0][1]
    p_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                     *[p for p, _ in parts])
    opt_state = optimizer.init(p_stack)

    loss_hist = []
    for i, T in enumerate(T_grid):
        if hasattr(opt_state, 'hyperparams'):
            opt_state.hyperparams['learning_rate'] = jnp.array(lrs[i])
        p_stack, opt_state, losses = stage_fns[T](
            p_stack, opt_state, jax.random.fold_in(k_train, i), stacked_mix)
        losses = np.asarray(losses)
        loss_hist.append(losses)
        if verbose:
            per = "  ".join(f"t{j}: {losses[0, j]:+.5f}->{losses[-1, j]:+.5f}"
                            for j in range(N_TYPES))
            print(f"    T={T:<5d} {per}", flush=True)

    triple = tuple(
        eqx.combine(jax.tree_util.tree_map(lambda x: x[j], p_stack), static)
        for j in range(N_TYPES))
    return triple, loss_hist


# ═════════════════════════════════════════════════════════════════════════════
# Exploitability under common noise
# ═════════════════════════════════════════════════════════════════════════════

def exploitability_r2(env_T, j, new_policy, history, A_ver, P_ver, idio,
                      ret_fn, hist_fn, static):
    """E_j averaged over the frozen common-noise set.

    Both terms use the SAME price path per sample and the SAME idiosyncratic
    draws, so the difference is paired at both levels. Under common noise this
    matters far more than it did in Rung 1 — the common draw is now the dominant
    variance source, and unpaired estimates would swamp a gap of a few percent.
    """
    R_br   = ret_fn(new_policy, A_ver, P_ver, idio)                    # (S, M)
    stacked, _ = stack_policies([h[j] for h in history])
    R_hist = hist_fn(stacked, A_ver, P_ver, idio)                      # (L,S,M)
    R_mix  = jnp.mean(R_hist, axis=0)                                  # (S, M)

    d_sample = jnp.mean(R_br - R_mix, axis=1)     # (S,) paired, per common draw
    gap   = float(jnp.mean(d_sample))
    se    = float(jnp.std(d_sample, ddof=1) / jnp.sqrt(d_sample.shape[0]))
    scale = float(env_T.reward_scale(j))
    return {
        'gap': gap, 'gap_se': se, 'gap_relative': gap / scale,
        'V_br':  float(jnp.mean(R_br)),
        'V_mix': float(jnp.mean(R_mix)),
        'scale': scale,
        'gap_p10': float(jnp.quantile(d_sample, 0.1)),
        'gap_p90': float(jnp.quantile(d_sample, 0.9)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Fictitious play, Rung 2
# ═════════════════════════════════════════════════════════════════════════════

def fictitious_play_rung2(env, gamma_ref=GAMMA_REF, K=50,
                          T_grid=(50,), iters=(6000,), lrs=(1e-3,),
                          B=256, S=500, M=32, eval_chunk=50,
                          window=200, key=None, save_dir=None, plot_dir=None,
                          mm_warmstart=None, lr_schedule=None, lr_min=None,
                          verbose=True, meta=None):
    """Rung 2 fictitious play — one scenario, stochastic common noise.

    What changes from Rung 1
    ------------------------
    Exactly one thing: the common noise stops being a single frozen path. Every
    gradient step draws fresh common noise (the exogenous good-price path and the
    allowance shock) for each of the B markets, so the mean field becomes a
    RANDOM measure rather than a deterministic one. The scenario is still frozen
    at `gamma_ref`, and the policies are still unconditional.

    What it is for
    --------------
    To isolate the effect of common noise on convergence. A pathology that shows
    up here but not at Rung 1 is caused by the common noise and nothing else —
    the scenario has not moved and the policy class has not changed.

    Why the verification set is frozen
    ----------------------------------
    The metric is measured on S FIXED common-noise draws, reused at every FP
    iteration, rather than on fresh ones. With fresh draws the iteration-to-
    iteration change in E would mix a real change in the mixture with a change of
    sample, and the convergence curve would be unreadable. Freezing them makes
    successive iterations paired (common random numbers), so the curve reflects
    the policies alone. The training draws stay fresh — only the measurement is
    frozen, and its seed is never one used for training.

    Resumable, like Rung 1.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    k_init_base, k_iter_base = jax.random.split(key)

    env     = env.set_context(jnp.asarray(gamma_ref, dtype=jnp.float32))
    T_final = T_grid[-1]
    envs_T  = {T: env.set_T(T) for T in T_grid}
    env_f   = envs_T[T_final]

    # Frozen evaluation set — drawn once, identical at every FP iteration.
    ver = make_verification_set(env_f, T_final, S, M)
    if verbose:
        print(f"verification set: S={S} frozen common-noise draws, M={M} "
              f"paired agents/type, chunk={eval_chunk}", flush=True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    progress_path = os.path.join(save_dir, 'progress.json') if save_dir else None

    # ── resume ──────────────────────────────────────────────────────────────
    records, start_k, resumed = [], 0, False
    if progress_path and os.path.exists(progress_path):
        with open(progress_path) as f:
            prev = json.load(f)
        records = prev.get('records', [])
        n_avail = 0
        while os.path.exists(os.path.join(save_dir, f'policy_{n_avail}_type0.eqx')):
            n_avail += 1
        start_k = min(len(records), max(0, n_avail - 1))
        records = records[:start_k]
        lo = max(0, start_k + 1 - window)
        history = [load_triple(save_dir, i) for i in range(lo, start_k + 1)]
        resumed = True
        if verbose:
            print(f"=== RESUMING at iteration {start_k+1} "
                  f"({len(history)} triples restored) ===", flush=True)

    if not resumed:
        ik = jax.random.split(k_init_base, N_TYPES)
        history = [tuple(TypePolicyNN(key=ik[j]) for j in range(N_TYPES))]
        if save_dir:
            save_triple(save_dir, 0, history[0])
    if start_k >= K:
        return history, records

    mm_warm = None
    if mm_warmstart:
        mm_warm = eqx.tree_deserialise_leaves(
            mm_warmstart, TypePolicyNN(key=jax.random.PRNGKey(0)))
        if verbose:
            print(f"market-maker warm start: {mm_warmstart}", flush=True)

    _, model_static = eqx.partition(history[0][0], eqx.is_array)
    statics    = tuple(stack_policies([history[0][j]])[1] for j in range(N_TYPES))
    price_fns  = {T: make_price_fn_batched(envs_T[T], statics) for T in T_grid}
    ret_fns    = [make_batched_return_fn(env_f, j) for j in range(N_TYPES)]
    hist_fns   = [make_batched_hist_fn(env_f, j, model_static) for j in range(N_TYPES)]

    if lr_schedule == 'cosine':
        if len(T_grid) != 1:
            raise ValueError("cosine lr expects a single horizon")
        lo_lr = lr_min if lr_min is not None else lrs[0] / 10.0
        optimizer = optax.adam(optax.cosine_decay_schedule(
            init_value=lrs[0], decay_steps=iters[0], alpha=lo_lr / lrs[0]))
        if verbose:
            print(f"lr: cosine {lrs[0]:g} -> {lo_lr:g} over {iters[0]} steps",
                  flush=True)
    else:
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lrs[0])
    stage_fns = {T: make_br_stage_fn_r2(envs_T[T], statics, optimizer,
                                        iters[i], B, model_static)
                 for i, T in enumerate(T_grid)}

    # One fixed common-noise draw for every diagnostic sheet, so the
    # iteration-to-iteration comparison is not reading a different market each
    # time. Separate from the S-sample verification set, which is for the
    # expectation; this one is for behaviour.
    plot_noise = None
    if plot_dir:
        from helper_plot import make_plot_noise
        plot_noise = make_plot_noise(env_f, T_final)
        os.makedirs(plot_dir, exist_ok=True)
        if verbose:
            print("plot sheets use one fixed common-noise draw "
                  "(helper_plot.PLOT_SEED)", flush=True)

    shares, floor = env.type_shares, float(env.Afloor)
    t_start, A_prev = time.time(), None

    for k in range(start_k, K):
        t_iter = time.time()
        if verbose:
            done = k - start_k
            eta = ("" if done == 0 else
                   f"   ETA {_hms((time.time()-t_start)/done*(K-k))}")
            print(f"\n{'='*72}\nRUNG 2 — FP ITERATION {k+1}/{K}   "
                  f"[mixture {len(history)}]   "
                  f"elapsed {_hms(time.time()-t_start)}{eta}\n{'='*72}", flush=True)

        stacked = tuple(stack_policies([h[j] for h in history])[0]
                        for j in range(N_TYPES))

        # ── verification price paths: FROZEN common noise, chunked ──────────
        A_ver = chunked(lambda a, b: price_fns[T_final](
            stacked, ver['P'][a:b], ver['eps0'][a:b], ver['idio_pop']),
            S, eval_chunk)

        if verbose:
            a = np.asarray(A_ver)
            print(f"  price over {S} frozen draws: mean {a.mean():.3f}  "
                  f"final median {np.median(a[:, -1]):.3f}  "
                  f"[p10 {np.quantile(a[:, -1], .1):.2f}, "
                  f"p90 {np.quantile(a[:, -1], .9):.2f}]  "
                  f"on-floor {float(np.mean(a <= floor + 1e-6)):.1%}", flush=True)

        # ── best responses ──────────────────────────────────────────────────
        t_br = time.time()
        new_triple, loss_hist = train_best_response_r2(
            stacked, jax.random.fold_in(k_iter_base, k), stage_fns, optimizer,
            mm_warm, T_grid, iters, lrs, verbose=verbose)
        br_secs = time.time() - t_br
        if verbose:
            print(f"    trained in {_hms(br_secs)}", flush=True)

        # ── exploitability on the frozen set ────────────────────────────────
        per_type = [
            exploitability_r2(env_f, j, new_triple[j], history, A_ver, ver['P'],
                              ver['idio_eval'][j], ret_fns[j], hist_fns[j],
                              model_static)
            for j in range(N_TYPES)]
        weighted     = sum(shares[j] * per_type[j]['gap'] for j in range(N_TYPES))
        weighted_rel = sum(shares[j] * per_type[j]['gap_relative']
                           for j in range(N_TYPES))

        if verbose:
            names = ("private     ", "large public", "market maker")
            print("  exploitability (mean over the frozen common-noise set):",
                  flush=True)
            for j, m in enumerate(per_type):
                flag = ("  <-- NEGATIVE: BR under-trained"
                        if m['gap'] < -m['gap_se'] else "")
                print(f"    {names[j]}  E = {m['gap']:+12.1f} +/- {m['gap_se']:8.1f}"
                      f"  ({m['gap_relative']:+8.4%})  "
                      f"[p10 {m['gap_p10']:+.4g}, p90 {m['gap_p90']:+.4g}]  "
                      f"V_br {m['V_br']:+11.5g}{flag}", flush=True)
            print(f"    weighted {weighted:+.2f}  ({weighted_rel:+.4%})", flush=True)

        # ── price-path movement, paired on the same frozen draws ────────────
        path_l2 = float('nan')
        if A_prev is not None:
            path_l2 = float(jnp.mean(jnp.linalg.norm(A_ver - A_prev, axis=1)))
            if verbose:
                print(f"  mean_s ||A^k - A^k-1||_2 = {path_l2:.6f}", flush=True)
        A_prev = A_ver

        rec = {
            'iteration': k + 1, 'mixture_size': len(history),
            'per_type': per_type, 'weighted': weighted,
            'weighted_relative': weighted_rel, 'path_l2': path_l2,
            'A_mean': float(np.mean(np.asarray(A_ver))),
            'A_final_median': float(np.median(np.asarray(A_ver)[:, -1])),
            'A_final_p10': float(np.quantile(np.asarray(A_ver)[:, -1], .1)),
            'A_final_p90': float(np.quantile(np.asarray(A_ver)[:, -1], .9)),
            'A_on_floor': float(np.mean(np.asarray(A_ver) <= floor + 1e-6)),
            'br_seconds': br_secs,
            'br_loss_last': [[float(L[-1, j]) for j in range(N_TYPES)]
                             for L in loss_hist],
        }

        if plot_dir:
            from helper_plot import plot_iteration_r2, plot_convergence_r2
            plot_iteration_r2(
                env_f, history, new_triple, plot_noise,
                os.path.join(plot_dir, f'iter_{k+1:03d}.pdf'),
                title_suffix=(f"  —  Rung 2, iteration {k+1}, T = {T_final}, "
                              f"B = {B}, S = {S}"),
                statics=statics)
            plot_convergence_r2(
                records + [rec], os.path.join(plot_dir, 'convergence.pdf'),
                title_suffix=(f"  —  T = {T_final}, B = {B}, S = {S}"))
            if verbose:
                print(f"  plots -> {plot_dir}", flush=True)

        rec['iter_seconds'] = time.time() - t_iter
        records.append(rec)
        history = (history + [new_triple])[-window:]

        if save_dir:
            save_triple(save_dir, k + 1, new_triple)
            _atomic_json(progress_path, {
                'rung': 2, 'gamma_ref': [float(x) for x in gamma_ref],
                'T_grid': list(T_grid), 'iters': list(iters), 'lrs': list(lrs),
                'B': B, 'S': S, 'M': M,
                'window': window, 'K': K,
                'agent_per_policy': env.agent_per_policy,
                'type_counts': list(env.type_counts),
                'type_shares': list(shares),
                'verify_seed': VERIFY_SEED,
                'mm_warmstart': mm_warmstart,
                'lr_schedule': lr_schedule, 'lr_min': lr_min,
                **(meta or {}), 'records': records})
            if verbose:
                print(f"  checkpoint saved  [iteration "
                      f"{_hms(rec['iter_seconds'])}]", flush=True)

    if verbose:
        print(f"\nTotal wall time: {_hms(time.time()-t_start)}", flush=True)
    return history, records


# =============================================================================
# from utils_rung3.py
# =============================================================================

VERIFY_SEED_R3 = 70707        # frozen evaluation blocks; never reused for training

CONTEXT_DIM = 4
CTX_NAMES = ('init_alloc', 'kappa', 'Afloor', 'financial_inst')

# The reference context, and the row order of the convergence sheet. The fixed
# rows are ONE-AT-A-TIME deviations from `ref`, so reading down a column of the
# sheet isolates a single regulatory instrument. BM_uniform is deliberately
# absent: it is held at the environment's constructor value and the conditioner
# never sees it, so a row varying it would be off-distribution.
CTX_REF = (GAMMA_REF[0], GAMMA_REF[1], A_FLOOR_BASE, 0.4)

_R = dict(zip(CTX_NAMES, CTX_REF))


def _ctx(**kw):
    d = dict(_R, **kw)
    return tuple(float(d[n]) for n in CTX_NAMES)


FIXED_SCENARIOS = (
    ('ref',        _ctx()),
    ('alloc_hi',   _ctx(init_alloc=0.9)),
    ('alloc_lo',   _ctx(init_alloc=0.1)),
    ('kappa_hi',   _ctx(kappa=5.0)),
    ('kappa_lo',   _ctx(kappa=1.0)),
    ('floor_lo',   _ctx(Afloor=FLOOR_MIN)),
    ('floor_hi',   _ctx(Afloor=FLOOR_MAX)),
    ('mm_thin',    _ctx(financial_inst=0.05)),
    ('mm_thick',   _ctx(financial_inst=MM_MAX)),
)
BLOCK_ORDER = ('prior',) + tuple(n for n, _ in FIXED_SCENARIOS)

_G_LO = jnp.array([ALLOC_MIN, KAPPA_MIN, FLOOR_MIN, MM_MIN], dtype=jnp.float32)
_G_HI = jnp.array([ALLOC_MAX, KAPPA_MAX, FLOOR_MAX, MM_MAX], dtype=jnp.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Binding a scenario onto the environment and onto a policy
# ═════════════════════════════════════════════════════════════════════════════

def norm_gamma(g):
    """Raw context -> [-1, 1]^4 for the FiLM input."""
    return 2.0 * (jnp.asarray(g, dtype=jnp.float32) - _G_LO) / (_G_HI - _G_LO) - 1.0


def bind_env(env, g):
    """The same environment under context g, all four leaves at once.

    `mm_share` rather than `financial_inst`: the latter fixes the SIMULATED
    population (type_counts, a static field) and must not move, while the former
    is the value the market responds to through context_weights.
    """
    return eqx.tree_at(
        lambda e: (e.init_alloc, e.kappa, e.Afloor, e.mm_share),
        env, (g[0], g[1], g[2], g[3]))


def bind_policy(model, g):
    """A BoundContextNN (or its params pytree) conditioned on g."""
    return eqx.tree_at(lambda m: m.ctx, model, norm_gamma(g))


def bind_stacked(stacked, g):
    """Same, for a stack of L mixture members: ctx has shape (L, 4)."""
    L = stacked.ctx.shape[0]
    return eqx.tree_at(lambda s: s.ctx, stacked,
                       jnp.broadcast_to(norm_gamma(g), (L, CONTEXT_DIM)))


def new_policy(key):
    return BoundContextNN(key=key)


def load_triple_r3(save_dir, k):
    """utils.load_triple deserialises into a TypePolicyNN; Rung 3 needs the
    FiLM module as the template."""
    like = new_policy(jax.random.PRNGKey(0))
    return tuple(
        eqx.tree_deserialise_leaves(
            os.path.join(save_dir, f'policy_{k}_type{j}.eqx'), like)
        for j in range(N_TYPES))


# ═════════════════════════════════════════════════════════════════════════════
# Frozen evaluation blocks
# ═════════════════════════════════════════════════════════════════════════════

def make_ver_block(env, T, S, M, seed, gamma_fixed=None):
    """One frozen block.

    `gamma_fixed=None` draws S scenarios from the prior, PAIRED one-to-one with
    the S noise draws — the correct Monte Carlo for E_{gamma~q}[E_j(gamma)].
    Otherwise all S rows carry the same scenario and the block resolves that one
    scenario over S noise draws.

    idio_pop is shared across the S rows, as in Rung 2, so the aggregate price
    path is a deterministic function of (policies, gamma_s, common noise_s).
    """
    kc, kp, ke, kg = jax.random.split(jax.random.PRNGKey(seed), 4)
    P, eps0 = draw_common_noise(env, T, S, kc)
    if gamma_fixed is None:
        gam = env.generate_context(kg, S)
    else:
        gam = jnp.broadcast_to(
            jnp.asarray(gamma_fixed, dtype=jnp.float32), (S, CONTEXT_DIM))
    return {
        'gam': gam, 'P': P, 'eps0': eps0, 'S': S, 'M': M,
        'idio_pop':  draw_idio(env, T, env.type_counts, kp),
        'idio_eval': draw_idio(env, T, (M, M, M), ke),
        # per-row population shares: financial_inst is part of the context now,
        # so the weighting of E_j across types varies row to row on the prior
        # block and must be applied BEFORE averaging over rows.
        'w': jax.vmap(lambda c: bind_env(env, c).context_weights())(gam),
    }


def make_ver_blocks(env, T, S, S_fixed, M, seed=VERIFY_SEED_R3):
    """The four blocks of §8.5.1–8.5.2, in row order."""
    blocks = {'prior': make_ver_block(env, T, S, M, seed)}
    for i, (name, g) in enumerate(FIXED_SCENARIOS):
        blocks[name] = make_ver_block(env, T, S_fixed, M, seed + 101 * (i + 1),
                                      gamma_fixed=g)
    return blocks


# ═════════════════════════════════════════════════════════════════════════════
# Jitted closures (env captured, never traced — env.T stays concrete)
# ═════════════════════════════════════════════════════════════════════════════

def make_price_fn_r3(env_T, statics):
    """Aggregate price path per (scenario, common-noise) row. -> (S, T+1)."""
    @eqx.filter_jit
    def fn(stacked, gam, P, eps0, idio_pop):
        def one(g, p, e):
            sm = tuple(bind_stacked(stacked[j], g) for j in range(N_TYPES))
            return bind_env(env_T, g).ensemble_price_path(
                sm, statics, p, e, idio_pop)
        return jax.vmap(one)(gam, P, eps0)
    return fn


def make_batched_return_fn_r3(env_T, j):
    """Returns of one policy over the rows of a block -> (S, M)."""
    @eqx.filter_jit
    def fn(policy, gam, A, P, idio):
        def one(g, a, p):
            _, R = bind_env(env_T, g).rollout_agent_batch(
                bind_policy(policy, g), j, a, p, idio)
            return R
        return jax.vmap(one)(gam, A, P)
    return fn


def make_batched_hist_fn_r3(env_T, j, static):
    """Same for the whole stacked mixture -> (S, L, M)."""
    @eqx.filter_jit
    def fn(stacked_params, gam, A, P, idio):
        def per_row(g, a, p):
            env_g = bind_env(env_T, g)
            sp    = bind_stacked(stacked_params, g)

            def per_policy(prm):
                _, R = env_g.rollout_agent_batch(
                    eqx.combine(prm, static), j, a, p, idio)
                return R
            return jax.vmap(per_policy)(sp)
        return jax.vmap(per_row)(gam, A, P)
    return fn


# ═════════════════════════════════════════════════════════════════════════════
# Best response — fresh (scenario, noise) at every gradient step
# ═════════════════════════════════════════════════════════════════════════════

def make_br_stage_fn_r3(env_T, statics, optimizer, nb_iterations, B, static):
    """One jitted best-response stage, all three types at once.

    Every gradient step draws B independent markets, and each market draws its
    own scenario alongside its own common and idiosyncratic noise. Cost per step
    is unchanged from Rung 2 — the B markets are vmapped into one wide batch, so
    a step is still one scan of T sequential timesteps.

    As in Rung 2 the price path is computed outside value_and_grad and
    stop_gradient'd: the responding agent is infinitesimal and does not move the
    price, and keeping it out means one forward evaluation per step rather than
    one per forward and one per backward.
    """
    T_int  = int(env_T.T)
    js     = jnp.arange(N_TYPES)
    counts = env_T.type_counts

    def sample_market(key):
        """B scenarios + the common and population-idiosyncratic noise."""
        kG, kP, kE, kI = jax.random.split(key, 4)
        gam  = env_T.generate_context(kG, B)
        P    = jax.vmap(env_T.generate_P,    in_axes=(0, None))(
            jax.random.split(kP, B), T_int)
        eps0 = jax.vmap(env_T.generate_eps0, in_axes=(0, None))(
            jax.random.split(kE, B), T_int)
        ki = jax.random.split(kI, N_TYPES)
        idio_pop = tuple(
            jax.vmap(lambda k, j=j: jax.vmap(
                env_T.generate_idiosyncratic_noise, in_axes=(0, None, None))(
                    jax.random.split(k, counts[j]),
                    env_T.agent_params[j, 7], T_int)
            )(jax.random.split(ki[j], B))
            for j in range(N_TYPES))
        return gam, P, eps0, idio_pop

    def prices(stacked_mix, gam, P, eps0, idio_pop):
        def one(g, p, e, i0, i1, i2):
            sm = tuple(bind_stacked(stacked_mix[j], g) for j in range(N_TYPES))
            return bind_env(env_T, g).ensemble_price_path(
                sm, statics, p, e, (i0, i1, i2))
        return jax.vmap(one)(gam, P, eps0, *idio_pop)

    def loss_given_market(p_stack, key, gam, A, P):
        """gam, A and P are constants here — the gradient only flows through the
        responding agent's own trajectory."""
        def per_type(p, j, kk):
            model = eqx.combine(p, static)
            idio  = jax.vmap(env_T.generate_idiosyncratic_noise,
                             in_axes=(0, None, None))(
                jax.random.split(kk, B), env_T.agent_params[j, 7], T_int)

            def one(g, a, pp, ii):              # one agent per market
                _, R = bind_env(env_T, g).rollout_agent_batch(
                    bind_policy(model, g), j, a, pp, ii[None, :])
                return R[0]

            R = jax.vmap(one)(gam, A, P, idio)
            # reward_scale depends only on the type's own parameters, not on
            # gamma, so one constant still normalises the whole family.
            return -jnp.mean(R) / env_T.reward_scale(j)

        per = jax.vmap(per_type, in_axes=(0, 0, 0))(
            p_stack, js, jax.random.split(key, N_TYPES))
        return jnp.sum(per), per

    @jax.jit
    def stage(p_stack, s, loop_key, stacked_mix):
        def one(carry, n):
            p_, s_ = carry
            k_noise, k_grad = jax.random.split(jax.random.fold_in(loop_key, n))
            gam, P, eps0, idio_pop = sample_market(k_noise)
            A = jax.lax.stop_gradient(prices(stacked_mix, gam, P, eps0, idio_pop))
            (_, per), g = jax.value_and_grad(loss_given_market, has_aux=True)(
                p_, k_grad, gam, A, P)
            upd, s_new = optimizer.update(g, s_, p_)
            return (eqx.apply_updates(p_, upd), s_new), per

        (p_f, s_f), losses = jax.lax.scan(one, (p_stack, s),
                                          jnp.arange(nb_iterations))
        return p_f, s_f, losses

    return stage


def train_best_response_r3(stacked_mix, key, stage_fns, optimizer, mm_warm,
                           T_grid, iters, lrs, verbose=True):
    """Best responses for all three types, from fresh FiLM networks (market
    maker warm-started, see pretrain_mm_r3.py)."""
    k_model, k_train = jax.random.split(key)
    mkeys  = jax.random.split(k_model, N_TYPES)
    models = [new_policy(mkeys[j]) for j in range(N_TYPES)]
    if mm_warm is not None:
        models[2] = mm_warm

    parts   = [eqx.partition(m, eqx.is_array) for m in models]
    static  = parts[0][1]
    p_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                     *[p for p, _ in parts])
    opt_state = optimizer.init(p_stack)

    loss_hist = []
    for i, T in enumerate(T_grid):
        if hasattr(opt_state, 'hyperparams'):
            opt_state.hyperparams['learning_rate'] = jnp.array(lrs[i])
        p_stack, opt_state, losses = stage_fns[T](
            p_stack, opt_state, jax.random.fold_in(k_train, i), stacked_mix)
        losses = np.asarray(losses)
        loss_hist.append(losses)
        if verbose:
            per = "  ".join(f"t{j}: {losses[0, j]:+.5f}->{losses[-1, j]:+.5f}"
                            for j in range(N_TYPES))
            print(f"    T={T:<5d} {per}", flush=True)

    triple = tuple(
        eqx.combine(jax.tree_util.tree_map(lambda x: x[j], p_stack), static)
        for j in range(N_TYPES))
    return triple, loss_hist


# ═════════════════════════════════════════════════════════════════════════════
# Exploitability, per block
# ═════════════════════════════════════════════════════════════════════════════

def exploitability_r3(env_f, j, new_pol, stacked_hist, gam, A, P, idio,
                      ret_fn, hist_fn):
    """E_j on one frozen block.

    Both terms use the same scenario, the same price path and the same
    idiosyncratic draws per row, so the difference is paired at every level. On
    the prior block the row-to-row spread mixes scenario and noise variation,
    which is why the fixed blocks exist alongside it.
    """
    R_br   = ret_fn(new_pol, gam, A, P, idio)              # (S, M)
    R_hist = hist_fn(stacked_hist, gam, A, P, idio)        # (S, L, M)
    R_mix  = jnp.mean(R_hist, axis=1)                      # (S, M)

    d_row = jnp.mean(R_br - R_mix, axis=1)                 # (S,)
    gap   = float(jnp.mean(d_row))
    se    = float(jnp.std(d_row, ddof=1) / jnp.sqrt(d_row.shape[0]))
    scale = float(env_f.reward_scale(j))
    return d_row, {
        'gap': gap, 'gap_se': se, 'gap_relative': gap / scale,
        'V_br':  float(jnp.mean(R_br)),
        'V_mix': float(jnp.mean(R_mix)),
        'V_br_relative': float(jnp.mean(R_br)) / scale,
        'scale': scale,
        'gap_p10': float(jnp.quantile(d_row, 0.1)),
        'gap_p90': float(jnp.quantile(d_row, 0.9)),
    }


def eval_block(env_f, blk, stacked, stacked_hist, new_triple, price_fn,
               ret_fns, hist_fns, eval_chunk, shares, floor, A_prev):
    """Price the block, then measure every type on it.

    The population weighting is applied PER ROW. financial_inst is part of the
    context, so on the prior block each row carries its own share vector; a
    single fixed `shares` there would score a market holding almost no market
    maker as though it held 40%. On a fixed block every row shares one context
    and this reduces to the old formula exactly.
    """
    A = chunked(lambda a, b: price_fn(stacked, blk['gam'][a:b], blk['P'][a:b],
                                      blk['eps0'][a:b], blk['idio_pop']),
                blk['S'], eval_chunk)

    rows, per_type = [], []
    for j in range(N_TYPES):
        d_row, m = exploitability_r3(
            env_f, j, new_triple[j], stacked_hist[j],
            blk['gam'], A, blk['P'], blk['idio_eval'][j],
            ret_fns[j], hist_fns[j])
        rows.append(np.asarray(d_row))
        per_type.append(m)

    w = np.asarray(blk['w'])                       # (S, 3)
    abs_rows = np.stack(rows, axis=1)              # (S, 3) reward units
    rel_rows = abs_rows / np.array([m['scale'] for m in per_type])[None, :]
    wtd_abs = np.sum(w * abs_rows, axis=1)
    wtd_rel = np.sum(w * rel_rows, axis=1)

    a = np.asarray(A)
    # the floor is part of the context, so "on floor" is measured against each
    # market's own support rather than one global constant
    fl = np.asarray(blk['gam'])[:, 2][:, None]
    out = {
        'per_type': per_type,
        'weighted': float(wtd_abs.mean()),
        'weighted_relative': float(wtd_rel.mean()),
        'weighted_relative_se': float(wtd_rel.std(ddof=1)
                                      / np.sqrt(wtd_rel.size)),
        'mean_shares': [float(x) for x in w.mean(axis=0)],
        'path_l2': (float('nan') if A_prev is None else
                    float(jnp.mean(jnp.linalg.norm(A - A_prev, axis=1)))),
        'A_mean': float(a.mean()),
        'A_final_median': float(np.median(a[:, -1])),
        'A_final_p10': float(np.quantile(a[:, -1], .1)),
        'A_final_p90': float(np.quantile(a[:, -1], .9)),
        'A_on_floor': float(np.mean(a <= fl + 1e-6)),
        'A_on_floor_ref': float(np.mean(a <= floor + 1e-6)),
    }
    return out, A


# ═════════════════════════════════════════════════════════════════════════════
# Fictitious play, Rung 3
# ═════════════════════════════════════════════════════════════════════════════

def fictitious_play_rung3(env, K=50, T_grid=(365,), iters=(8000,), lrs=(1e-3,),
                          B=256, S=500, S_fixed=250, M=32, eval_chunk=50,
                          window=200, key=None, save_dir=None, plot_dir=None,
                          mm_warmstart=None, lr_schedule=None, lr_min=None,
                          verbose=True, meta=None):
    """Rung 3 fictitious play — the BE layer, and the object of the study.

    What changes from Rung 2
    ------------------------
    The scenario stops being frozen. Every market in the batch draws its OWN
    context c = (init_alloc, kappa, Afloor, financial_inst) from the prior q,
    jointly with its own common and idiosyncratic noise, and the policy is
    conditioned on that context through the FiLM conditioner. Within a market
    the draw is shared by every agent, every type and every t — which is what
    makes this a Bayesian-Extended MFG rather than a market with heterogeneous
    regimes.

    What it is for
    --------------
    One Nash solve that covers a whole family of regulatory settings instead of
    one cell of it. The quantity BE-MFG says must decrease is

        E_{c ~ q} [ E_j(c) ]

    estimated on the frozen `prior` block. The remaining blocks each pin one
    fixed context and exist as a diagnostic: the prior average can look healthy
    while a corner of the box is unsolved, and a depressed V_br on one fixed
    block is the signature of a conditional deviator that is under-trained
    exactly there. Because they are one-at-a-time deviations from `ref`, reading
    down a column of the convergence sheet isolates one regulatory instrument.

    Population weighting
    --------------------
    `financial_inst` is part of the context, so the population shares differ from
    row to row on the prior block. The weighted exploitability is therefore
    formed PER ROW and then averaged; a single share vector would score a market
    holding almost no market maker as though it held the reference proportion.

    Resumable, like Rungs 1 and 2.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    k_init_base, k_iter_base = jax.random.split(key)

    # The env's own context is a placeholder: every rollout rebinds it per
    # market. GAMMA_REF is used only so the unbound object is well defined.
    env     = env.set_context(jnp.asarray(GAMMA_REF, dtype=jnp.float32))
    T_final = T_grid[-1]
    envs_T  = {T: env.set_T(T) for T in T_grid}
    env_f   = envs_T[T_final]

    blocks = make_ver_blocks(env_f, T_final, S, S_fixed, M)
    if verbose:
        print(f"frozen blocks: prior S={S} joint (gamma, noise) draws; "
              f"{', '.join(n for n, _ in FIXED_SCENARIOS)} S={S_fixed} each; "
              f"M={M} paired agents/type, chunk={eval_chunk}", flush=True)
        for name, g in FIXED_SCENARIOS:
            print(f"    {name:<10s} init_alloc={g[0]:.2f}  kappa={g[1]:.2f}",
                  flush=True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    progress_path = os.path.join(save_dir, 'progress.json') if save_dir else None

    # ── resume ──────────────────────────────────────────────────────────────
    records, start_k, resumed = [], 0, False
    if progress_path and os.path.exists(progress_path):
        with open(progress_path) as f:
            prev = json.load(f)
        records = prev.get('records', [])
        n_avail = 0
        while os.path.exists(os.path.join(save_dir, f'policy_{n_avail}_type0.eqx')):
            n_avail += 1
        start_k = min(len(records), max(0, n_avail - 1))
        records = records[:start_k]
        lo = max(0, start_k + 1 - window)
        history = [load_triple_r3(save_dir, i) for i in range(lo, start_k + 1)]
        resumed = True
        if verbose:
            print(f"=== RESUMING at iteration {start_k+1} "
                  f"({len(history)} triples restored) ===", flush=True)

    # The warm start is loaded BEFORE anything is written: a missing file should
    # fail on an empty directory, not after policy_0 is on disk.
    mm_warm = None
    if mm_warmstart:
        mm_warm = eqx.tree_deserialise_leaves(
            mm_warmstart, new_policy(jax.random.PRNGKey(0)))
        if verbose:
            print(f"market-maker warm start: {mm_warmstart}", flush=True)

    if not resumed:
        ik = jax.random.split(k_init_base, N_TYPES)
        history = [tuple(new_policy(ik[j]) for j in range(N_TYPES))]
        if save_dir:
            save_triple(save_dir, 0, history[0])
    if start_k >= K:
        return history, records

    _, model_static = eqx.partition(history[0][0], eqx.is_array)
    statics   = tuple(stack_policies([history[0][j]])[1] for j in range(N_TYPES))
    price_fns = {T: make_price_fn_r3(envs_T[T], statics) for T in T_grid}
    ret_fns   = [make_batched_return_fn_r3(env_f, j) for j in range(N_TYPES)]
    hist_fns  = [make_batched_hist_fn_r3(env_f, j, model_static)
                 for j in range(N_TYPES)]

    if lr_schedule == 'cosine':
        if len(T_grid) != 1:
            raise ValueError("cosine lr expects a single horizon")
        lo_lr = lr_min if lr_min is not None else lrs[0] / 10.0
        optimizer = optax.adam(optax.cosine_decay_schedule(
            init_value=lrs[0], decay_steps=iters[0], alpha=lo_lr / lrs[0]))
        if verbose:
            print(f"lr: cosine {lrs[0]:g} -> {lo_lr:g} over {iters[0]} steps",
                  flush=True)
    else:
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lrs[0])
    stage_fns = {T: make_br_stage_fn_r3(envs_T[T], statics, optimizer,
                                        iters[i], B, model_static)
                 for i, T in enumerate(T_grid)}

    plot_noise = None
    if plot_dir:
        from helper_plot import make_plot_noise_r3
        plot_noise = make_plot_noise_r3(env_f, T_final)
        os.makedirs(plot_dir, exist_ok=True)
        if verbose:
            print("state/action sheets: one fixed common-noise draw at "
                  "GAMMA_REF (helper_plot.PLOT_SEED)", flush=True)

    shares, floor = env.type_shares, float(env.Afloor)
    t_start = time.time()
    A_prev  = {name: None for name in BLOCK_ORDER}

    for k in range(start_k, K):
        t_iter = time.time()
        if verbose:
            done = k - start_k
            eta = ("" if done == 0 else
                   f"   ETA {_hms((time.time()-t_start)/done*(K-k))}")
            print(f"\n{'='*72}\nRUNG 3 — FP ITERATION {k+1}/{K}   "
                  f"[mixture {len(history)}]   "
                  f"elapsed {_hms(time.time()-t_start)}{eta}\n{'='*72}", flush=True)

        stacked = tuple(stack_policies([h[j] for h in history])[0]
                        for j in range(N_TYPES))

        # ── best responses ──────────────────────────────────────────────────
        t_br = time.time()
        new_triple, loss_hist = train_best_response_r3(
            stacked, jax.random.fold_in(k_iter_base, k), stage_fns, optimizer,
            mm_warm, T_grid, iters, lrs, verbose=verbose)
        br_secs = time.time() - t_br
        if verbose:
            print(f"    trained in {_hms(br_secs)}", flush=True)

        # Full best-response training curves. The record keeps only the last
        # value; the appendix wants the whole descent, and one array per
        # iteration is ~150 KB, so it goes to its own file rather than into
        # progress.json.
        loss_file = None
        if save_dir:
            os.makedirs(os.path.join(save_dir, 'losses'), exist_ok=True)
            loss_file = os.path.join('losses', f'br_loss_{k+1:03d}.npz')
            np.savez_compressed(
                os.path.join(save_dir, loss_file),
                **{f'stage{i}_T{T}': np.asarray(L, dtype=np.float32)
                   for i, (T, L) in enumerate(zip(T_grid, loss_hist))})

        # ── the four frozen blocks ──────────────────────────────────────────
        blk_recs = {}
        for name in BLOCK_ORDER:
            rec_b, A = eval_block(env_f, blocks[name], stacked, stacked,
                                  new_triple, price_fns[T_final], ret_fns,
                                  hist_fns, eval_chunk, shares, floor,
                                  A_prev[name])
            A_prev[name] = A
            blk_recs[name] = rec_b

            if verbose:
                head = ("E_{gamma~q}" if name == 'prior' else name)
                print(f"  [{head}]  price mean {rec_b['A_mean']:.3f}  "
                      f"final median {rec_b['A_final_median']:.3f}  "
                      f"on-floor {rec_b['A_on_floor']:.1%}", flush=True)
                names = ("private     ", "large public", "market maker")
                for j, m in enumerate(rec_b['per_type']):
                    flag = ("  <-- NEGATIVE: BR under-trained"
                            if m['gap'] < -m['gap_se'] else "")
                    print(f"      {names[j]}  E = {m['gap']:+12.1f} "
                          f"+/- {m['gap_se']:8.1f}  ({m['gap_relative']:+8.4%})"
                          f"  V_br/scale {m['V_br_relative']:+7.2%}{flag}",
                          flush=True)
                print(f"      weighted {rec_b['weighted']:+.2f}  "
                      f"({rec_b['weighted_relative']:+.4%})   "
                      f"||A^k-A^k-1||_2 {rec_b['path_l2']:.4f}", flush=True)

        rec = {
            'iteration': k + 1, 'mixture_size': len(history),
            'blocks': blk_recs,
            # flattened aliases so Rung 2 readers keep working
            'per_type': blk_recs['prior']['per_type'],
            'weighted': blk_recs['prior']['weighted'],
            'weighted_relative': blk_recs['prior']['weighted_relative'],
            'path_l2': blk_recs['prior']['path_l2'],
            'br_seconds': br_secs,
            'br_loss_file': loss_file,
            'br_loss_last': [[float(L[-1, j]) for j in range(N_TYPES)]
                             for L in loss_hist],
        }

        if plot_dir:
            from helper_plot import (plot_iteration_r3, plot_convergence_r3,
                                    plot_deviator_r3)
            plot_iteration_r3(
                env_f, history, new_triple, plot_noise,
                os.path.join(plot_dir, f'iter_{k+1:03d}.pdf'),
                title_suffix=(f"  —  Rung 3 at GAMMA_REF, iteration {k+1}, "
                              f"T = {T_final}, B = {B}"),
                statics=statics)
            plot_convergence_r3(
                records + [rec], os.path.join(plot_dir, 'convergence.pdf'),
                title_suffix=(f"  —  T = {T_final}, B = {B}, S = {S}"))
            plot_deviator_r3(
                records + [rec], os.path.join(plot_dir, 'deviator.pdf'),
                title_suffix=(f"  —  T = {T_final}, B = {B}"))
            if verbose:
                print(f"  plots -> {plot_dir}", flush=True)

        rec['iter_seconds'] = time.time() - t_iter
        records.append(rec)
        history = (history + [new_triple])[-window:]

        if save_dir:
            save_triple(save_dir, k + 1, new_triple)
            _atomic_json(progress_path, {
                'rung': 3,
                'T_grid': list(T_grid), 'iters': list(iters), 'lrs': list(lrs),
                'B': B, 'S': S, 'S_fixed': S_fixed, 'M': M,
                'window': window, 'K': K,
                'context_dim': CONTEXT_DIM,
                'context_names': list(CTX_NAMES),
                'context_ref': list(CTX_REF),
                'context_box': [[float(lo), float(hi)]
                                for lo, hi in zip(_G_LO, _G_HI)],
                'gamma_box': [[float(ALLOC_MIN), float(ALLOC_MAX)],
                              [float(KAPPA_MIN), float(KAPPA_MAX)]],
                'fixed_scenarios': {n: list(g) for n, g in FIXED_SCENARIOS},
                'agent_per_policy': env.agent_per_policy,
                'type_counts': list(env.type_counts),
                'type_shares': list(shares),
                'verify_seed': VERIFY_SEED_R3,
                'mm_warmstart': mm_warmstart,
                'lr_schedule': lr_schedule, 'lr_min': lr_min,
                **(meta or {}), 'records': records})
            if verbose:
                print(f"  checkpoint saved  [iteration "
                      f"{_hms(rec['iter_seconds'])}]", flush=True)

    if verbose:
        print(f"\nTotal wall time: {_hms(time.time()-t_start)}", flush=True)
    return history, records
