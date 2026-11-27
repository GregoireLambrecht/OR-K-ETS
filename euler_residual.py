"""No-arbitrage (Euler) residual of the minors' pricing condition, on a finished
major-minor run (no conditional expectation needed).

    python3 euler_residual.py --run-dir results_majmin/<run> [--blocks prior ref --margin 0.9]

Minors only. Private producers: the Euler residual below. Financial institutions never
emit (Y = 0, they hold no allowances to T for compliance), so they are tested with the
PRICE-MARTINGALE condition instead: (A_{t+1} - A_t) / A_til_t over their interior orders
(t < T - 1), which must be ~ 0 if they can buy today and sell tomorrow.
At the end of a market each
particle's realized marginal value of one allowance is the derivative of its terminal
penalty g(x_T) (envs/environment.py, terminal_reward):
    index penalty (the study):      g = kappa^2 A_avg_T (E_T - theta_T)^+,  Y^i = kappa^2 A_avg_T 1{E^i_T > theta^i_T}
    fixed fine (the LP benchmark):  g = kappa^1 (E_T - theta_T)^+,          Y^i = kappa^1 1{E^i_T > theta^i_T}
with theta_T the holdings after the last trade and E_T the emissions including the
last day's noise. This is CHECKED against a finite difference of the environment's
own terminal_reward on every particle away from the kink.

A minor's order on day t is INTERIOR if |f| <= margin * C^f, f >= -margin * theta_t
(the tanh schedule approaches C^f without reaching it, so a near-saturated order is
not interior) and its side was fully filled (rho = 1). Selection uses day-t
information only. Optimality says E[Y^i - A_t | F_t] = 0 for interior orders; the
residual is reported relative to the running mean price A_til_t = mean_{s<=t} A_s (the
penalty index after day t): (Y^i - A_t) / A_til_t (days with A_til_t < a_min dropped).

Markets: the frozen evaluation blocks of the run (VERIFY_SEED), simulated with the
final mixture. Each particle is replayed along its market's frozen path with its own
component and noise (checked against the simulated final states). Intervals are
CLUSTERED BY MARKET: average within each market first, then mean and 95% interval
(1.96 sd / sqrt(#markets)) across markets.
Figures in <run-dir>/euler/:
  euler_<block>          relative residual vs day t over interior orders, per type
  euler_vs_f_<block>     relative residual vs f / C^f over ALL orders (binned), per type:
                         a positive residual concentrated near f/C^f = 1 is the tanh
                         parametrisation, not an equilibrium failure
"""
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
ap.add_argument('--blocks', nargs='+', default=['prior', 'ref'])
ap.add_argument('--margin', type=float, default=0.9)
ap.add_argument('--a-min', type=float, default=1.0, help='USD: days whose running mean price A_til_t is below are dropped')
ap.add_argument('--bins', type=int, default=20)
ap.add_argument('--last-br', action='store_true',
                help='every player uses only the LAST best response (policy_K, k = 1) instead of the mixture')
args = ap.parse_args()

from majmin_eval_common import read_progress, set_env_from_progress
prog = read_progress(args.run_dir)
set_env_from_progress(prog, prog['floor_max'])
os.environ['KETS_CLEAR_ITERS'] = str(prog.get('clear_iters', 40))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import paper_style as ps
from envs.base_params import PENALTY_MODE
from envs.environment_majmin import MINORS
from envs.environment_v2 import stratified_components
from envs.models_majmin import gather_apply
from utils import norm_gamma, _atomic_json, CTX_REF
from utils_majmin import make_sim, make_blocks, COMP_SEED
from majmin_eval_common import make_env, load_mixture

# Y = derivative of the penalty in theta_T: index penalty kappa^2 A_avg (E - theta)^+ (the study),
# or the fixed fine kappa^1 (E - theta)^+ (the LP benchmark, KETS_PENALTY=fixed)
T, n, rung = int(prog['T_grid'][0]), int(prog['n_sim']), int(prog['rung'])
env = make_env(T)
mix = load_mixture(args.run_dir, prog)
if args.last_br:                       # a population of the last best responses only (k = 1)
    from utils_majmin import make_buffers
    bufs, stats = make_buffers([mix['nets'][mix['K']]])
    mix.update(k=1, minor_buf=tuple(bufs[j] for j in MINORS), minor_static=tuple(stats[j] for j in MINORS),
               maj_buf=bufs[1], maj_static=stats[1])
k = mix['k']
SUF = '_lastbr' if args.last_br else ''
sim = make_sim(env, mix['minor_static'], mix['maj_static'])
blocks, _ = make_blocks(env, rung, tuple(prog.get('context', CTX_REF)), T, n,
                        int(prog['S']), int(prog.get('S_fixed', 100)), int(prog['M']))
out_dir = os.path.join(args.run_dir, 'euler')
os.makedirs(out_dir, exist_ok=True)
idx = env.dic_indexes
NAMES = {0: ('private producer', 'C0'), 2: ('financial inst.', 'C2')}
print(f"Euler residual (minors) | {args.run_dir} | {'LAST BR only' if args.last_br else 'mixture'} | T={T} n={n} k={k} | blocks {args.blocks} | "
      f"interior: |f| <= {args.margin} C^f, f >= -{args.margin} theta, rho = 1", flush=True)


def replay(g, A_traj, rho_traj, H_traj, P, idio, comp, a):
    """Type MINORS[a] particles along the frozen path. Per day: order f, its C^f, the
    pre-trade holdings and whether its side was fully filled. At T: Y, the finite-
    difference check of Y, the distance to the kink, the final holdings."""
    j = MINORS[a]
    eg = env.bind(g)
    ctx = norm_gamma(g)
    X0 = jnp.broadcast_to(eg.type_initial_state(j), (n, 16))

    def step(X, t):
        A_t, rho = A_traj[t + 1], rho_traj[t + 1]
        inp = jax.vmap(eg.net_input, in_axes=(0, None))(X, H_traj[t])
        out = gather_apply(mix['minor_buf'][a], mix['minor_static'][a], comp, inp, ctx)
        al, be, lo, hi, xi, eta = jax.vmap(eg.schedule_params)(X, out)
        f = jax.vmap(eg.trade_at, in_axes=(0, 0, 0, 0, 0, None))(X, al, be, lo, hi, A_t)
        filled = jnp.where(f > 0, rho[1] >= 1.0, jnp.where(f < 0, rho[0] >= 1.0, (rho[0] >= 1.0) & (rho[1] >= 1.0)))
        ex = jnp.where(f >= 0, rho[1] * f, rho[0] * f)
        Xn = jax.vmap(eg.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
            X, jnp.stack([ex, xi, eta], axis=1), idio[:, t], A_t, P[t + 1])
        return Xn, (f, X[:, idx['cap_f']], X[:, idx['theta']], filled)

    Xf, (f, capf, th, filled) = jax.lax.scan(step, X0, jnp.arange(T))     # each (T, n)
    E = jnp.sum(Xf[:, idx['xi']] * Xf[:, idx['e']], axis=-1) + Xf[:, idx['eps']]
    gap = E - Xf[:, idx['theta']]                                          # (n,) E_T - theta_T
    unit = 1.0 if PENALTY_MODE == 'fixed' else Xf[:, idx['A_avg']]
    Y = eg.kappa * unit * (gap > 0)
    # check: Y = d(-g)/d(theta_T), by a one-sided finite difference of terminal_reward
    dl = 1.0                       # one allowance: float32-safe against a penalty ~ 1e5
    Xp = Xf.at[:, idx['theta']].add(dl)
    Y_fd = (jax.vmap(eg.terminal_reward)(Xp) - jax.vmap(eg.terminal_reward)(Xf)) / dl
    return f.T, capf.T, th.T, filled.T, Y, Y_fd, gap, Xf[:, idx['theta']]


def clustered(vals, mask, w_market):
    """vals, mask (S, P) per market over P items, weights (S,) unused if None: market means
    of the selected items, then mean and 95% half-width across the markets that have any."""
    cnt = mask.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mm = np.where(cnt > 0, (np.where(mask, vals, 0.0)).sum(1) / cnt, np.nan)
    ok = np.isfinite(mm)
    m = int(ok.sum())
    if m < 2:
        return np.nan, np.nan, m
    return float(mm[ok].mean()), float(1.96 * mm[ok].std(ddof=1) / np.sqrt(m)), m


summary = {}
edges = np.linspace(-1.0, 1.0, args.bins + 1)
for nm in args.blocks:
    b = blocks[nm]
    S = b['S']
    ck = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(COMP_SEED), k), S)
    u = jnp.arange(S) % k

    @jax.jit
    def run_block():
        A, rho, H, r0, ys, (Xf, X0f) = jax.vmap(lambda g, p, e, im, c, uu: sim(
            mix['maj_buf'], mix['minor_buf'], jnp.int32(k), g, p, e, im, b['imin'], c, uu, False))(
            b['gam'], b['P'], b['eps0'], b['imaj'], ck, u)
        res = []
        for a in range(len(MINORS)):
            comp = jax.vmap(lambda c: stratified_components(jax.random.split(c, len(MINORS))[a], k, n))(ck)
            out = jax.vmap(lambda g, aa, rr, hh, p, cc: replay(g, aa, rr, hh, p, b['imin'][a], cc, a))(
                b['gam'], A, rho, H, b['P'], comp)
            res.append(out + (Xf[a][..., idx['theta']],))
        return A, res
    A, res = run_block()
    A = np.asarray(A)[:, 1:]                                       # (S, T): A_t cleared on day t
    res = [[np.asarray(x) for x in r] for r in res]
    chk = max(float(np.max(np.abs(r[7] - r[8]) / np.maximum(np.abs(r[8]), 1.0))) for r in res)
    assert chk < 1e-3, f'the replay does not reproduce the market ({chk:.2e}): STOP'
    # Y against the finite difference of the environment's penalty, away from the kink
    fd_err, n_fd = 0.0, 0
    for r in res:
        away = np.abs(r[6]) > 2.0                                  # |E - theta| > 2x the FD step
        fd_err = max(fd_err, float(np.max(np.abs(r[4] - r[5])[away] / np.maximum(r[4][away], 1.0))))
        n_fd += int(away.sum())
    print(f"[{nm}] replay vs simulation: max rel diff {chk:.2e} | Y vs finite difference of terminal_reward: "
          f"max rel diff {fd_err:.2e} over {n_fd} particles", flush=True)
    assert fd_err < 1e-2, 'Y is not the derivative of the environment penalty: STOP'

    A_til = np.cumsum(A, axis=1) / np.arange(1, T + 1)[None, :]    # (S, T): running mean of A up to day t
    valid_day = A_til > args.a_min                                 # (S, T): days with A_til < a_min dropped
    norm = np.where(valid_day, A_til, np.nan)                      # residuals are relative to A_til_t
    curves, vs_f = {}, {}
    per_type = {}
    for a, j in enumerate(MINORS):
        f, capf, th, filled, Y = res[a][0], res[a][1], res[a][2], res[a][3], res[a][4]    # (S, n, T), Y (S, n)
        x = f / capf
        if j == 0:      # producers: Euler residual against the terminal value of an allowance
            rel = (Y[:, :, None] - A[:, None, :]) / norm[:, None, :]
        else:           # financial institutions never emit (Y = 0): price-martingale residual
            mart = np.concatenate([(A[:, 1:] - A[:, :-1]), np.full((S, 1), np.nan)], axis=1)
            rel = np.broadcast_to((mart / norm)[:, None, :], f.shape)
        interior = (np.abs(f) <= args.margin * capf) & (f >= -args.margin * th) & filled & valid_day[:, None, :]
        per_type[j] = (x, rel, interior, valid_day[:, None, :] & np.isfinite(rel))
    # per day, interior orders: per type and both types (mass-weighted within a market)
    w = np.asarray(jax.vmap(lambda g: env.bind(g).context_weights())(b['gam']))
    for lab, js in ((NAMES[0][0], (0,)), (NAMES[2][0], (2,))):
        mean, half, share = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
        for t in range(T):
            num = sum(w[:, j] * np.where(per_type[j][2][:, :, t], per_type[j][1][:, :, t], 0.0).sum(1) for j in js)
            den = sum(w[:, j] * per_type[j][2][:, :, t].sum(1) for j in js)
            tot = sum(w[:, j] * n for j in js)
            with np.errstate(invalid='ignore', divide='ignore'):
                mm = np.where(den > 0, num / den, np.nan)
            ok = np.isfinite(mm)
            if ok.sum() >= 2:
                mean[t] = mm[ok].mean(); half[t] = 1.96 * mm[ok].std(ddof=1) / np.sqrt(ok.sum())
            share[t] = np.mean(den / tot)
        curves[lab] = dict(mean=mean, half=half, share=share)
    # versus f / C^f, all orders of all days (binned), per type, clustered by market
    for j in MINORS:
        x, rel, _, okd = per_type[j]
        xs, rs, ms = x.reshape(S, -1), rel.reshape(S, -1), okd.reshape(S, -1)
        rows = []
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            sel = ms & (xs >= lo_) & ((xs < hi_) | (hi_ == 1.0) & (xs <= hi_))
            mu, hw, m = clustered(rs, sel, None)
            rows.append((0.5 * (lo_ + hi_), mu, hw, m, float(sel.sum()) / max(1, ms.sum())))
        vs_f[j] = np.array(rows)

    # ── figures ──────────────────────────────────────────────────────────────
    ps.apply()
    t = np.arange(T)
    fig, axs = plt.subplots(2, 1, figsize=(4.2, 3.8), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for lab, col, lab2 in ((NAMES[0][0], NAMES[0][1], r'private producer: $(Y - A_t)/\tilde A_t$'),
                           (NAMES[2][0], NAMES[2][1], r'financial inst.: $(A_{t+1} - A_t)/\tilde A_t$')):
        c = curves[lab]
        axs[0].plot(t, 100 * c['mean'], color=col, lw=1.0, label=lab2)
        axs[0].fill_between(t, 100 * (c['mean'] - c['half']), 100 * (c['mean'] + c['half']), color=col, alpha=0.15, lw=0)
        axs[1].plot(t, 100 * c['share'], color=col, lw=0.9)
    axs[0].axhline(0, color='0.5', lw=0.8)
    axs[0].set_ylabel(r'residual / $\tilde A_t$ (%)')
    axs[0].legend(fontsize=6, frameon=False)
    axs[0].set_title(f"{'last BR' if args.last_br else 'mixture'}, block '{nm}', {S} markets: interior orders (|f| $\\leq$ {args.margin} $C^f$, "
                     f"f $\\geq$ -{args.margin} $\\theta$, $\\rho$ = 1); 95% CI by market", fontsize=6)
    axs[1].set_ylabel('interior (%)'); axs[1].set_xlabel('Day $t$')
    fig.tight_layout()
    ps.save(fig, os.path.join(out_dir, f'euler_{nm}{SUF}'), check=False)

    fig, axs = plt.subplots(2, 1, figsize=(4.2, 3.8), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for j in MINORS:
        r = vs_f[j]; lab, col = NAMES[j]
        axs[0].errorbar(r[:, 0], 100 * r[:, 1], yerr=100 * r[:, 2], color=col, marker='o', ms=2.5, lw=0.9,
                        capsize=1.5, label=lab)
        axs[1].bar(r[:, 0] + (0.02 if j == 2 else -0.02), 100 * r[:, 4], width=0.04, color=col)
    for ax in axs:
        for v in (-args.margin, args.margin):
            ax.axvline(v, color='0.6', ls=':', lw=0.8)
    axs[0].axhline(0, color='0.5', lw=0.8)
    axs[0].set_ylabel(r'residual / $\tilde A_t$ (%)'); axs[0].legend(fontsize=6, frameon=False)
    axs[0].set_title(f"{'last BR' if args.last_br else 'mixture'}, block '{nm}': all orders, binned by f / $C^f$; 95% CI by market", fontsize=6)
    axs[1].set_ylabel('orders (%)'); axs[1].set_xlabel('$f / C^f$ (order at the cleared price)')
    fig.tight_layout()
    ps.save(fig, os.path.join(out_dir, f'euler_vs_f_{nm}{SUF}'), check=False)

    np.savez_compressed(os.path.join(out_dir, f'euler_{nm}{SUF}.npz'), A=A, edges=edges,
                        **{f'day_{k_}_{lab.replace(" ", "_").replace(".", "")}': v
                           for lab, c in curves.items() for k_, v in c.items()},
                        **{f'vs_f_type{j}': vs_f[j] for j in MINORS})
    c = curves[NAMES[0][0]]
    ok = np.isfinite(c['mean'])
    summary[nm] = {'markets': S, 'replay_check': chk, 'Y_vs_fd_check': fd_err, 'margin': args.margin,
                   **{f'{lab}: mean relative residual (interior)': float(np.nanmean(curves[lab]['mean']))
                      for lab in curves},
                   'days_with_zero_outside_interval (producers)': int(np.sum(np.abs(c['mean'][ok]) > c['half'][ok])),
                   'days': int(ok.sum()), 'mean_price': float(A.mean()),
                   **{f'type{j} vs f/C^f (bin center, mean, half, markets, share)': vs_f[j].tolist() for j in MINORS}}
    print(f"[{nm}] relative residual over interior orders, mean over days: " +
          ", ".join(f"{lab} {100 * np.nanmean(curves[lab]['mean']):+.1f}%" for lab in curves) +
          f" | producers: 0 outside the 95% CI on {summary[nm]['days_with_zero_outside_interval (producers)']}/"
          f"{summary[nm]['days']} days", flush=True)
    for j in MINORS:
        r = vs_f[j]
        print(f"   {NAMES[j][0]:17s} vs f/C^f: " + " ".join(
            f"[{c_:+.2f}] {100 * m_:+.0f}%({100 * s_:.0f}%)" for c_, m_, _, _, s_ in r if np.isfinite(m_)), flush=True)
_atomic_json(os.path.join(out_dir, f'summary{SUF}.json'), summary)
print(f"-> {out_dir}", flush=True)
