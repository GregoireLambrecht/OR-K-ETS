"""Rung 3 market-maker warm start — the Rung 1/2 fit, on a FiLM network.

Why it has to be refit
----------------------
The saved `mm_warmstart/mm_policy.eqx` holds a `TypePolicyNN`; Rung 3 policies
are `BoundContextNN`, so the leaves do not load. The TARGET is unchanged:
`MARKET_MAKER[0] = 0`, so theta_0 = 0 x multiplier x init_alloc = 0 for *every*
scenario. The zero-gradient trap of plan.md §6.3 is gamma-independent, and one
warm start covers the whole box — the network only has to learn to ignore the
context on the f head.

So the fit is the same buy-then-sell regression, with gamma drawn from the
prior alongside the synthetic state:

    if t_n <= buy_frac:  buy at maximum          (hat_f = +1)
    else:                sell down proportionally to the time remaining

  python3 pretrain_mm_r3.py --out mm_warmstart_r3 --steps 6000 --batch 4096
"""

import os, json, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.base_params import *
from envs.models import BoundContextNN, stack_policies
from utils import (make_frozen_noise, norm_gamma, bind_env, bind_policy,
                   make_price_fn_r3, new_policy, FIXED_SCENARIOS, CTX_NAMES,
                   CTX_REF, CONTEXT_DIM, _G_LO, _G_HI)
from helper_plot import (TYPE_COLORS, INK, INK_2, INK_MUTED, GRID, SURFACE,
                        ZERO_REF, LW_MIX, LW_BR, DASH_BR, _style, _save)

MM = 2
SEQ_BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal ramp

ap = argparse.ArgumentParser()
ap.add_argument('--out',       type=str,   default='mm_warmstart_r3')
ap.add_argument('--steps',     type=int,   default=6000)
ap.add_argument('--batch',     type=int,   default=4096)
ap.add_argument('--lr',        type=float, default=1e-3)
ap.add_argument('--buy-frac',  type=float, default=0.2)
ap.add_argument('--theta-max', type=float, default=0.45)
ap.add_argument('--T',         type=int,   default=50,
                help='horizon for the demo plot only')
ap.add_argument('--seed',      type=int,   default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


# ── the target policy (independent of gamma, by construction) ────────────────
def target_hat_f(t_n, th_n, buy_frac):
    sell = -jnp.minimum(th_n / jnp.maximum(1.0 - t_n, 1e-3), 1.0)
    return jnp.where(t_n <= buy_frac, jnp.ones_like(sell), sell)


def sample_batch(key, B, buy_frac, theta_max, env_ctx):
    """Synthetic states plus a scenario per sample. The state ranges are the
    Rung 1/2 ones; gamma is drawn from the prior box so the FiLM layers see the
    whole family during the fit."""
    k = jax.random.split(key, 7)
    t_n   = jax.random.uniform(k[0], (B,))
    th_n  = jax.random.uniform(k[1], (B,), minval=0.0, maxval=theta_max)
    A_n   = jax.random.uniform(k[2], (B,), minval=0.5, maxval=8.0)
    P_n   = jax.random.uniform(k[3], (B,), minval=0.2, maxval=2.5)
    Aa_n  = jax.random.uniform(k[4], (B,), minval=0.5, maxval=8.0)
    eps_n = jax.random.uniform(k[5], (B,), minval=0.0, maxval=0.05)
    z     = jnp.zeros((B,))
    S   = jnp.stack([t_n, th_n, z, z, eps_n, A_n, P_n, Aa_n], axis=1)
    # drawn from the environment's own prior, so the FiLM layers see exactly the
    # four-dimensional family they will later be trained against
    gam = env_ctx.generate_context(k[6], B)
    return S, gam, target_hat_f(t_n, th_n, buy_frac)


# ── the environment supplying the context prior ──────────────────────────────
env_ctx = ExogenousMarketEnvJAX(
    kappa=GAMMA_REF[1], T=args.T,
    generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
    market_impact_func=market_impact_base,
    generate_eps0_func=white_noise_A_base,
    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=100,
)

# ── fit ──────────────────────────────────────────────────────────────────────
model = new_policy(jax.random.PRNGKey(args.seed))
params, static = eqx.partition(model, eqx.is_array)
opt = optax.adam(args.lr)
st  = opt.init(params)


def loss_fn(p, key):
    S, gam, Y = sample_batch(key, args.batch, args.buy_frac, args.theta_max, env_ctx)
    net = eqx.combine(p, static).net
    pred = jax.vmap(lambda s, g: net(s, norm_gamma(g)))(S, gam)[:, 0]
    return jnp.mean((pred - Y) ** 2)


@jax.jit
def train(p, s, key):
    def one(carry, n):
        p_, s_ = carry
        l, g     = jax.value_and_grad(loss_fn)(p_, jax.random.fold_in(key, n))
        u, s_new = opt.update(g, s_, p_)
        return (eqx.apply_updates(p_, u), s_new), l
    (p_f, _), ls = jax.lax.scan(one, (p, s), jnp.arange(args.steps))
    return p_f, ls


print(f"fitting (FiLM, context_dim={CONTEXT_DIM}): {args.steps} steps, "
      f"batch {args.batch}, lr {args.lr}, buy_frac {args.buy_frac}", flush=True)
print("context ~ " + "  ".join(
    f"{n} U[{float(lo):.3g}, {float(hi):.3g}]"
    for n, lo, hi in zip(CTX_NAMES, _G_LO, _G_HI)), flush=True)
params, losses = train(params, st, jax.random.PRNGKey(args.seed + 1))
losses = np.asarray(losses)
model  = eqx.combine(params, static)
print(f"  mse {losses[0]:.5f} -> {losses[-1]:.7f}", flush=True)

path = os.path.join(args.out, 'mm_policy.eqx')
eqx.tree_serialise_leaves(path, model)
with open(os.path.join(args.out, 'meta.json'), 'w') as f:
    json.dump({'rung': 3, 'steps': args.steps, 'batch': args.batch,
               'lr': args.lr, 'buy_frac': args.buy_frac,
               'theta_max': args.theta_max, 'seed': args.seed,
               'mse_first': float(losses[0]), 'mse_last': float(losses[-1])},
              f, indent=2)
print(f"  saved -> {path}", flush=True)


# ── behaviour against a mixture of ONE random network, at three scenarios ────
env = ExogenousMarketEnvJAX(
    kappa=GAMMA_REF[1], T=args.T,
    generate_P_func=generate_prices_ou, A0=A0_BASE, P0=P0,
    market_impact_func=market_impact_base,
    generate_eps0_func=white_noise_A_base,
    generate_eps_idiosyncratic_func=idiosyncratic_noise_base,
    A_scale=A_SCALE_BASE, P_scale=P_SCALE_BASE, agent_per_policy=100,
)
bp, bq = list(PRIVATE_GENERATOR), list(BIG_PUBLIC_GENERATOR)
bp[8], bq[8] = TEC_PRIVATE, TEC_PUBLIC
env = eqx.tree_at(lambda e: e.agent_params, env, jnp.array([bp, bq, MARKET_MAKER]))
env = env.set_context(jnp.asarray(CTX_REF, dtype=jnp.float32))
noise = make_frozen_noise(env, args.T, 64)

rand = [new_policy(jax.random.PRNGKey(900 + j)) for j in range(N_TYPES)]
statics = tuple(stack_policies([rand[j]])[1] for j in range(N_TYPES))
stacked = tuple(stack_policies([rand[j]])[0] for j in range(N_TYPES))
price_fn = make_price_fn_r3(env, statics)

SCEN = list(FIXED_SCENARIOS)


def _scen_colors():
    """One ordinal colour per audit context (there are nine now, not three)."""
    cm = plt.get_cmap('viridis')
    return [cm(i / max(len(SCEN) - 1, 1)) for i in range(len(SCEN))]


idio  = noise['idio_eval'][MM]
scale = float(env.reward_scale(MM))
paths, runs = {}, {}
for name, g in SCEN:
    ga = jnp.asarray(g, dtype=jnp.float32)
    A = price_fn(stacked, ga[None, :], noise['P_traj'][None, :],
                 noise['eps0'][None, :], noise['idio_pop'])[0]
    env_g = bind_env(env, ga)
    warm = env_g.agent_batch_trajectory(bind_policy(model, ga), MM, A,
                                        noise['P_traj'], idio)
    ctrl = env_g.agent_batch_trajectory(bind_policy(rand[MM], ga), MM, A,
                                        noise['P_traj'], idio)
    Vw = float(np.asarray(warm['cum_r'])[-1] + warm['term_r'])
    Vc = float(np.asarray(ctrl['cum_r'])[-1] + ctrl['term_r'])
    paths[name], runs[name] = A, (warm, ctrl, Vw, Vc)
    print(f"  {name:<10s} A_0 {float(A[0]):.2f} -> A_T {float(A[-1]):.2f}   "
          f"warm V = {Vw:>10.1f} ({Vw/scale:+.1%})   "
          f"random V = {Vc:>10.1f} ({Vc/scale:+.1%})", flush=True)


# ── figure ───────────────────────────────────────────────────────────────────
C = TYPE_COLORS[MM]
ta, ts = np.arange(args.T), np.arange(args.T + 1)
fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.4), facecolor=SURFACE)
fig.suptitle("Rung 3 market-maker warm start — FiLM network, fitted once and "
             "checked at every audit context",
             color=INK, fontsize=12.5, x=0.006, ha="left", y=1.0)

ax = axes[0, 0]
for c, (name, _) in zip(_scen_colors(), SCEN):
    ax.plot(ts, np.asarray(paths[name]), color=c, lw=LW_MIX, label=name)
ax.axhline(float(env.Afloor), color=ZERO_REF, lw=1.0, ls=(0, (3, 3)))
_style(ax, "Allowance price it faces $A_t$", "price", "day")

ax = axes[0, 1]
for c, (name, _) in zip(_scen_colors(), SCEN):
    warm, ctrl, _, _ = runs[name]
    ax.plot(ta, np.asarray(warm['action'])[:, 0], color=c, lw=LW_MIX)
ax.plot(ta, np.asarray(runs['ref'][1]['action'])[:, 0],
        color=INK_MUTED, lw=LW_BR, ls=DASH_BR)
ax.axhline(0.0, color=ZERO_REF, lw=1.0)
ax.axvline(args.buy_frac * args.T, color=ZERO_REF, lw=1.0, ls=(0, (1, 3)))
ax.annotate(f"switch at {args.buy_frac:.0%}", (args.buy_frac * args.T, 0.97),
            xycoords=("data", "axes fraction"), color=INK_MUTED, fontsize=7.5,
            va="top", ha="left")
_style(ax, "Trade $f_t$  (dashed = random control)", "allowances / step", "day")

ax = axes[0, 2]
for c, (name, _) in zip(_scen_colors(), SCEN):
    warm, _, _, _ = runs[name]
    ax.plot(ts, np.asarray(warm['state'])[:, 1], color=c, lw=LW_MIX)
ax.axhline(0.0, color=ZERO_REF, lw=1.0)
_style(ax, "Inventory $\\theta_t$  ($\\theta_0 = 0$ at every $\\gamma$)",
       "allowances", "day")

ax = axes[1, 0]
for c, (name, _) in zip(_scen_colors(), SCEN):
    warm, ctrl, Vw, Vc = runs[name]
    ax.plot(ta, np.asarray(warm['cum_r']), color=c, lw=LW_MIX)
    ax.plot(ta, np.asarray(ctrl['cum_r']), color=INK_MUTED, lw=1.0, ls=DASH_BR)
ax.axhline(0.0, color=ZERO_REF, lw=1.0)
_style(ax, "Cumulative reward", "reward", "day")

# fit quality across the sampled space, at the two extreme scenarios
ax = axes[1, 1]
grid_t = np.linspace(0, 1, 200)
for c, th in zip(SEQ_BLUE, [0.05, 0.15, 0.30, 0.45]):
    S = jnp.stack([jnp.asarray(grid_t),
                   jnp.full(200, th), jnp.zeros(200), jnp.zeros(200),
                   jnp.zeros(200), jnp.full(200, 3.0), jnp.full(200, 1.0),
                   jnp.full(200, 3.0)], axis=1)
    for (name, g), ls in zip([SCEN[0], SCEN[2]], ['-', (0, (2, 2))]):
        ga = jnp.asarray(g, dtype=jnp.float32)
        pred = np.asarray(jax.vmap(bind_policy(model, ga))(S))[:, 0]
        ax.plot(grid_t, pred, color=c, lw=LW_MIX if ls == '-' else 1.2, ls=ls)
    tgt = np.asarray(target_hat_f(jnp.asarray(grid_t), jnp.full(200, th),
                                  args.buy_frac))
    ax.plot(grid_t, tgt, color=INK_MUTED, lw=1.0, ls=(0, (1, 2)))
ax.axhline(0.0, color=ZERO_REF, lw=1.0)
_style(ax, "Fitted $\\hat f$ at $\\gamma_{hi}$ (solid) vs $\\gamma_{lo}$ "
           "(dashed)\ngrey dotted = target", "$\\hat f$", "$t/T$")

ax = axes[1, 2]
ax.plot(np.arange(len(losses)), losses, color=INK, lw=LW_MIX)
ax.set_yscale('log')
_style(ax, "Regression MSE", "mse", "gradient step")

handles = (
    [Line2D([], [], color=c, lw=LW_MIX, label=name)
     for c, (name, _) in zip(SEQ_BLUE[1:], SCEN)]
    + [Line2D([], [], color=INK_MUTED, lw=LW_BR, ls=DASH_BR,
              label="random network (control)")]
    + [Line2D([], [], color=c, lw=LW_MIX, label=f"$\\theta_n$ = {th}")
       for c, th in zip(SEQ_BLUE, [0.05, 0.15, 0.30, 0.45])]
)
fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
           fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.055))

fig.tight_layout(rect=(0, 0.035, 1, 0.98))
out = _save(fig, os.path.join(args.out, 'mm_warmstart.pdf'))
print(f"  figure -> {out} (+ .png)", flush=True)
print("done", flush=True)
