"""Blind major (POLICY_SURFACES.md §4.1): the major's best response AS A PRICE TAKER
against the final equilibrium mixture of a rung-3 run.

    python3 blind_major.py --run-dir results_majmin/fast_r3_T30_K30_n100_block_seed42_bm1 \
        [--iters 4000 --batch 256 --bank 65536 --fresh]

Price bank (as utils_majmin_fast.build_bank) with the FINAL buffers, k = K + 1,
gamma ~ the whole box, major component u ~ U{0..k-1} per market; loss
-mean(rollout_major) / s0 with fresh idiosyncratic noise every step; cosine lr
1e-3 -> 1e-4; initialised from policy_{K}_type1.eqx (and, with --fresh, also from
a fresh major_policy). Check on the frozen 'prior' block: the blind major's
frozen-path value must be >= the mixture's up to one standard error.
Writes <run-dir>/counterfactual/blind_major{,_fresh}.{eqx,json}.
"""
import os
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
ap.add_argument('--iters', type=int, default=4000)
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--bank', type=int, default=65536)
ap.add_argument('--bank-chunk', type=int, default=8192)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--lr-min', type=float, default=1e-4)
ap.add_argument('--seed', type=int, default=4141)
ap.add_argument('--floor-max', type=float, default=30.0)
ap.add_argument('--clear-iters', type=int, default=None, help='default: the run\'s')
ap.add_argument('--fresh', action='store_true', help='also train from a fresh major_policy')
args = ap.parse_args()

from majmin_eval_common import read_progress, set_env_from_progress
prog = read_progress(args.run_dir)
set_env_from_progress(prog, args.floor_max)
os.environ['KETS_CLEAR_ITERS'] = str(args.clear_iters or prog.get('clear_iters', 40))

import json
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from envs.environment_majmin import MINORS, MAJOR
from envs.models_majmin import USE_H, major_policy, component
from utils import norm_gamma, VERIFY_SEED, _atomic_json
from utils_majmin import load_net, make_sim, draw_markets, COMP_SEED
from majmin_eval_common import make_env, load_mixture

assert prog['rung'] == 3, 'rung-3 runs only'
T, n = int(prog['T_grid'][0]), int(prog['n_sim'])
s0 = float(prog['s0'])
env = make_env(T)
mix = load_mixture(args.run_dir, prog)
K, k = mix['K'], mix['k']
sim = make_sim(env, mix['minor_static'], mix['maj_static'])
prior = lambda kk, b: env.generate_context(kk, b)
out_dir = os.path.join(args.run_dir, 'counterfactual')
os.makedirs(out_dir, exist_ok=True)
print(f"blind major | run {args.run_dir} | K={K} (k={k}) T={T} n={n} s0={s0:.4g} "
      f"floor_mode={prog['floor_mode']} USE_H={USE_H} devices {jax.devices()}", flush=True)


# ── price bank with the final mixture (no gradient) ──────────────────────────
chunk = min(args.bank_chunk, args.bank)


@jax.jit
def bank_chunk_fn(kk, minor_buf, maj_buf, k_):
    kn, ku = jax.random.split(kk)
    gam, P, eps0, imaj, imin, ck = draw_markets(env, kn, chunk, T, n, prior)
    u = jax.random.randint(ku, (chunk,), 0, k_)
    A, rho, H, _, _, _ = jax.vmap(
        lambda g, p, e, im, i0, i2, c, uu: sim(maj_buf, minor_buf, k_, g, p, e, im, (i0, i2), c, uu, False)
    )(gam, P, eps0, imaj, imin[0], imin[1], ck, u)
    return {'gam': gam, 'P': P, 'A': A, 'rho': rho, 'H': H}


t0 = time.time()
kbank = jax.random.PRNGKey(args.seed)
parts = [bank_chunk_fn(jax.random.fold_in(kbank, c), mix['minor_buf'], mix['maj_buf'], jnp.int32(k))
         for c in range(max(1, args.bank // chunk))]
bank = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *parts)
jax.block_until_ready(bank)
N = bank['A'].shape[0]
print(f"bank: {N} markets in {time.time() - t0:.0f}s  (A mean {float(bank['A'][:, 1:].mean()):.2f})",
      flush=True)


# ── training ─────────────────────────────────────────────────────────────────
_, maj_stat = eqx.partition(major_policy(jax.random.PRNGKey(0)), eqx.is_array)
sig_maj = env.agent_params[MAJOR, 7]


def returns(net, A, rho, H, P, imaj_raw, gam):
    """Frozen-path values of `net` (vmapped over markets). imaj_raw: type-1 sigma units."""
    def one(a, r, h, p, im, g):
        eg = env.bind(g)
        return eg.rollout_major(net, a, r, h, p, im * eg.major_mass(), norm_gamma(g))
    return jax.vmap(one)(A, rho, H, P, imaj_raw, gam)


opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.iters, alpha=args.lr_min / args.lr))


@jax.jit
def train(p, key):
    def one(carry, it):
        p_, s_ = carry
        ks, ki = jax.random.split(jax.random.fold_in(key, it))
        sel = jax.random.randint(ks, (args.batch,), 0, N)
        A, rho, H, P, gam = (bank['A'][sel], bank['rho'][sel], bank['H'][sel],
                             bank['P'][sel], bank['gam'][sel])
        im = jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, None, None))(
            jax.random.split(ki, args.batch), sig_maj, T)

        def loss(q):
            return -jnp.mean(returns(eqx.combine(q, maj_stat), A, rho, H, P, im, gam)) / s0
        l, g = jax.value_and_grad(loss)(p_)
        upd, s2 = opt.update(g, s_, p_)
        return (eqx.apply_updates(p_, upd), s2), l
    (pf, _), L = jax.lax.scan(one, (p, opt.init(p)), jnp.arange(args.iters))
    return pf, L


# ── frozen 'prior' block (as utils_majmin_fast), for the check ───────────────
Sb = int(prog['S'])
gamB, PB, eB, imB, iminB, _ = draw_markets(env, jax.random.fold_in(jax.random.PRNGKey(VERIFY_SEED), 0),
                                           Sb, T, n, prior)
iminB = tuple(x[0] for x in iminB)            # population noise shared across the block
ckB = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(COMP_SEED), k), Sb)
uB = jnp.arange(Sb) % k


@jax.jit
def block_sim(minor_buf, maj_buf):
    return jax.vmap(lambda g, p, e, im, c, uu: sim(maj_buf, minor_buf, jnp.int32(k), g, p, e, im,
                                                   iminB, c, uu, False))(gamB, PB, eB, imB, ckB, uB)


AB, rhoB, HB, r0B, _, _ = block_sim(mix['minor_buf'], mix['maj_buf'])


@jax.jit
def mix_frozen(maj_buf):
    def one(a, r, h, p, im, g, uu):
        eg = env.bind(g)
        net = component(maj_buf, mix['maj_static'], uu)
        return eg.rollout_major(net, a, r, h, p, im * eg.major_mass(), norm_gamma(g))
    return jax.vmap(one)(AB, rhoB, HB, PB, imB, gamB, uB)


V_mix = mix_frozen(mix['maj_buf'])
cons = float(jnp.max(jnp.abs(V_mix - r0B)) / jnp.maximum(jnp.max(jnp.abs(r0B)), 1.0))
print(f"check rollout_major vs cleared r0 on the prior block: max rel diff {cons:.2e}", flush=True)
assert cons < 1e-3, f"rollout_major does not reproduce the cleared r0 ({cons:.2e}): STOP"
blind_eval = eqx.filter_jit(lambda net: returns(net, AB, rhoB, HB, PB, imB, gamB))


def run(tag, init_net, key):
    t1 = time.time()
    p0 = eqx.partition(init_net, eqx.is_array)[0]
    V_init = blind_eval(init_net)
    pf, L = train(p0, key)
    L = np.asarray(L)
    net = eqx.combine(pf, maj_stat)
    V_b = blind_eval(net)
    d = np.asarray(V_b - V_mix)
    m5 = max(1, len(L) // 20)
    res = {'tag': tag, 'loss_first5': float(L[:m5].mean()), 'loss_last5': float(L[-m5:].mean()),
           'seconds': time.time() - t1, 'iters': args.iters, 'batch': args.batch, 'bank': N,
           'lr': args.lr, 'lr_min': args.lr_min, 'seed': args.seed, 'K': K, 'k': k,
           'init': 'policy_%d_type1' % K if tag == 'blind_major' else 'fresh major_policy',
           'check_prior_block': {
               'S': Sb, 'V_mix_frozen': float(V_mix.mean()), 'V_blind_frozen': float(V_b.mean()),
               'V_init_frozen': float(V_init.mean()),
               'diff_mean': float(d.mean()), 'diff_se': float(d.std(ddof=1) / np.sqrt(Sb)),
               'diff_relative': float(d.mean()) / s0, 'r0_rollout_max_rel_diff': cons,
               'passed': bool(d.mean() >= -d.std(ddof=1) / np.sqrt(Sb))}}
    eqx.tree_serialise_leaves(os.path.join(out_dir, tag + '.eqx'), net)
    _atomic_json(os.path.join(out_dir, tag + '.json'), res)
    c = res['check_prior_block']
    print(f"[{tag}] loss {res['loss_first5']:+.4f} -> {res['loss_last5']:+.4f} in {res['seconds']:.0f}s | "
          f"frozen prior block: blind {c['V_blind_frozen']:.4g} vs mixture {c['V_mix_frozen']:.4g}: "
          f"diff {c['diff_mean']:+.4g} +/- {c['diff_se']:.3g} ({c['diff_relative']:+.3%} of s0)  "
          f"{'OK' if c['passed'] else 'FAILED: blind major below the mixture'}", flush=True)
    return res


r = run('blind_major', load_net(args.run_dir, K, MAJOR), jax.random.PRNGKey(args.seed + 1))
if args.fresh:
    rf = run('blind_major_fresh', major_policy(jax.random.PRNGKey(args.seed + 2)),
             jax.random.PRNGKey(args.seed + 3))
    print(f"warm vs fresh blind major on the prior block: "
          f"{r['check_prior_block']['V_blind_frozen']:.4g} vs {rf['check_prior_block']['V_blind_frozen']:.4g}",
          flush=True)
print(f"-> {out_dir}", flush=True)
