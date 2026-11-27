"""Policy surfaces of the major–minor market: simulation.

    python3 policy_surfaces.py --run-dirs <seed run 1> ... <seed run L> --out <study>/surfaces \
        [--S 512 --res 13 --floor-max 30 --chunk 1024]

One common-noise draw (NOISE_KEY) is reused at every grid point, for every seed and
for the equilibrium and blind-major markets of Fig. 3. Per grid point, seed and
market the raw traded-price arrays (ys['A'], days 0..T-1; never the pre-open
A_traj[0]) and the floor diagnostics are stored; metrics and aggregations are in
plot_policy_surfaces.py. Writes fig1.npz (init_alloc x kappa), fig2.npz and
fig3.npz (init_alloc x Afloor, one simulation), fig4.npz (mm_share x Afloor),
checks.json.
"""
import os
import argparse
import hashlib
import time

ap = argparse.ArgumentParser()
ap.add_argument('--run-dirs', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--S', type=int, default=512)
ap.add_argument('--res', type=int, default=13)
ap.add_argument('--chunk', type=int, default=1024, help='markets per vmapped call')
ap.add_argument('--floor-max', type=float, default=30.0)
ap.add_argument('--noise-seed', type=int, default=5150)
ap.add_argument('--figs', type=int, nargs='+', default=[1, 2, 4], help='2 also writes 3')
ap.add_argument('--no-blind', action='store_true', help='skip the blind-major markets (no Fig. 3)')
ap.add_argument('--fixed', type=float, nargs=4, default=None,
                help='fixed coordinates (init_alloc kappa Afloor mm_share); default CTX_REF')
ap.add_argument('--range', nargs=3, action='append', default=[], metavar=('AXIS', 'LO', 'HI'),
                help='grid axis range, e.g. --range Afloor 0 15 --range init_alloc 0.5 1.0')
args = ap.parse_args()

from majmin_eval_common import read_progress, set_env_from_progress
progs = [read_progress(d) for d in args.run_dirs]
p0 = progs[0]
for p in progs:
    assert p['rung'] == 3, 'rung-3 runs only'
    assert p['T_grid'] == p0['T_grid'] and p['n_sim'] == p0['n_sim'], 'all runs: same T and n'
    assert p['floor_mode'] == p0['floor_mode'] and p.get('public_history_input') == p0.get('public_history_input')
assert p0['floor_mode'] == 'block', "floor_mode must be 'block' (rationing) for Fig. 2"
set_env_from_progress(p0, args.floor_max)
os.environ['KETS_CLEAR_ITERS'] = str(p0.get('clear_iters', 40))

import json
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from envs.base_params import (ALLOC_MIN, ALLOC_MAX, FLOOR_MIN, FLOOR_MAX, MM_MIN, MM_MAX,
                               KAPPA_MIN, KAPPA_MAX)
from envs.models_majmin import major_policy
from utils import CTX_REF
from utils_majmin import make_sim, draw_markets
from majmin_eval_common import make_env, load_mixture

assert abs(FLOOR_MAX - args.floor_max) < 1e-9
T, n, S = int(p0['T_grid'][0]), int(p0['n_sim']), args.S
L = len(args.run_dirs)
env = make_env(T)
fixed = tuple(args.fixed) if args.fixed else tuple(CTX_REF)
os.makedirs(args.out, exist_ok=True)
print(f"policy surfaces | {L} run(s) | T={T} n={n} S={S} res={args.res} | fixed {fixed} | "
      f"floor max {FLOOR_MAX} | devices {jax.devices()}", flush=True)

mixes = [load_mixture(d, p) for d, p in zip(args.run_dirs, progs)]
sim = make_sim(env, mixes[0]['minor_static'], mixes[0]['maj_static'])   # same architecture for all seeds
blind = [None] * L
if not args.no_blind:
    like = major_policy(jax.random.PRNGKey(0))
    for l, d in enumerate(args.run_dirs):
        f = os.path.join(d, 'counterfactual', 'blind_major.eqx')
        if not os.path.exists(f):
            raise SystemExit(f"{f} missing: run blind_major.py --run-dir {d} first (or --no-blind)")
        blind[l] = eqx.tree_deserialise_leaves(f, like)

# ── the common noise: drawn ONCE (§2); gamma discarded ───────────────────────
NOISE_KEY = jax.random.PRNGKey(args.noise_seed + 7)
_, P, eps0, imaj, imin, ck = draw_markets(env, NOISE_KEY, S, T, n,
                                          lambda kk, b: jnp.broadcast_to(jnp.asarray(fixed, jnp.float32), (b, 4)))
NOISE = dict(P=P, eps0=eps0, imaj=imaj, imin0=imin[0], imin1=imin[1], ck=ck)


def noise_hash():
    h = hashlib.sha256()
    for kk in sorted(NOISE):
        h.update(np.asarray(jax.device_get(NOISE[kk])).tobytes())
    return h.hexdigest()


H0 = noise_hash()
print(f"noise hash {H0[:16]}", flush=True)


def _outs(r, gam):
    A, rho, H, r0, ys, (Xf, X0f) = r
    # year-end short indicator per type (minors: share of agents; major: 0/1), and the
    # mass-weighted probability of being short p = sum_j w_j(g) short_j
    short_t = jax.vmap(lambda g, xf0, xf2, x0: env.bind(g).minor_gaps((xf0, xf2), x0)[1])(
        gam, Xf[0], Xf[1], X0f)
    w = jax.vmap(lambda g: env.bind(g).context_weights())(gam)
    return {'A': ys['A'], 'on_floor': ys['on_floor'], 'on_ceil': ys['on_ceil'], 'G': ys['G'], 'O': ys['O'],
            'r0': r0, 'short_t': short_t, 'p_short': jnp.sum(w * short_t, axis=-1),
            'resid': jnp.max(ys['resid'], axis=1), 'cons': jnp.max(jnp.abs(ys['cons_err']), axis=1)}


@jax.jit
def sim_mix(maj_buf, minor_buf, k, gam, si):
    u = si % k
    r = jax.vmap(lambda g, s, uu: sim(maj_buf, minor_buf, k, g, NOISE['P'][s], NOISE['eps0'][s],
                                      NOISE['imaj'][s], (NOISE['imin0'][s], NOISE['imin1'][s]),
                                      NOISE['ck'][s], uu, False))(gam, si, u)
    o = _outs(r, gam)
    o['cap'] = jax.vmap(lambda g: env.bind(g).cap_total())(gam)
    return o


@eqx.filter_jit
def sim_blind(net, minor_buf, k, gam, si):
    r = jax.vmap(lambda g, s: sim(net, minor_buf, k, g, NOISE['P'][s], NOISE['eps0'][s],
                                  NOISE['imaj'][s], (NOISE['imin0'][s], NOISE['imin1'][s]),
                                  NOISE['ck'][s], 0, True))(gam, si)
    o = _outs(r, gam)
    o['cap'] = jax.vmap(lambda g: env.bind(g).cap_total())(gam)
    return o


def run_grid(gpts, l, use_blind=False):
    """gpts (G, 4) -> dict of arrays (G, S, ...) for seed l."""
    G = gpts.shape[0]
    gam = np.repeat(gpts, S, axis=0).astype(np.float32)          # (G*S, 4)
    si = np.tile(np.arange(S), G).astype(np.int32)                # market s = same world everywhere
    N, ch = gam.shape[0], args.chunk
    pad = (-N) % ch
    gam = np.concatenate([gam, np.repeat(gam[-1:], pad, 0)]); si = np.concatenate([si, np.zeros(pad, np.int32)])
    m = mixes[l]
    kk = jnp.int32(m['k'])
    res = []
    for a in range(0, N + pad, ch):
        g_, s_ = jnp.asarray(gam[a:a + ch]), jnp.asarray(si[a:a + ch])
        o = sim_blind(blind[l], m['minor_buf'], kk, g_, s_) if use_blind else \
            sim_mix(m['maj_buf'], m['minor_buf'], kk, g_, s_)
        res.append(jax.device_get(o))
    out = {kk_: np.concatenate([r[kk_] for r in res])[:N] for kk_ in res[0]}
    return {kk_: v.reshape((G, S) + v.shape[1:]) for kk_, v in out.items()}


def grid_points(xname, xs, yname, ys_):
    names = ('init_alloc', 'kappa', 'Afloor', 'mm_share')
    pts = []
    for x in xs:
        for y in ys_:
            g = list(fixed)
            g[names.index(xname)] = x
            g[names.index(yname)] = y
            pts.append(g)
    return np.asarray(pts, np.float32)


R = args.res
AX = {'init_alloc': np.linspace(ALLOC_MIN, ALLOC_MAX, R),
      'kappa': np.linspace(1.0, 6.0, R),
      'Afloor': np.linspace(FLOOR_MIN, FLOOR_MAX, R),
      'mm_share': np.linspace(MM_MIN, MM_MAX, R)}
BOX = {'init_alloc': (ALLOC_MIN, ALLOC_MAX), 'kappa': (KAPPA_MIN, KAPPA_MAX),
       'Afloor': (FLOOR_MIN, FLOOR_MAX), 'mm_share': (MM_MIN, MM_MAX)}
outside = {}
for nm, lo, hi in args.range:
    assert nm in AX, f'unknown axis {nm}'
    AX[nm] = np.linspace(float(lo), float(hi), R)
for nm, v in AX.items():
    if v.min() < BOX[nm][0] - 1e-9 or v.max() > BOX[nm][1] + 1e-9:
        outside[nm] = [float(v.min()), float(v.max())]
        print(f"  WARNING: {nm} grid [{v.min():g}, {v.max():g}] leaves the training box {BOX[nm]}: "
              f"the networks extrapolate there (FiLM input outside [-1, 1])", flush=True)
checks = {'outside_box': outside, 'noise_hash': H0, 'S': S, 'T': T, 'n': n, 'seeds': args.run_dirs, 'fixed': list(fixed),
          'clearing': {}}

KEEP = {1: ('A', 'on_floor', 'on_ceil', 'p_short', 'short_t'),
        2: ('A', 'on_floor', 'on_ceil', 'G', 'O', 'r0', 'p_short', 'short_t'), 4: ('A', 'on_floor')}
SPEC = {1: ('init_alloc', 'kappa'), 2: ('init_alloc', 'Afloor'), 4: ('mm_share', 'Afloor')}


def clearing_check(tag, o, gp):
    rr, cc = o['resid'] / o['cap'], o['cons'] / o['cap']          # (G, S)
    wr, wc = int(np.argmax(rr.max(axis=1))), int(np.argmax(cc.max(axis=1)))
    d = {'resid_max_rel': float(rr.max()), 'resid_worst_g': gp[wr].tolist(),
         'cons_max_rel': float(cc.max()), 'cons_worst_g': gp[wc].tolist()}
    print(f"  [{tag}] clearing: max resid/cap {d['resid_max_rel']:.1e} at g={np.round(gp[wr], 3).tolist()}  "
          f"max |cons_err|/cap {d['cons_max_rel']:.1e} at g={np.round(gp[wc], 3).tolist()}", flush=True)
    assert d['resid_max_rel'] < 1e-3 and d['cons_max_rel'] < 1e-3, f"clearing quality at {tag}"
    return d


for fig in args.figs:
    xn, yn = SPEC[fig]
    xs, ys_ = AX[xn], AX[yn]
    gp = grid_points(xn, xs, yn, ys_)
    t0 = time.time()
    data = {kk: [] for kk in KEEP[fig]}
    blind_data = {'A': [], 'r0': []}
    for l in range(L):
        o = run_grid(gp, l)
        checks['clearing'][f'fig{fig}_seed{l}'] = clearing_check(f'fig{fig} seed {l}', o, gp)
        assert np.all(o['A'] >= 0) and np.isfinite(o['A']).all()
        nz = int((o['A'] <= 0).any(-1).sum())
        if nz:
            print(f"  [fig{fig} seed {l}] {nz} market(s) trade at a zero price (floor 0)", flush=True)
        for kk in KEEP[fig]:
            data[kk].append(o[kk].reshape((R, R) + o[kk].shape[1:]))
        if fig == 2 and not args.no_blind:
            ob = run_grid(gp, l, use_blind=True)
            checks['clearing'][f'fig3_blind_seed{l}'] = clearing_check(f'fig3 blind seed {l}', ob, gp)
            for kk in ('A', 'r0'):
                blind_data[kk].append(ob[kk].reshape((R, R) + ob[kk].shape[1:]))
    assert noise_hash() == H0, 'noise arrays changed'
    base = dict(x=xs, y=ys_, xname=xn, yname=yn, fixed=np.asarray(fixed), seeds=np.asarray(args.run_dirs),
                S=S, T=T, noise_hash=H0)
    arr = {kk: np.stack(v).astype(np.float32) for kk, v in data.items()}     # (L, nx, ny, S, ...)
    if fig == 2:
        np.savez_compressed(os.path.join(args.out, 'fig2.npz'), **base,
                            **{kk: arr[kk] for kk in ('A', 'on_floor', 'on_ceil', 'G', 'O', 'p_short', 'short_t')})
        if not args.no_blind:
            np.savez_compressed(os.path.join(args.out, 'fig3.npz'), **base, A=arr['A'], r0=arr['r0'],
                                A_blind=np.stack(blind_data['A']).astype(np.float32),
                                r0_blind=np.stack(blind_data['r0']).astype(np.float32))
    else:
        np.savez_compressed(os.path.join(args.out, f'fig{fig}.npz'), **base, **arr)
    print(f"fig{fig}{' + fig3' if fig == 2 and not args.no_blind else ''}: {xn} x {yn}, "
          f"{len(gp)} points x {S} markets x {L} seed(s) in {time.time() - t0:.0f}s  "
          f"(A mean {arr['A'].mean():.2f})", flush=True)

# ── §7: A-bar at CTX_REF vs plot_majmin_scenarios ────────────────────────────
gref = np.asarray([CTX_REF], np.float32)
ref = []
for l in range(L):
    o = run_grid(gref, l)
    Ab = o['A'][0].mean(axis=1)
    sj = os.path.join(args.run_dirs[l], 'scenarios', 'ref', 'summary.json')
    other = json.load(open(sj))['A_mean'] if os.path.exists(sj) else None
    ref.append({'A_bar': float(Ab.mean()), 'A_bar_se': float(Ab.std(ddof=1) / np.sqrt(S)),
                'scenarios_ref_A_mean': other})
    print(f"  CTX_REF seed {l}: A-bar {Ab.mean():.2f} +/- {Ab.std(ddof=1) / np.sqrt(S):.2f} (S={S})  "
          f"vs plot_majmin_scenarios A_mean {other}", flush=True)
checks['ctx_ref'] = ref
with open(os.path.join(args.out, 'checks.json'), 'w') as f:
    json.dump(checks, f, indent=1)
print(f"-> {args.out}", flush=True)
