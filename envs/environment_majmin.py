"""Major–minor market clearing (MAJOR_MINOR_CLEARING.md), on top of the v2
market-clearing environment (states, rewards, dynamics, scenario-scaled supply
shock sigma_S(gamma) = c Theta(gamma)/sqrt(365), corridor [floor, A_CEIL_V2]).

Players
  minors : types j = 0 (private producers) and j = 2 (financial institutions),
           n simulated agents each, weights w_j = context_weights()[j];
  major  : N0 = 1 aggregate player for type 1 (large public producers), mass
           m0 = w_1, aggregate quantities (theta, C^f, C^xi, sigma_eps x m0), enters
           the clearing with weight 1.

Clearing (§4): D(a) = f0(a) + sum_{j in {0,2}} w_j mean_i f_ji(a) - S_t, two
bisections for [a-, a+], A_t = clip(A_pre, a-, a+) (day 0: A_0 = a-, the root
closest to the floor; the pre-open A_REF only centres the schedules); the long side is rationed
at a binding bound. A_t carries a custom JVP (implicit function theorem, §7):
  interior root : dA = -dD(A_t)/D'(A_t), D' guarded at -eps_D;
  bound binds   : dA = 0;   selection keeps A_pre in a flat interval : dA = dA_pre.
The fill ratios are explicit functions of the orders at A_t, so plain autodiff
differentiates them.

Public history H_t (§2), pre-trade, seen by every player:
  (EMA of price changes / A_scale, last sellers' fill ratio, EMA of executed volume / cap)
with EMA rate lambda = 1 - exp(-dt / EMA_TAU_DAYS) (horizon-invariant).
"""
import jax
import jax.numpy as jnp
from jax import vmap

from envs.environment_v2 import MarketClearingEnvV2, stratified_components
from envs.params_v2 import A_REF, A_CEIL_V2, CLEAR_ITERS_V2, FLOOR_MODE, SUPPLY_C
from envs.base_params import LAYOUT, CTX_REF_V5
from envs.models_majmin import gather_apply, H_DIM, USE_H

MINORS = (0, 2)
MAJOR = 1
EMA_TAU_DAYS = 30.0
EPS_D_REL = 1e-3        # guard on D'(A_t), relative to sum_j w_j C^f_j / A_scale
FLAT_TOL = 1e-4         # USD


# ── the clearing as a function of explicit schedule parameters ───────────────

def _order(a, p, sc):
    """p = (alpha, beta, lo, hi, cap, apre) (scalars or arrays) -> order at price a."""
    al, be, lo, hi, cap, ap = p
    return jnp.clip(cap * jnp.tanh(al - be * (a - ap) / sc), lo, hi)


def _order_slope(a, p, sc):
    al, be, lo, hi, cap, ap = p
    z = al - be * (a - ap) / sc
    raw = cap * jnp.tanh(z)
    inside = (raw > lo) & (raw < hi)
    return jnp.where(inside, -cap * be / sc / jnp.cosh(z) ** 2, 0.0)


def _D(a, maj, mins, wm, S, sc):
    d = _order(a, maj, sc)
    for p, w in zip(mins, wm):
        d = d + w * jnp.mean(_order(a, p, sc))
    return d - S


def _Dprime(a, maj, mins, wm, sc):
    d = _order_slope(a, maj, sc)
    for p, w in zip(mins, wm):
        d = d + w * jnp.mean(_order_slope(a, p, sc))
    return d


def _bisect(maj, mins, wm, S, apre, fl, ce, sc):
    D = lambda a: _D(a, maj, mins, wm, S, sc)
    d_lo, d_hi = D(fl), D(ce)

    def left(_, lh):
        l, h = lh
        m = 0.5 * (l + h)
        pos = D(m) > 0
        return jnp.where(pos, m, l), jnp.where(pos, h, m)

    def right(_, lh):
        l, h = lh
        m = 0.5 * (l + h)
        neg = D(m) < 0
        return jnp.where(neg, l, m), jnp.where(neg, m, h)

    am = jnp.where(d_lo <= 0, fl, jnp.where(d_hi > 0, ce,
                   jax.lax.fori_loop(0, CLEAR_ITERS_V2, left, (fl, ce))[1]))
    ap = jnp.where(d_hi >= 0, ce, jnp.where(d_lo < 0, fl,
                   jax.lax.fori_loop(0, CLEAR_ITERS_V2, right, (fl, ce))[0]))
    return jnp.clip(apre, am, ap), am, ap, d_lo < 0, d_hi > 0


@jax.custom_jvp
def clear_price(maj, mins, wm, S, apre, fl, ce, sc, eps_d):
    """Cleared price A_t, differentiable by the implicit function theorem."""
    return _bisect(maj, mins, wm, S, apre, fl, ce, sc)[0]


@clear_price.defjvp
def _clear_price_jvp(primals, tangents):
    maj, mins, wm, S, apre, fl, ce, sc, eps_d = primals
    dmaj, dmins, dwm, dS, dapre = tangents[:5]
    A, am, ap, fb, cb = _bisect(maj, mins, wm, S, apre, fl, ce, sc)
    A = jax.lax.stop_gradient(A)
    _, dD = jax.jvp(lambda m, mi, w, s: _D(A, m, mi, w, s, sc),
                    (maj, mins, wm, S), (dmaj, dmins, dwm, dS))
    Dp = jnp.minimum(_Dprime(A, maj, mins, wm, sc), -eps_d)
    keep = ((ap - am) > FLAT_TOL) & (A > am) & (A < ap)
    dA = jnp.where(fb | cb, 0.0, jnp.where(keep, dapre, -dD / Dp))
    return A, dA


def clearing_info(maj, mins, wm, S, apre, fl, ce, sc, eps_d):
    """Non-differentiated side information of the clearing (diagnostics)."""
    A, am, ap, fb, cb = _bisect(*jax.lax.stop_gradient((maj, mins, wm, S, apre)), fl, ce, sc)
    Dp_raw = _Dprime(A, maj, mins, wm, sc)
    keep = ((ap - am) > FLAT_TOL) & (A > am) & (A < ap)
    moving = ~(fb | cb | keep)
    return {'a_minus': am, 'a_plus': ap, 'floor_binds': fb, 'ceil_binds': cb, 'keep': keep,
            'moving': moving, 'guard': moving & (Dp_raw > -eps_d),
            'impact': jnp.where(moving, -1.0 / jnp.minimum(Dp_raw, -eps_d), 0.0)}


class MajorMinorEnv(MarketClearingEnvV2):
    """v2 env + one strategic major (type 1 as an aggregate player)."""

    def bind(self, g):
        """The same environment under scenario g: v4 (init_alloc, kappa, Afloor, financial_inst);
        v5 (c, s, Afloor, mm_share, kappa2), c kept in the init_alloc slot (RUNS_SETTING.md)."""
        import equinox as eqx
        if LAYOUT == 'v5':
            return eqx.tree_at(lambda e: (e.init_alloc, e.major_share, e.Afloor, e.mm_share, e.kappa),
                               self, (g[0], g[1], g[2], g[3], g[4]))
        return eqx.tree_at(lambda e: (e.init_alloc, e.kappa, e.Afloor, e.mm_share),
                           self, (g[0], g[1], g[2], g[3]))

    def sigma_S(self):
        """v5: the exogenous supply is FIXED at gamma_ref (its scale does not move with c, s
        or mm_share); v4: scaled by the scenario's total allocation (environment_v2)."""
        if LAYOUT == 'v5':
            return SUPPLY_C * self.bind(jnp.asarray(CTX_REF_V5, jnp.float32)).total_allocation() / jnp.sqrt(365.0)
        return super().sigma_S()

    def major_mass(self):
        return self.context_weights()[MAJOR]

    def major_initial_state(self):
        """(16,) aggregate state of the major: type-1 holdings and capacities x m0."""
        idx = self.dic_indexes
        x = self.type_initial_state(MAJOR)
        m0 = self.major_mass()
        for k in ('theta', 'cap_f', 'cap_xi', 'sigma_eps'):
            x = x.at[idx[k]].multiply(m0)
        return x

    def major_reward_scale(self, m0):
        """Fixed scale of the major's reward: the type-1 scale x m0 (linear in caps)."""
        return jnp.maximum(m0 * self.reward_scale(MAJOR), 1.0)

    def cap_total(self):
        w = self.context_weights()
        return jnp.sum(w * self.agent_params[:, 5])

    def eps_d(self):
        return EPS_D_REL * self.cap_total() / self.A_scale

    def ema_rate(self):
        return 1.0 - jnp.exp(-self.dt / EMA_TAU_DAYS)

    def net_input(self, x, H):
        """Network input: normalised own state (private + public: t, A_pre, A_avg, P),
        plus the public-history summary H only if KETS_USE_H=1."""
        if not USE_H:
            return self.normalize_state(x)
        return jnp.concatenate([self.normalize_state(x), H])

    def sched(self, x, out):
        """-> schedule parameters (alpha, beta, lo, hi, cap, apre), xi, eta."""
        idx = self.dic_indexes
        al, be, lo, hi, xi, eta = self.schedule_params(x, out)
        return (al, be, lo, hi, x[idx['cap_f']], x[idx['A']]), xi, eta

    # ── one day of one market ────────────────────────────────────────────────

    def market_step(self, carry, t, major_fn, minor_buf, minor_static, comp, ctx, P_traj,
                    eps0, idio_min, idio_maj, myopic):
        """carry = (Xmin tuple of (n,16), X0 (16,), H (3,), r0 scalar).
        major_fn(inp (11,), ctx) -> (4,). Returns new carry and per-day outputs."""
        Xmin, X0, H, r0 = carry
        idx = self.dic_indexes
        w = self.context_weights()
        wm = tuple(w[j] for j in MINORS)
        sc = self.A_scale
        Hs = jax.lax.stop_gradient(H) if myopic else H

        # forward passes on the pre-trade states
        out0 = major_fn(self.net_input(X0, Hs), ctx)
        p0, xi0, eta0 = self.sched(X0, out0)
        outs, pm, xim, etam = [], [], [], []
        for a, j in enumerate(MINORS):
            Xj = jax.lax.stop_gradient(Xmin[a]) if myopic else Xmin[a]
            inp = vmap(self.net_input, in_axes=(0, None))(Xj, Hs)
            o = gather_apply(minor_buf[a], minor_static[a], comp[a], inp, ctx)
            if myopic:
                o = jax.lax.stop_gradient(o)
            p, xi, eta = vmap(self.sched)(Xj, o)
            outs.append(o); pm.append(p); xim.append(xi); etam.append(eta)
        pm = tuple(pm)

        # clearing
        S = self.supply(eps0[t])
        apre = X0[idx['A']]
        fl = jnp.asarray(self.Afloor, jnp.float32)
        ce = jnp.asarray(A_CEIL_V2, jnp.float32)
        eps_d = self.eps_d()
        # selection point: the last traded price; on day 0 there is none (the
        # pre-open A_REF only centres the schedules), so the price is settled by
        # the clearing alone and, if the root set is an interval, its lowest
        # point a- (the root closest to the floor) is taken
        sel = jnp.where(t == 0, fl, apre)
        A_t = clear_price(p0, pm, wm, S, sel, fl, ce, sc, eps_d)
        info = clearing_info(p0, pm, wm, S, sel, fl, ce, sc, eps_d)

        # rationing (explicit in the orders at A_t)
        f0 = _order(A_t, p0, sc)
        fm = [_order(A_t, p, sc) for p in pm]
        B = jnp.maximum(f0, 0.0) + sum(w_ * jnp.mean(jnp.maximum(f, 0.0)) for w_, f in zip(wm, fm)) \
            + jnp.maximum(-S, 0.0)
        O = jnp.maximum(-f0, 0.0) + sum(w_ * jnp.mean(jnp.maximum(-f, 0.0)) for w_, f in zip(wm, fm)) \
            + jnp.maximum(S, 0.0)
        fb, cb = info['floor_binds'], info['ceil_binds']
        rho_s = jnp.where(fb, B / jnp.maximum(O, 1e-12), 1.0)
        rho_b = jnp.where(cb, O / jnp.maximum(B, 1e-12), 1.0)
        G = jnp.where(fb | cb, O - B, 0.0)
        if FLOOR_MODE == 'support':
            rho_s, rho_b = jnp.ones_like(rho_s), jnp.ones_like(rho_b)
        ex = lambda f: jnp.where(f >= 0.0, rho_b * f, rho_s * f)

        # execution, rewards, dynamics
        a0 = jnp.stack([ex(f0), xi0, eta0])
        r0 = r0 + self.running_reward(X0, a0, A_t)
        X0n = self.single_step_dynamics(X0, a0, idio_maj[t], A_t, P_traj[t + 1])
        Xn, acts = [], []
        for a in range(len(MINORS)):
            act = jnp.stack([ex(fm[a]), xim[a], etam[a]], axis=-1)
            acts.append(act)
            Xn.append(vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                Xmin[a], act, idio_min[a][:, t], A_t, P_traj[t + 1]))

        # public history, for tomorrow
        lam = self.ema_rate()
        vol = rho_b * B
        dA = jnp.where(t == 0, 0.0, A_t - apre)          # no price change on day 0
        Hn = jnp.stack([(1 - lam) * H[0] + lam * dA / sc, rho_s,
                        (1 - lam) * H[2] + lam * vol / self.cap_total()])

        S_exec = jnp.where(S > 0, rho_s * S, rho_b * S)
        dth = (X0n[idx['theta']] - X0[idx['theta']]) + sum(
            w_ * jnp.mean(Xn[a][:, idx['theta']] - Xmin[a][:, idx['theta']])
            for a, w_ in enumerate(wm))
        G_gov = G if FLOOR_MODE == 'support' else 0.0
        D_At = _D(jax.lax.stop_gradient(A_t), *jax.lax.stop_gradient((p0, pm, wm, S)), sc)
        y = {
            'A': A_t, 'rho': jnp.stack([rho_s, rho_b]), 'H': H,
            'on_floor': fb, 'on_ceil': cb, 'flat': (info['a_plus'] - info['a_minus']) > FLAT_TOL,
            'G': G, 'B': B, 'O': O, 'S': S, 'S_exec': S_exec, 'cons_err': dth - (S_exec - G_gov),
            'resid': jnp.where(fb | cb, 0.0, jnp.abs(D_At)),
            'impact': info['impact'], 'moving': info['moving'], 'guard': info['guard'],
            'X0': X0, 'a0': a0,
            'f0': a0[0], 'R0': apre + sc * out0[0] / (out0[1] + 1e-6),
            'beta0': out0[1],
            'f_minor': jnp.stack([jnp.mean(acts[a][:, 0]) for a in range(len(MINORS))]),
            # per minor type, mean over its particles: post-trade holdings and production
            # (fossil, gas) of the day -- trajectories of the LP benchmark (plot_benchmark.py)
            'theta_minor': jnp.stack([jnp.mean(Xn[a][:, idx['theta']]) for a in range(len(MINORS))]),
            'q_minor': jnp.stack([jnp.stack([jnp.mean(acts[a][:, 1] * acts[a][:, 2]),
                                             jnp.mean(acts[a][:, 1] * (1.0 - acts[a][:, 2]))])
                                  for a in range(len(MINORS))]),
        }
        return (tuple(Xn), X0n, Hn, r0), y

    def simulate(self, major_fn, minor_buf, minor_static, k, comp_key, ctx, P_traj, eps0,
                 idio_min, idio_maj, myopic=False, checkpoint=False):
        """One market for one year. idio_min: tuple of (n, T) for the minor types;
        idio_maj: (T,) (already in aggregate units). Returns
        A_traj (T+1,), rho_traj (T+1, 2), H_traj (T, 3) (pre-trade H of each day),
        the major's total reward, per-day outputs, final states."""
        T_int = int(self.T)
        n = idio_min[0].shape[0]
        Xmin0 = tuple(jnp.broadcast_to(self.type_initial_state(j), (n, 16)) for j in MINORS)
        X00 = self.major_initial_state()
        H0 = jnp.array([0.0, 1.0, 0.0], jnp.float32)
        if comp_key is None:
            comp = tuple(stratified_components(None, k, n) for _ in MINORS)
        else:
            ks = jax.random.split(comp_key, len(MINORS))
            comp = tuple(stratified_components(ks[a], k, n) for a in range(len(MINORS)))

        def step(carry, t):
            return self.market_step(carry, t, major_fn, minor_buf, minor_static, comp, ctx,
                                    P_traj, eps0, idio_min, idio_maj, myopic)
        if checkpoint:
            step = jax.checkpoint(step)
        (Xf, X0f, _, r0), ys = jax.lax.scan(step, (Xmin0, X00, H0, jnp.float32(0.0)),
                                            jnp.arange(T_int))
        r0 = r0 + self.terminal_reward(X0f)
        A_traj = jnp.concatenate([jnp.atleast_1d(jnp.float32(A_REF)), ys['A']])
        rho_traj = jnp.concatenate([jnp.ones((1, 2), jnp.float32), ys['rho']], axis=0)
        return A_traj, rho_traj, ys['H'], r0, ys, (Xf, X0f)

    def minor_gaps(self, Xf, X0f):
        """Year-end E - theta: aggregate (weighted, major included) and per minor type."""
        idx = self.dic_indexes
        w = self.context_weights()
        gap = lambda X: (jnp.sum(X[..., idx['xi']] * X[..., idx['e']], axis=-1)
                         + X[..., idx['eps']] - X[..., idx['theta']])
        g0 = gap(X0f)
        gm = [gap(X) for X in Xf]
        agg = g0 + sum(w[j] * jnp.mean(g) for j, g in zip(MINORS, gm))
        return agg, jnp.stack([jnp.mean(gm[0] > 0), g0 > 0, jnp.mean(gm[1] > 0)])

    # ── a minor deviator against frozen public paths ─────────────────────────

    def rollout_minor(self, policy, j, A_traj, rho_traj, H_traj, P_traj, idio, ctx):
        """Type-j price-taking agents against frozen (A, rho, H). idio (B, T).
        Returns total reward (B,)."""
        T_int = int(self.T)
        B = idio.shape[0]
        X0 = jnp.broadcast_to(self.type_initial_state(j), (B, 16))

        def step(carry, t):
            X, cum = carry
            A_exec, rho = A_traj[t + 1], rho_traj[t + 1]
            inp = vmap(self.net_input, in_axes=(0, None))(X, H_traj[t])
            out = vmap(lambda z: policy(z, ctx))(inp)
            a = vmap(self.act_at, in_axes=(0, 0, None))(X, out, A_exec)
            a = a.at[:, 0].set(jnp.where(a[:, 0] >= 0, rho[1] * a[:, 0], rho[0] * a[:, 0]))
            cum = cum + vmap(self.running_reward, in_axes=(0, 0, None))(X, a, A_exec)
            Xn = vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                X, a, idio[:, t], A_exec, P_traj[t + 1])
            return (Xn, cum), None

        (Xf, cum), _ = jax.lax.scan(step, (X0, jnp.zeros(B)), jnp.arange(T_int))
        return cum + vmap(self.terminal_reward)(Xf)

    # ── the major as a price taker against frozen public paths ───────────────

    def rollout_major(self, policy, A_traj, rho_traj, H_traj, P_traj, idio_maj, ctx):
        """The aggregate major (state x m0) price-taking against frozen (A, rho, H),
        mirroring rollout_minor for one player.
        idio_maj (T,) in aggregate units (raw x m0). Returns the total reward (scalar)."""
        T_int = int(self.T)
        X0 = self.major_initial_state()

        def step(carry, t):
            X, cum = carry
            A_exec, rho = A_traj[t + 1], rho_traj[t + 1]
            out = policy(self.net_input(X, H_traj[t]), ctx)
            a = self.act_at(X, out, A_exec)
            a = a.at[0].set(jnp.where(a[0] >= 0, rho[1] * a[0], rho[0] * a[0]))
            cum = cum + self.running_reward(X, a, A_exec)
            Xn = self.single_step_dynamics(X, a, idio_maj[t], A_exec, P_traj[t + 1])
            return (Xn, cum), None

        (Xf, cum), _ = jax.lax.scan(step, (X0, jnp.float32(0.0)), jnp.arange(T_int))
        return cum + self.terminal_reward(Xf)
