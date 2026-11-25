"""Market clearing, version 2 (MARKET_CLEARING.md rev. 7be619c), on top of the
v1 environment. Only the clearing, the population simulation and the
deviator's execution change; states, rewards, dynamics and normalisation are
inherited unchanged.

Clearing (§1.2–1.6): price-independent exogenous supply S_t = sigma_S(gamma) eps sqrt(dt),
sigma_S(gamma) = c Theta(gamma) / sqrt(365), Theta = sum_j w_j theta^j_0 (allocation per unit mass);
corridor [floor, A_CEIL_V2]; clearing interval [a-, a+] by two bisections and
A_t = clip(A_pre, a-, a+); at a bound the long side is rationed pro rata
(sellers rho_sell at the floor, buyers rho_buy at the ceiling); G = O - B.

Arrays (§1.7):
  A_traj   (T+1,)   [0] = A_REF (pre-open), [t+1] = A_t, price of step t
  rho_traj (T+1, 2) [0] = (1, 1),           [t+1] = (rho_sell_t, rho_buy_t)
Population ('agent' sampling, §1.8): Xs[j] is (n, 16) for every type j, and
comp[j] (n,) gives the mixture component of each simulated agent.
"""
import jax
import jax.numpy as jnp
from jax import vmap

from envs.environment import ExogenousMarketEnvJAX, N_TYPES
from envs.params_v2 import (A_REF, A_CEIL_V2, SUPPLY_C, CLEAR_ITERS_V2, FLOOR_MODE)
from envs.models_v2 import apply_gathered

BOUND_TOL = 1e-4      # USD: A_t counted on a bound


def stratified_components(key, k, n):
    """(n,) indices l_i = (o + floor(i k / n)) mod k, o ~ U{0..k-1}.
    k may be a traced integer. key=None -> offset 0 (the 'block' layout)."""
    i = jnp.arange(n)
    o = 0 if key is None else jax.random.randint(key, (), 0, k)
    return (o + (i * k) // n) % k


class MarketClearingEnvV2(ExogenousMarketEnvJAX):
    """v1 env + blocking corridor with two-sided rationing, per-agent sampling."""

    # ── supply and clearing ──────────────────────────────────────────────────

    def total_allocation(self):
        """Theta(gamma) = sum_j w_j theta^j_0(gamma): initial allowances per unit mass.
        Depends on init_alloc and on the share of financial institutions (which
        hold none), through context_weights. Traceable, so it is per market under vmap."""
        w = self.context_weights()
        th0 = jnp.stack([self.type_initial_state(j)[self.dic_indexes['theta']]
                         for j in range(N_TYPES)])
        return jnp.sum(w * th0)

    def sigma_S(self):
        """sigma_S(gamma) = c Theta(gamma) / sqrt(365): the yearly cumulative supply
        shock, sigma_S sqrt(T dt) = sigma_S sqrt(365), has std c Theta whatever T."""
        return SUPPLY_C * self.total_allocation() / jnp.sqrt(365.0)

    def supply(self, eps_t):
        """Exogenous net supply S_t; positive = exogenous traders sell. Price-independent."""
        return self.sigma_S() * eps_t * jnp.sqrt(self.dt)

    def clear(self, orders_fn, eps_t, A_pre):
        """orders_fn(a) -> (buy, sell): weighted mean buy and sell volumes of the
        agents (both >= 0). Returns A_t, rho_sell, rho_buy, G and diagnostics."""
        S = self.supply(eps_t)

        def D(a):
            buy, sell = orders_fn(a)
            return buy - sell - S

        lo_ = jnp.asarray(self.Afloor, jnp.float32)
        hi_ = jnp.asarray(A_CEIL_V2, jnp.float32)
        d_lo, d_hi = D(lo_), D(hi_)

        def left(_, lh):      # a_minus = sup{a : D(a) > 0}; invariant D(l) > 0 >= D(h)
            l, h = lh
            m = 0.5 * (l + h)
            pos = D(m) > 0
            return jnp.where(pos, m, l), jnp.where(pos, h, m)

        def right(_, lh):     # a_plus = inf{a : D(a) < 0}; invariant D(l) >= 0 > D(h)
            l, h = lh
            m = 0.5 * (l + h)
            neg = D(m) < 0
            return jnp.where(neg, l, m), jnp.where(neg, m, h)

        a_minus = jnp.where(d_lo <= 0, lo_, jnp.where(
            d_hi > 0, hi_, jax.lax.fori_loop(0, CLEAR_ITERS_V2, left, (lo_, hi_))[1]))
        a_plus = jnp.where(d_hi >= 0, hi_, jnp.where(
            d_lo < 0, lo_, jax.lax.fori_loop(0, CLEAR_ITERS_V2, right, (lo_, hi_))[0]))
        A_t = jnp.clip(A_pre, a_minus, a_plus)

        # rationing of the long side at a bound (ratios 1 in the interior)
        buy, sell = orders_fn(A_t)
        B = buy + jnp.maximum(-S, 0.0)        # exogenous traders buying
        O = sell + jnp.maximum(S, 0.0)        # exogenous traders selling
        # ration only where a bound BINDS (D(floor) < 0 or D(ceil) > 0); in the
        # interior B = O up to the bisection residual and both ratios are 1.
        # (A tolerance relative to B + O fails at an interior root where every
        # order is ~0: the residual then exceeds it and rationing is spurious.)
        floor_binds = d_lo < 0
        ceil_binds = d_hi > 0
        rho_sell = jnp.where(floor_binds, B / jnp.maximum(O, 1e-12), 1.0)
        rho_buy = jnp.where(ceil_binds, O / jnp.maximum(B, 1e-12), 1.0)
        G = jnp.where(floor_binds | ceil_binds, O - B, 0.0)   # >0 floor, <0 ceiling, 0 inside
        if FLOOR_MODE == 'support':           # the government takes the other side of G
            rho_sell = jnp.ones_like(rho_sell)
            rho_buy = jnp.ones_like(rho_buy)
        return A_t, rho_sell, rho_buy, G, {
            'B': B, 'O': O, 'S': S, 'D_At': D(A_t),
            'a_minus': a_minus, 'a_plus': a_plus,
            'on_floor': floor_binds, 'on_ceil': ceil_binds}

    @staticmethod
    def executed(f, rho_sell, rho_buy):
        """Executed trade: each side scaled by its fill ratio (1 when not rationed)."""
        return jnp.where(f >= 0.0, rho_buy * f, rho_sell * f)

    # ── population ('agent' sampling, §1.8) ──────────────────────────────────

    def population_initial_states_agent(self, n):
        """Tuple of N_TYPES arrays, each (n, 16). Same n for every type; the
        proportions enter only through context_weights()."""
        return tuple(jnp.broadcast_to(self.type_initial_state(j), (n, 16))
                     for j in range(N_TYPES))

    def population_components(self, key, k, n):
        """Per type, the (n,) stratified component indices, one offset per type."""
        if key is None:
            return tuple(stratified_components(None, k, n) for _ in range(N_TYPES))
        ks = jax.random.split(key, N_TYPES)
        return tuple(stratified_components(ks[j], k, n) for j in range(N_TYPES))

    def _population_step_v2(self, buffers, statics, comp, Xs, t_idx, P_traj, eps0,
                            idio_pop, w):
        """ONE clearing step. buffers[j]: params with leading dim K_MAX;
        comp[j], Xs[j], idio_pop[j]: (n,), (n, 16), (n, T).
        Returns new_Xs, A_t, (rho_sell, rho_buy), actions, diag."""
        idx = self.dic_indexes
        norm = vmap(self.normalize_state)
        sp = vmap(self.schedule_params)
        tr = vmap(self.trade_at, in_axes=(0, 0, 0, 0, 0, None))
        rp = vmap(self.reservation_price)
        step_dyn = vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))

        outs = [apply_gathered(buffers[j], statics[j], comp[j], norm(Xs[j]))
                for j in range(N_TYPES)]
        sps = [sp(Xs[j], outs[j]) for j in range(N_TYPES)]

        def orders_fn(a):
            buy, sell = 0.0, 0.0
            for j in range(N_TYPES):
                al, be, lo, hi, _, _ = sps[j]
                f = tr(Xs[j], al, be, lo, hi, a)
                buy = buy + w[j] * jnp.mean(jnp.maximum(f, 0.0))
                sell = sell + w[j] * jnp.mean(jnp.maximum(-f, 0.0))
            return buy, sell

        A_pre = Xs[0][0, idx['A']]            # every agent holds the same last price
        A_t, rho_s, rho_b, G, cd = self.clear(orders_fn, eps0[t_idx], A_pre)

        actions, f_raw = [], []
        for j in range(N_TYPES):
            al, be, lo, hi, xi, eta = sps[j]
            f = tr(Xs[j], al, be, lo, hi, A_t)
            f_raw.append(f)
            actions.append(jnp.stack([self.executed(f, rho_s, rho_b), xi, eta], axis=-1))

        new_Xs = tuple(step_dyn(Xs[j], actions[j], idio_pop[j][:, t_idx], A_t,
                                P_traj[t_idx + 1]) for j in range(N_TYPES))

        S = cd['S']
        S_exec = jnp.where(S > 0, rho_s * S, rho_b * S)  # exogenous orders rationed too
        dth = sum(w[j] * jnp.mean(new_Xs[j][:, idx['theta']] - Xs[j][:, idx['theta']])
                  for j in range(N_TYPES))
        G_gov = G if FLOOR_MODE == 'support' else 0.0
        sell_t = jnp.stack([w[j] * jnp.mean(jnp.maximum(-f_raw[j], 0.0)) for j in range(N_TYPES)])
        buy_t = jnp.stack([w[j] * jnp.mean(jnp.maximum(f_raw[j], 0.0)) for j in range(N_TYPES)])
        on_bound = cd['on_floor'] | cd['on_ceil']
        diag = {
            'on_floor': cd['on_floor'], 'on_ceil': cd['on_ceil'],
            'flat': (cd['a_plus'] - cd['a_minus']) > BOUND_TOL,
            'a_minus': cd['a_minus'], 'a_plus': cd['a_plus'],
            'rho_sell': rho_s, 'rho_buy': rho_b, 'G': G, 'B': cd['B'], 'O': cd['O'],
            'resid': jnp.where(on_bound, 0.0, jnp.abs(cd['D_At'])),
            'S': S, 'S_exec': S_exec,
            'cons_err': dth - (S_exec - G_gov),
            'blocked_sell_type': (1.0 - rho_s) * sell_t,
            'blocked_buy_type': (1.0 - rho_b) * buy_t,
            'R_mean': jnp.stack([jnp.mean(rp(Xs[j], outs[j])) for j in range(N_TYPES)]),
            'R_median': jnp.stack([jnp.median(rp(Xs[j], outs[j])) for j in range(N_TYPES)]),
            'beta_mean': jnp.stack([jnp.mean(outs[j][:, 1]) for j in range(N_TYPES)]),
            'beta_std': jnp.stack([jnp.std(outs[j][:, 1]) for j in range(N_TYPES)]),
            'f_mean': jnp.stack([jnp.mean(actions[j][:, 0]) for j in range(N_TYPES)]),
        }
        return new_Xs, A_t, jnp.stack([rho_s, rho_b]), actions, diag

    def ensemble_price_path_v2(self, buffers, statics, k, comp_key, P_traj, eps0,
                               idio_pop, return_diag=False):
        """Cleared price and fill ratios under the mixture (buffer, size k).
        comp_key: one key per market for the stratified assignment (None ->
        offset 0). idio_pop: tuple of (n, T). Returns A_traj (T+1,) and
        rho_traj (T+1, 2); with return_diag also per-step diagnostics + terminal gaps."""
        T_int = int(self.T)
        n = idio_pop[0].shape[0]
        X0 = self.population_initial_states_agent(n)
        w = self.context_weights()
        comp = self.population_components(comp_key, k, n)

        def step_fn(Xs, t_idx):
            new_Xs, A_t, rho, _, diag = self._population_step_v2(
                buffers, statics, comp, Xs, t_idx, P_traj, eps0, idio_pop, w)
            return new_Xs, ((A_t, rho, diag) if return_diag else (A_t, rho))

        Xf, ys = jax.lax.scan(step_fn, X0, jnp.arange(T_int))
        A, rho = ys[0], ys[1]
        A_traj = jnp.concatenate([jnp.atleast_1d(jnp.float32(A_REF)), A])
        rho_traj = jnp.concatenate([jnp.ones((1, 2), jnp.float32), rho], axis=0)
        if not return_diag:
            return A_traj, rho_traj
        diag = dict(ys[2], agg_gap_T=self.aggregate_gap(Xf, w), **self.terminal_gaps(Xf))
        return A_traj, rho_traj, diag

    # ── the deviator (price taker) ───────────────────────────────────────────

    def rollout_agent_batch_v2(self, policy, j, A_traj, rho_traj, P_traj, idio):
        """Type-j agents against frozen (A, rho): schedule evaluated at
        A_exec = A_traj[t+1], executed with rho_traj[t+1] = (rho_sell, rho_buy)
        (gradient flows through f_exec = rho f with rho fixed).
        Returns (X_final (B,16), total reward (B,))."""
        T_int = int(self.T)
        B = idio.shape[0]
        X0 = jnp.broadcast_to(self.type_initial_state(j), (B, 16))

        def step_fn(carry, t_idx):
            X_t, cum = carry
            A_exec, rho = A_traj[t_idx + 1], rho_traj[t_idx + 1]
            out = vmap(policy)(vmap(self.normalize_state)(X_t))
            a_t = vmap(self.act_at, in_axes=(0, 0, None))(X_t, out, A_exec)
            a_t = a_t.at[:, 0].set(self.executed(a_t[:, 0], rho[0], rho[1]))
            r_t = vmap(self.running_reward, in_axes=(0, 0, None))(X_t, a_t, A_exec)
            next_X = vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                X_t, a_t, idio[:, t_idx], A_exec, P_traj[t_idx + 1])
            return (next_X, cum + r_t), None

        (X_final, cum), _ = jax.lax.scan(step_fn, (X0, jnp.zeros(B)), jnp.arange(T_int))
        return X_final, cum + vmap(self.terminal_reward)(X_final)
