import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from envs.base_params import *
from envs.models import apply_stacked

N_TYPES = 3   # 0 = private producer, 1 = large public producer, 2 = market maker


class ExogenousMarketEnvJAX(eqx.Module):
    """K-ETS mean-field environment.

    Rung 1 conventions (plan.md §6):
      * policies are NOT stored on the env — they are passed explicitly, one
        network per type per FP block;
      * the common noise (P path, allowance shock) and the population's
        idiosyncratic draws are passed in frozen, so the aggregate price path
        is a deterministic function of the policies.
    """

    # --- Truly static ---
    agent_per_policy: int   = eqx.field(static=True)
    type_counts:      tuple = eqx.field(static=True)   # (n1, n2, n3)
    dic_indexes:      dict  = eqx.field(static=True)

    # --- Callables ---
    # NOTE: stored raw, taking T explicitly. The previous
    #   self.get_P_trajectory = lambda key: generate_P_func(key, self.T)
    # closed over the instance under construction, so after set_T (which builds
    # a NEW instance via tree_at) the lambda still read the OLD self.T — the
    # multigrid would silently have generated price paths at the wrong horizon.
    generate_P:                   callable = eqx.field(static=True)
    market_impact:                callable = eqx.field(static=True)
    generate_eps0:                callable = eqx.field(static=True)
    generate_idiosyncratic_noise: callable = eqx.field(static=True)

    # --- JAX arrays ---
    kappa:          jnp.ndarray
    T:              jnp.ndarray
    dt:             jnp.ndarray
    A_scale:        jnp.ndarray
    P_scale:        jnp.ndarray
    A0:             jnp.ndarray
    P0:             jnp.ndarray
    agent_params:   jnp.ndarray
    max_cap_f:      jnp.ndarray
    max_cap_xi:     jnp.ndarray
    max_sigma_eps:  jnp.ndarray

    # --- Context fields (set per-scenario via set_context) ---
    init_alloc:     jnp.ndarray
    Afloor:         jnp.ndarray
    financial_inst: jnp.ndarray
    mm_share:       jnp.ndarray
    major_share:    jnp.ndarray     # v5: s, the major's share of the total cap
    BM_uniform:     jnp.ndarray

    def __init__(self, kappa, T, generate_P_func, A0, P0, market_impact_func,
                 generate_eps0_func, generate_eps_idiosyncratic_func,
                 A_scale=1.0, P_scale=1.0, agent_per_policy=100,
                 financial_inst=0.4):

        self.kappa   = jnp.float32(kappa)
        self.T       = jnp.float32(T)
        self.dt      = jnp.float32(365 / T)
        self.A_scale = jnp.float32(A_scale)
        self.P_scale = jnp.float32(P_scale)
        self.A0      = jnp.float32(A0)
        self.P0      = jnp.float32(P0)

        self.agent_per_policy = int(agent_per_policy)

        self.agent_params  = jnp.array([PRIVATE_GENERATOR, BIG_PUBLIC_GENERATOR, MARKET_MAKER])
        self.max_cap_f     = jnp.max(self.agent_params[:, 5])
        self.max_cap_xi    = jnp.max(self.agent_params[:, 6])
        self.max_sigma_eps = jnp.max(self.agent_params[:, 7])

        self.generate_P                   = generate_P_func
        self.market_impact                = market_impact_func
        self.generate_eps0                = generate_eps0_func
        self.generate_idiosyncratic_noise = generate_eps_idiosyncratic_func

        self.init_alloc     = jnp.float32(0.0)   # overridden by set_context
        self.Afloor         = jnp.float32(A_FLOOR_BASE)
        # financial_inst keeps its constructor meaning: it decides the SIMULATED
        # population (type_counts) and never changes afterwards. mm_share is the
        # CONTEXT value, rebound per market, and is what the market responds to.
        # Two fields on purpose: eqx.tree_at on financial_inst would move the
        # recorded number while type_counts silently kept the old population.
        self.financial_inst = jnp.float32(financial_inst)
        self.mm_share       = jnp.float32(financial_inst)
        self.major_share    = jnp.float32(0.0)
        self.BM_uniform     = jnp.float32(1.0)

        # Population composition: fixed proportions, static so shapes are known.
        n_fi   = round(float(financial_inst) * int(agent_per_policy))
        n_prod = int(agent_per_policy) - n_fi
        n_big  = round(BIG_ON_PRIVATE * n_prod)
        self.type_counts = (n_prod - n_big, n_big, n_fi)

        self.dic_indexes = {
            "t": 0, "theta": 1,
            "xi": slice(2, 4), "xi_f": 2, "xi_g": 3,
            "eps": 4, "A": 5, "P": 6, "A_avg": 7,
            "static": 8,
            "e": slice(8, 10), "ef": 8, "eg": 9,
            "c": slice(10, 12), "cf": 10, "cg": 11,
            "cap_f": 12, "cap_xi": 13, "sigma_eps": 14,
            "tech": 15, "dim": 16
        }

    # ── configuration ────────────────────────────────────────────────────────

    def set_T(self, T):
        model = eqx.tree_at(lambda e: e.T,  self,  jnp.float32(T))
        model = eqx.tree_at(lambda e: e.dt, model, jnp.float32(365 / T))
        return model

    def set_context(self, context):
        """context: (4,) — [init_alloc, kappa, Afloor, financial_inst].

        A (2,) context is still accepted and read as [init_alloc, kappa], so the
        first two slots keep the meaning they have always had.
        """
        c = jnp.asarray(context, dtype=jnp.float32)
        model = eqx.tree_at(lambda e: e.init_alloc, self,  jnp.float32(c[0]))
        model = eqx.tree_at(lambda e: e.kappa,      model, jnp.float32(c[1]))
        if c.shape == (2,):
            return model
        model = eqx.tree_at(lambda e: e.Afloor,   model, jnp.float32(c[2]))
        model = eqx.tree_at(lambda e: e.mm_share, model, jnp.float32(c[3]))
        return model

    def generate_context(self, key, B):
        """B draws of the four-dimensional regulatory context.

        init_alloc ~ U(0, 1.2), kappa ~ U(-0.5, 7.2), Afloor ~ U(FLOOR_MIN,
        FLOOR_MAX), financial_inst ~ U(0, 0.5). BM_uniform is NOT part of the
        context: it stays at its constructor value, so the conditioner never
        sees it move.
        """
        if LAYOUT == 'v5':             # (c, s, Afloor, mm_share, kappa2), uniform on the box
            ks = jax.random.split(key, 5)
            lo = jnp.array([C_MIN, S_MIN, FLOOR_MIN, MM_MIN, KAPPA_MIN], jnp.float32)
            hi = jnp.array([C_MAX, S_MAX, FLOOR_MAX, MM_MAX, KAPPA_MAX], jnp.float32)
            return jnp.stack([jax.random.uniform(ks[i], (B,), minval=lo[i], maxval=hi[i])
                              for i in range(5)], axis=1)
        k_a, k_k, k_f, k_m = jax.random.split(key, 4)
        allocation = jax.random.uniform(k_a, (B,), minval=ALLOC_MIN, maxval=ALLOC_MAX)
        kappa      = jax.random.uniform(k_k, (B,), minval=KAPPA_MIN, maxval=KAPPA_MAX)
        floor      = jax.random.uniform(k_f, (B,), minval=FLOOR_MIN, maxval=FLOOR_MAX)
        mm         = jax.random.uniform(k_m, (B,), minval=MM_MIN,    maxval=MM_MAX)
        return jnp.stack([allocation, kappa, floor, mm], axis=1)

    def context_weights(self):
        """Population shares implied by mm_share -> (3,).

        Why the proportion can be a context dimension at all: `type_counts` is
        static and fixes array shapes, but it only controls how many agents are
        SIMULATED — it is not how the proportion reaches the market. The price
        update aggregates with a plain mean over a LINEAR impact (see
        compute_next_A), so

            mean_i h(f_i) = h( sum_j (n_j/N) * mean_{i in j} f_i )

        and the proportion enters only as the weights of a convex combination of
        per-type mean trades. Integer counts are a finite-population artifact;
        the mean-field object is the measure over types. Weights are traceable,
        so they can be sampled per market and vmapped like any other value.

        Rounded to whole agents first, so a weight is exactly int(N*p)/N and, h
        being linear, is identical to simulating that many agents rather than an
        approximation of it.
        """
        if LAYOUT == 'v5':             # continuous masses (RUNS_SETTING.md §1)
            m0 = 0.0 if NO_MAJOR else MAJOR_MASS_V5
            return jnp.stack([PRIV_MASS_V5 - self.mm_share, jnp.float32(m0) + 0.0 * self.mm_share,
                              self.mm_share])
        N      = jnp.float32(self.agent_per_policy)
        n_fi   = jnp.round(N * self.mm_share)
        n_prod = N - n_fi
        n_big  = 0.0 * n_prod if NO_MAJOR else jnp.round(jnp.float32(BIG_ON_PRIVATE) * n_prod)
        return jnp.stack([n_prod - n_big, n_big, n_fi]) / N

    @property
    def type_shares(self):
        n = float(sum(self.type_counts))
        return tuple(c / n for c in self.type_counts)

    # ── initial states ───────────────────────────────────────────────────────

    def _bm_multipliers(self):
        """Free-allocation benchmark multiplier, one per type."""
        bm_fuel_green = jnp.array([
            BM_FUEL * BM_RATIO['base']  + BM_GREEN * (1 - BM_RATIO['base']),
            BM_FUEL * BM_RATIO['large'] + BM_GREEN * (1 - BM_RATIO['large']),
            0.0,
        ])
        return jnp.where(self.BM_uniform >= 0.5, BM_UNIFORM, bm_fuel_green)

    @staticmethod
    def create_initial_state(agent_params, A0, P0, T):
        """Built with jnp.stack rather than jnp.array([...]) on a python list:
        the type index is a tracer when the best response is vmapped over
        types, so agent_params[0] is a tracer and a mixed python-list literal
        is a rough edge."""
        dt_   = agent_params.dtype
        zero  = jnp.zeros((), dt_)
        head  = jnp.stack([zero, agent_params[0] * T, zero, zero, zero])
        mid   = jnp.stack([jnp.asarray(A0, dt_), jnp.asarray(P0, dt_), zero])
        return jnp.concatenate([head, mid, agent_params[1:]])

    def type_initial_state(self, j):
        """(16,) initial state of a type-j agent under the current scenario.

        Replaces initialize_state_training (D2): the best response is now
        trained on the type's true parameters, not on a random hyperbox that
        averaged 3.5x the population's capacity.
        """
        base = self.create_initial_state(self.agent_params[j], self.A0, self.P0, self.T)
        if LAYOUT == 'v5':
            return base.at[1].set(self.theta_v5()[j])
        theta = base[1] * self._bm_multipliers()[j] * self.init_alloc
        return base.at[1].set(theta)

    def total_cap_v5(self):
        """Theta_tot = c T sum_l n_l C^xi_l e^{l,fuel}: the emissions at full capacity with the
        dirtiest technology, times the coverage ratio c (= init_alloc's slot in v5)."""
        w = self.context_weights()
        ap = self.agent_params
        full = self.T * ap[:, 6] * jnp.maximum(ap[:, 1], ap[:, 2])      # per unit of mass
        return self.init_alloc * jnp.sum(w * full)

    def theta_v5(self):
        """Initial holdings PER UNIT OF MASS of each label: the major gets s Theta_tot (its
        state is per unit of mass x m0), the private producers the rest, market makers 0.
        Without a major everything goes to the private producers."""
        w = self.context_weights()
        tot = self.total_cap_v5()
        s_priv = jnp.where(w[1] > 0, 1.0 - self.major_share, 1.0)
        th_priv = s_priv * tot / jnp.maximum(w[0], 1e-12)
        th_major = jnp.where(w[1] > 0, self.major_share * tot / jnp.maximum(w[1], 1e-12), 0.0)
        return jnp.stack([th_priv, th_major, jnp.float32(0.0) * tot])

    def population_initial_states(self, K):
        """Tuple of N_TYPES arrays, each (K, n_j, 16).

        Every FP block holds the same type composition, so the population
        shares are preserved automatically for any K (plan.md §3).
        """
        return tuple(
            jnp.broadcast_to(self.type_initial_state(j), (K, self.type_counts[j], 16))
            for j in range(N_TYPES)
        )

    # ── reward and dynamics ──────────────────────────────────────────────────

    @staticmethod
    def running_reward(x, a, A_exec):
        """A_exec: the price CLEARED at this step (A_traj[t+1]), at which the
        trade is executed — not the pre-trade price x[5]."""
        prod_coef     = x[6] - x[10:12]                       # P - (cf, cg)
        quantities    = a[1] * jnp.array([a[2], 1.0 - a[2]])  # xi * (eta, 1-eta)
        prod_revenue  = jnp.dot(quantities, prod_coef)
        trade_revenue = -a[0] * A_exec                        # -f * A_t
        return prod_revenue + trade_revenue

    def terminal_reward(self, x):
        idx        = self.dic_indexes
        emissions  = jnp.dot(x[idx["xi"]], x[idx["e"]]) + x[idx["eps"]]
        allowances = x[idx["theta"]]
        shortage   = jnp.clip(emissions - allowances, min=0.0)
        if PENALTY_MODE == 'fixed':                 # LP benchmark: kappa^1 * shortage
            return -self.kappa * shortage
        return -self.kappa * shortage * x[idx["A_avg"]]

    def reward_scale(self, j):
        """Fixed per-type scale so each net's loss is O(1) and the three loss
        curves are comparable (plan.md §6.1). Not a shaping term: dividing the
        loss by a constant does not move the argmax."""
        p     = self.agent_params[j]
        T     = self.T
        prod  = jnp.abs(self.P0 - p[3]) * p[6] * T   # (P0 - cf) * cap_xi * T
        trade = self.A0 * p[5] * T                   # A0 * cap_f * T
        return jnp.maximum(jnp.maximum(prod, trade), 1.0)

    def compute_next_A_weighted(self, A_t, mean_f, eps0_t):
        """LEGACY (main branch price impact; not called under market clearing).

        compute_next_A given the population's already-weighted mean trade.

        Identical in value to compute_next_A when mean_f is the mean of the same
        trades: h is linear, so h(mean) == mean(h), and this evaluates it once
        instead of N times. Kept separate so every rollout in this class shares
        one aggregation rule — the price path, the trajectory and the
        diagnostics must not drift apart.
        """
        return jnp.maximum(
            A_t + self.market_impact(mean_f) * self.dt
            + eps0_t * jnp.sqrt(self.dt),
            self.Afloor,
        )

    def compute_next_A(self, A_t, array_f, eps0_t):
        """LEGACY (not called under market clearing). Allowance price, h linear.
        Floor only. Under clearing the corridor has an artificial ceiling A_CEIL,
        needed for existence of a clearing price only."""
        individual_impacts = vmap(self.market_impact)(array_f)
        mean_impact        = jnp.mean(individual_impacts)
        return jnp.maximum(
            A_t + mean_impact * self.dt + eps0_t * jnp.sqrt(self.dt),
            self.Afloor,
        )

    def single_step_dynamics(self, x, a, eps, next_A, next_P):
        idx = self.dic_indexes

        t_next     = x[idx["t"]]     + 1.0
        theta_next = x[idx["theta"]] + a[0]
        xi_next    = x[idx["xi"]]    + a[1] * jnp.array([a[2], 1.0 - a[2]])
        eps_next   = x[idx["eps"]]   + eps

        # running mean of the allowance price, used by the terminal penalty
        n_steps     = jnp.maximum(t_next, 0.0)
        current_avg = x[idx["A_avg"]]
        new_avg     = jnp.where(
            t_next > 0,
            (current_avg * (n_steps - 1.0) + next_A) / jnp.maximum(n_steps, 1.0),
            0.0,
        )

        static_part = x[idx['static']:]
        return jnp.concatenate([
            jnp.array([t_next, theta_next]),
            xi_next,
            jnp.array([eps_next, next_A, next_P, new_avg]),
            static_part
        ])

    # ── normalisation ────────────────────────────────────────────────────────

    def normalize_state(self, x):
        """(16,) state -> (8,) network input. Type descriptors are dropped:
        they are constant within a per-type network (plan.md §3)."""
        idx = self.dic_indexes
        return jnp.stack([
            x[idx["t"]]     / self.T,
            x[idx["theta"]] / (self.T * x[idx["cap_f"]]  + 1e-4),
            x[idx["xi_f"]]  / (self.T * x[idx["cap_xi"]] + 1e-4),
            x[idx["xi_g"]]  / (self.T * x[idx["cap_xi"]] + 1e-4),
            x[idx["eps"]]   / (jnp.sqrt(self.T) * x[idx["sigma_eps"]] + 1e-4),
            x[idx["A"]]     / self.A_scale,
            x[idx["P"]]     / self.P_scale,
            x[idx["A_avg"]] / self.A_scale,
        ])

    def unnormalize_action(self, x, hat_a):
        """Map the network output into A(j, x) = [-min(theta, cap_f), cap_f]
        x [0, cap_xi] x [0, 1].

        Uses a two-sided clip rather than the old one-sided `where`, which
        returned -theta whenever a0_raw < -theta and so could emit a trade
        larger than cap_f when theta < 0 (D6). With ALLOC_MIN = 0 that regime
        is gone, but the clip is the correct constraint regardless.

        The tech override lives here now, not in the network: the per-type nets
        no longer see the tech flag.
        """
        idx    = self.dic_indexes
        cap_f  = x[idx['cap_f']]
        theta  = x[idx['theta']]

        lo = -jnp.minimum(theta, cap_f)
        a0 = jnp.clip(hat_a[0] * cap_f, lo, cap_f)
        a1 = hat_a[1] * x[idx['cap_xi']]

        tech = x[idx['tech']]
        eta  = jnp.where(tech >= 0.0, tech, hat_a[2])

        return jnp.stack([a0, a1, eta])

    # ── market clearing (MARKET_CLEARING.md §1) ─────────────────────────────

    def schedule_params(self, x, out):
        """(16,) pre-trade state, (4,) net output -> (alpha, beta, lo, hi, xi, eta).
        lo/hi are the pre-trade clip bounds; xi/eta are unnormalised."""
        idx   = self.dic_indexes
        cap_f = x[idx['cap_f']]
        theta = x[idx['theta']]
        lo, hi = -jnp.minimum(theta, cap_f), cap_f
        xi  = out[2] * x[idx['cap_xi']]
        eta = jnp.where(x[idx['tech']] >= 0.0, x[idx['tech']], out[3])
        return out[0], out[1], lo, hi, xi, eta

    def trade_at(self, x, alpha, beta, lo, hi, a):
        """f(a) for one particle; continuous and nonincreasing in a."""
        idx = self.dic_indexes
        z = alpha - beta * (a - x[idx['A']]) / self.A_scale
        return jnp.clip(x[idx['cap_f']] * jnp.tanh(z), lo, hi)

    def act_at(self, x, out, a):
        """Action (f(a), xi, eta) of one particle at price a."""
        alpha, beta, lo, hi, xi, eta = self.schedule_params(x, out)
        return jnp.stack([self.trade_at(x, alpha, beta, lo, hi, a), xi, eta])

    def reservation_price(self, x, out):
        """R = A_pre + A_scale * alpha / beta: the price at which f = 0."""
        return x[self.dic_indexes['A']] + self.A_scale * out[0] / out[1]

    def clear_price(self, D, A_pre):
        """D: scalar a -> excess demand, nonincreasing.
        Returns (A_t, a_minus, a_plus); the solution set is [a_minus, a_plus] and
        A_t = clip(A_pre, a_minus, a_plus) moves the price as little as possible.
        Two fixed-length bisections (jit / vmap safe)."""
        lo_ = jnp.asarray(self.Afloor, jnp.float32)
        hi_ = jnp.asarray(A_CEIL, jnp.float32)

        def left(_, lh):    # a_minus = sup{a : D(a) > 0}; D(l) > 0 >= D(h)
            l, h = lh
            m = 0.5 * (l + h)
            pos = D(m) > 0
            return jnp.where(pos, m, l), jnp.where(pos, h, m)

        def right(_, lh):   # a_plus = inf{a : D(a) < 0}; D(l) >= 0 > D(h)
            l, h = lh
            m = 0.5 * (l + h)
            neg = D(m) < 0
            return jnp.where(neg, l, m), jnp.where(neg, m, h)

        d_lo, d_hi = D(lo_), D(hi_)
        a_minus = jnp.where(d_lo <= 0, lo_, jnp.where(
            d_hi > 0, hi_, jax.lax.fori_loop(0, CLEAR_ITERS, left, (lo_, hi_))[1]))
        a_plus = jnp.where(d_hi >= 0, hi_, jnp.where(
            d_lo < 0, lo_, jax.lax.fori_loop(0, CLEAR_ITERS, right, (lo_, hi_))[0]))
        return jnp.clip(A_pre, a_minus, a_plus), a_minus, a_plus

    def excess_demand(self, Xs, outs, S_t, w):
        """D(a) = sum_j w_j mean_{l,i} f^{j,i,l}(a) - S_t, closed form in a.
        Xs, outs: per type (K, n_j, 16) states and (K, n_j, 4) net outputs.
        Returns (D, schedule params per type)."""
        sp2 = vmap(vmap(self.schedule_params))
        tr2 = vmap(vmap(self.trade_at, in_axes=(0, 0, 0, 0, 0, None)),
                   in_axes=(0, 0, 0, 0, 0, None))
        sps = [sp2(Xs[j], outs[j]) for j in range(N_TYPES)]

        def D(a):
            tot = 0.0
            for j in range(N_TYPES):
                al, be, lo, hi, _, _ = sps[j]
                tot = tot + w[j] * jnp.mean(tr2(Xs[j], al, be, lo, hi, a))
            return tot - S_t
        return D, sps

    def _population_step(self, stacked, statics, Xs, t_idx, P_traj, eps0,
                         idio_pop, w):
        """ONE clearing step of the simulated population. Shared by every
        ensemble rollout so they cannot drift apart.

        Returns new_Xs, A_t, actions (per type (K, n_j, 3)), diag (dict)."""
        idx = self.dic_indexes
        K   = Xs[0].shape[0]
        norm2 = vmap(vmap(self.normalize_state))
        tr2   = vmap(vmap(self.trade_at, in_axes=(0, 0, 0, 0, 0, None)),
                     in_axes=(0, 0, 0, 0, 0, None))
        rp2   = vmap(vmap(self.reservation_price))
        step_dyn = vmap(
            vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None)),
            in_axes=(0, 0, 0, None, None))

        # 1-2. one forward pass per type on the pre-trade states
        outs = [apply_stacked(stacked[j], statics[j], norm2(Xs[j]))
                for j in range(N_TYPES)]

        # 3. aggregate excess demand, closed form in a
        S_t = SIGMA_S * eps0[t_idx] * jnp.sqrt(self.dt)
        D, sps = self.excess_demand(Xs, outs, S_t, w)

        # 4. clear
        A_pre = Xs[0][0, 0, idx['A']]
        A_t, a_minus, a_plus = self.clear_price(D, A_pre)

        # 5. execute at A_t
        actions = []
        for j in range(N_TYPES):
            al, be, lo, hi, xi, eta = sps[j]
            f = tr2(Xs[j], al, be, lo, hi, A_t)
            actions.append(jnp.stack([f, xi, eta], axis=-1))

        # 6. dynamics
        new_Xs = []
        for j in range(N_TYPES):
            eps_j = jnp.broadcast_to(idio_pop[j][:, t_idx], (K, self.type_counts[j]))
            new_Xs.append(step_dyn(Xs[j], actions[j], eps_j, A_t, P_traj[t_idx + 1]))

        # diagnostics. G = government purchases (>0 at the floor, <0 at the
        # ceiling); resid = violation of D(A_t) in N_[floor, ceil](A_t),
        # i.e. |D| inside, max(D, 0) on the floor, max(-D, 0) on the ceiling.
        D_t      = D(A_t)
        on_floor = A_t <= self.Afloor + CLEAR_TOL
        on_ceil  = A_t >= A_CEIL - CLEAR_TOL
        G = jnp.where(on_floor, jnp.maximum(-D_t, 0.0),
                      jnp.where(on_ceil, jnp.minimum(-D_t, 0.0), 0.0))
        resid = jnp.where(on_floor, jnp.maximum(D_t, 0.0),
                          jnp.where(on_ceil, jnp.maximum(-D_t, 0.0), jnp.abs(D_t)))
        # conservation, from the states themselves: d theta_bar = S - G
        dth = sum(w[j] * jnp.mean(new_Xs[j][..., idx['theta']] - Xs[j][..., idx['theta']])
                  for j in range(N_TYPES))
        diag = {
            'D': D_t, 'S': S_t, 'G': G, 'resid': resid,
            'on_floor': on_floor, 'on_ceil': on_ceil,
            'flat': (a_plus - a_minus) > CLEAR_TOL,
            'a_minus': a_minus, 'a_plus': a_plus,
            'cons_err': dth - (S_t - G),
            'R_mean': jnp.stack([jnp.mean(rp2(Xs[j], outs[j])) for j in range(N_TYPES)]),
            'R_median': jnp.stack([jnp.median(rp2(Xs[j], outs[j])) for j in range(N_TYPES)]),
            'beta_mean': jnp.stack([jnp.mean(outs[j][..., 1]) for j in range(N_TYPES)]),
            'beta_std': jnp.stack([jnp.std(outs[j][..., 1]) for j in range(N_TYPES)]),
            'f_mean': jnp.stack([jnp.mean(actions[j][..., 0]) for j in range(N_TYPES)]),
        }
        return tuple(new_Xs), A_t, actions, diag

    def terminal_gaps(self, Xs):
        """Per type: mean E_T - theta_T and the fraction of particles short."""
        idx = self.dic_indexes
        gaps, short = [], []
        for j in range(N_TYPES):
            X = Xs[j]
            g = (jnp.sum(X[..., idx['xi']] * X[..., idx['e']], axis=-1)
                 + X[..., idx['eps']] - X[..., idx['theta']])
            gaps.append(jnp.mean(g))
            short.append(jnp.mean(g > 0))
        return {'gap_T_type': jnp.stack(gaps), 'short_T_type': jnp.stack(short)}

    def aggregate_gap(self, Xs, w):
        """Weighted mean of E - theta over the population: > 0 = market short."""
        idx = self.dic_indexes
        g = 0.0
        for j in range(N_TYPES):
            X = Xs[j]
            E = jnp.sum(X[..., idx['xi']] * X[..., idx['e']], axis=-1) + X[..., idx['eps']]
            g = g + w[j] * jnp.mean(E - X[..., idx['theta']])
        return g

    # ── rollouts ─────────────────────────────────────────────────────────────
    # A_traj (T+1,): A_traj[0] = A_REF (pre-open), A_traj[t+1] = price CLEARED
    # at step t, at which the trades of step t are executed.

    def _scan_population(self, stacked, statics, P_traj, eps0, idio_pop):
        """Runs _population_step over the horizon -> (X_final, A (T,), actions
        per step, diag per step)."""
        T_int = int(self.T)
        K     = jax.tree_util.tree_leaves(stacked[0])[0].shape[0]
        X0    = self.population_initial_states(K)
        w     = self.context_weights()

        def step_fn(Xs, t_idx):
            new_Xs, A_t, actions, diag = self._population_step(
                stacked, statics, Xs, t_idx, P_traj, eps0, idio_pop, w)
            return new_Xs, (A_t, actions, diag, Xs)

        Xf, (A, actions, diag, Xs_hist) = jax.lax.scan(step_fn, X0, jnp.arange(T_int))
        return Xf, A, actions, diag, Xs_hist, w

    def ensemble_price_path(self, stacked, statics, P_traj, eps0, idio_pop,
                            return_diag=False):
        """Cleared allowance price under the FP mixture.

        stacked / statics : per-type stacked parameters, leading dim K
        P_traj            : (T+1,)  frozen good-price path
        eps0              : (T,)    frozen net-supply draws (standard normal)
        idio_pop          : tuple of (n_j, T), reused by every block

        Returns A_traj (T+1,), [A_REF, A_0, ..., A_{T-1}]; with return_diag,
        also a dict of per-step clearing diagnostics (each (T,) or (T, 3)) and
        the final aggregate gap (scalar).
        """
        T_int = int(self.T)
        K     = jax.tree_util.tree_leaves(stacked[0])[0].shape[0]
        X0    = self.population_initial_states(K)
        w     = self.context_weights()

        def step_fn(Xs, t_idx):
            new_Xs, A_t, _, diag = self._population_step(
                stacked, statics, Xs, t_idx, P_traj, eps0, idio_pop, w)
            return new_Xs, ((A_t, diag) if return_diag else A_t)

        Xf, ys = jax.lax.scan(step_fn, X0, jnp.arange(T_int))
        A_ref = jnp.atleast_1d(jnp.float32(A_REF))
        if not return_diag:
            return jnp.concatenate([A_ref, ys])
        A, diag = ys
        diag = dict(diag, agg_gap_T=self.aggregate_gap(Xf, w),
                    **self.terminal_gaps(Xf))
        return jnp.concatenate([A_ref, A]), diag

    def rollout_agent_batch(self, policy, j, A_traj, P_traj, idio):
        """A batch of type-j agents responding to a frozen cleared price path.

        The agent's schedule is evaluated at the execution price A_traj[t+1]
        (it is a price taker: it does not move the clearing). This is the only
        rollout used for both best-response training and the exploitability
        metric.

        policy : a TypePolicyNN (or any callable (8,) -> (4,))
        idio   : (B, T) frozen idiosyncratic draws
        Returns (X_final (B,16), total_reward (B,)).
        """
        T_int = int(self.T)
        B     = idio.shape[0]
        X0    = jnp.broadcast_to(self.type_initial_state(j), (B, 16))

        def step_fn(carry, t_idx):
            X_t, cum = carry
            A_exec = A_traj[t_idx + 1]
            out    = vmap(policy)(vmap(self.normalize_state)(X_t))
            a_t    = vmap(self.act_at, in_axes=(0, 0, None))(X_t, out, A_exec)
            r_t    = vmap(self.running_reward, in_axes=(0, 0, None))(X_t, a_t, A_exec)
            next_X = vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                X_t, a_t, idio[:, t_idx], A_exec, P_traj[t_idx + 1])
            return (next_X, cum + r_t), None

        (X_final, cum), _ = jax.lax.scan(
            step_fn, (X0, jnp.zeros(B)), jnp.arange(T_int))
        return X_final, cum + vmap(self.terminal_reward)(X_final)

    def ensemble_trajectory(self, stacked, statics, P_traj, eps0, idio_pop,
                            q_lo=0.1, q_hi=0.9):
        """Per-type trajectories of the FP mixture, with dispersion bands.
        Same outputs as on main; the population is cleared step by step."""
        idx = self.dic_indexes
        Xf, A, actions, diag, Xs_hist, w = self._scan_population(
            stacked, statics, P_traj, eps0, idio_pop)
        T_int = int(self.T)
        K     = Xf[0].shape[0]
        qs    = jnp.array([q_lo, q_hi])
        rew   = vmap(vmap(vmap(self.running_reward, in_axes=(0, 0, None)),
                          in_axes=(0, 0, None)), in_axes=(0, 0, 0))

        out = {'A_traj': jnp.concatenate([jnp.atleast_1d(jnp.float32(A_REF)), A]),
               'P_traj': P_traj, 'state': [], 'action': [], 'action_lo': [],
               'action_hi': [], 'cum_r': [], 'cum_r_lo': [], 'cum_r_hi': [],
               'emission': [], 'emission_lo': [], 'emission_hi': [], 'term_r': [],
               'clearing': diag}
        for j in range(N_TYPES):
            Xj, aj = Xs_hist[j], actions[j]                      # (T,K,n,16), (T,K,n,3)
            r = rew(Xj, aj, A)                                   # (T,K,n)
            cum = jnp.cumsum(r, axis=0).reshape(T_int, -1)
            eps_j = jnp.broadcast_to(idio_pop[j].T[:, None, :], (T_int, K, self.type_counts[j]))
            emis = (Xj[..., idx['ef']] * aj[..., 1] * aj[..., 2]
                    + Xj[..., idx['eg']] * aj[..., 1] * (1.0 - aj[..., 2])
                    + eps_j).reshape(T_int, -1)
            fa = aj.reshape(T_int, -1, 3)
            aq = jnp.quantile(fa, qs, axis=1)                    # (2,T,3)
            cq = jnp.quantile(cum, qs, axis=1)
            eq = jnp.quantile(emis, qs, axis=1)
            st = jnp.concatenate([Xj.reshape(T_int, -1, 16).mean(axis=1),
                                  Xf[j].reshape(-1, 16).mean(axis=0)[None]], axis=0)
            out['state'].append(st)
            out['action'].append(fa.mean(axis=1))
            out['action_lo'].append(aq[0]); out['action_hi'].append(aq[1])
            out['cum_r'].append(cum.mean(axis=1))
            out['cum_r_lo'].append(cq[0]); out['cum_r_hi'].append(cq[1])
            out['emission'].append(emis.mean(axis=1))
            out['emission_lo'].append(eq[0]); out['emission_hi'].append(eq[1])
            out['term_r'].append(jnp.mean(vmap(vmap(self.terminal_reward))(Xf[j])))
        for k in list(out):
            if isinstance(out[k], list):
                out[k] = tuple(out[k])
        return out

    def agent_batch_trajectory(self, policy, j, A_traj, P_traj, idio):
        """Mean trajectory of a batch of type-j agents against a frozen cleared
        price path. Same shapes as one type of ensemble_trajectory."""
        T_int = int(self.T)
        B     = idio.shape[0]
        X0    = jnp.broadcast_to(self.type_initial_state(j), (B, 16))
        idx   = self.dic_indexes

        def step_fn(carry, t_idx):
            X_t, cum = carry
            A_exec = A_traj[t_idx + 1]
            out    = vmap(policy)(vmap(self.normalize_state)(X_t))
            a_t    = vmap(self.act_at, in_axes=(0, 0, None))(X_t, out, A_exec)
            cum    = cum + vmap(self.running_reward, in_axes=(0, 0, None))(X_t, a_t, A_exec)
            eps_t  = idio[:, t_idx]
            next_X = vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                X_t, a_t, eps_t, A_exec, P_traj[t_idx + 1])
            emis = (X_t[:, idx["ef"]] * a_t[:, 1] * a_t[:, 2]
                    + X_t[:, idx["eg"]] * a_t[:, 1] * (1.0 - a_t[:, 2])
                    + eps_t)
            ys = (jnp.mean(X_t, axis=0), jnp.mean(a_t, axis=0),
                  jnp.mean(cum), jnp.mean(emis))
            return (next_X, cum), ys

        (X_final, _), (states, actions, cums, emis) = jax.lax.scan(
            step_fn, (X0, jnp.zeros(B)), jnp.arange(T_int))
        return {
            'state':    jnp.concatenate([states, jnp.mean(X_final, axis=0)[None, :]],
                                        axis=0),
            'action':   actions,
            'cum_r':    cums,
            'emission': emis,
            'term_r':   jnp.mean(vmap(self.terminal_reward)(X_final)),
        }

    def ensemble_diagnostics(self, stacked, statics, P_traj, eps0, idio_pop):
        """Same rollout as ensemble_price_path, plus per-type mean rewards and
        aggregate trade series. Used by the sanity checks, not by training."""
        Xf, A, actions, diag, Xs_hist, w = self._scan_population(
            stacked, statics, P_traj, eps0, idio_pop)
        K = Xf[0].shape[0]
        rew = vmap(vmap(vmap(self.running_reward, in_axes=(0, 0, None)),
                        in_axes=(0, 0, None)), in_axes=(0, 0, 0))
        per_type_reward = tuple(
            jnp.mean(rew(Xs_hist[j], actions[j], A).sum(axis=0)
                     + vmap(vmap(self.terminal_reward))(Xf[j]))
            for j in range(N_TYPES))
        N_tot = jnp.float32(K * int(self.agent_per_policy))
        buy  = N_tot * sum(w[j] * jnp.mean(jnp.clip(actions[j][..., 0], min=0.0),
                                           axis=(1, 2)) for j in range(N_TYPES))
        sell = N_tot * sum(w[j] * jnp.mean(-jnp.clip(actions[j][..., 0], max=0.0),
                                           axis=(1, 2)) for j in range(N_TYPES))
        return {
            'A_traj':          jnp.concatenate([jnp.atleast_1d(jnp.float32(A_REF)), A]),
            'per_type_reward': per_type_reward,
            'trade_buy':       buy,
            'trade_sell':      sell,
            'X_final':         Xf,
            'clearing':        diag,
        }
