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
        N      = jnp.float32(self.agent_per_policy)
        n_fi   = jnp.round(N * self.mm_share)
        n_prod = N - n_fi
        n_big  = jnp.round(jnp.float32(BIG_ON_PRIVATE) * n_prod)
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
        theta = base[1] * self._bm_multipliers()[j] * self.init_alloc
        return base.at[1].set(theta)

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
    def running_reward(x, a):
        prod_coef     = x[6] - x[10:12]                       # P - (cf, cg)
        quantities    = a[1] * jnp.array([a[2], 1.0 - a[2]])  # xi * (eta, 1-eta)
        prod_revenue  = jnp.dot(quantities, prod_coef)
        trade_revenue = -a[0] * x[5]                          # -f * A
        return prod_revenue + trade_revenue

    def terminal_reward(self, x):
        idx        = self.dic_indexes
        emissions  = jnp.dot(x[idx["xi"]], x[idx["e"]]) + x[idx["eps"]]
        allowances = x[idx["theta"]]
        shortage   = jnp.clip(emissions - allowances, min=0.0)
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
        """compute_next_A given the population's already-weighted mean trade.

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
        """Allowance price. h is linear, so the mean impact is h(mean trade).
        Floor only — there is deliberately no ceiling (plan.md §2.1)."""
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

    # ── rollouts ─────────────────────────────────────────────────────────────

    def ensemble_price_path(self, stacked, statics, P_traj, eps0, idio_pop):
        """Aggregate allowance price under the FP mixture.

        stacked / statics : per-type stacked parameters, leading dim K
        P_traj            : (T+1,)  frozen good-price path
        eps0              : (T,)    frozen allowance shock
        idio_pop          : tuple of (n_j, T), reused by every block (so blocks
                            share common random numbers and the aggregate is a
                            deterministic function of the policies)

        Returns A_traj of shape (T+1,).
        """
        T_int = int(self.T)
        K     = jax.tree_util.tree_leaves(stacked[0])[0].shape[0]
        X0    = self.population_initial_states(K)
        w     = self.context_weights()

        step_dyn = vmap(
            vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None)),
            in_axes=(0, 0, 0, None, None),
        )
        norm2 = vmap(vmap(self.normalize_state))
        unnorm2 = vmap(vmap(self.unnormalize_action))

        def step_fn(carry, t_idx):
            Xs, A_t = carry

            actions = []
            for j in range(N_TYPES):
                a_hat = apply_stacked(stacked[j], statics[j], norm2(Xs[j]))
                actions.append(unnorm2(Xs[j], a_hat))

            # Weighted by the CONTEXT proportions, not by the simulated counts:
            # each type contributes its own mean trade with weight w_j. At the
            # constructor proportion this is identical to the flat mean over
            # every simulated agent, since w_j = n_j / N there.
            per_type = jnp.stack([jnp.mean(a[..., 0]) for a in actions])
            A_next = self.compute_next_A_weighted(A_t, jnp.sum(w * per_type),
                                                  eps0[t_idx])
            P_next = P_traj[t_idx + 1]

            new_Xs = []
            for j in range(N_TYPES):
                eps_j = jnp.broadcast_to(idio_pop[j][:, t_idx],
                                         (K, self.type_counts[j]))
                new_Xs.append(step_dyn(Xs[j], actions[j], eps_j, A_next, P_next))

            return (tuple(new_Xs), A_next), A_next

        (_, _), A_traj = jax.lax.scan(step_fn, (X0, self.A0), jnp.arange(T_int))
        return jnp.concatenate([jnp.atleast_1d(self.A0), A_traj])

    def rollout_agent_batch(self, policy, j, A_traj, P_traj, idio):
        """A batch of type-j agents responding to a frozen price path.

        This is the *only* rollout used for both best-response training and the
        exploitability metric, so the two are computed on identical footing.

        policy : a TypePolicyNN (or any callable (8,) -> (3,))
        idio   : (B, T) frozen idiosyncratic draws
        Returns (X_final (B,16), total_reward (B,)).
        """
        T_int = int(self.T)
        B     = idio.shape[0]
        X0    = jnp.broadcast_to(self.type_initial_state(j), (B, 16))

        def step_fn(carry, t_idx):
            X_t, cum = carry
            a_hat  = vmap(policy)(vmap(self.normalize_state)(X_t))
            a_t    = vmap(self.unnormalize_action)(X_t, a_hat)
            r_t    = vmap(self.running_reward)(X_t, a_t)
            next_X = vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                X_t, a_t, idio[:, t_idx], A_traj[t_idx + 1], P_traj[t_idx + 1]
            )
            return (next_X, cum + r_t), None

        (X_final, cum), _ = jax.lax.scan(
            step_fn, (X0, jnp.zeros(B)), jnp.arange(T_int)
        )
        return X_final, cum + vmap(self.terminal_reward)(X_final)

    def ensemble_trajectory(self, stacked, statics, P_traj, eps0, idio_pop,
                            q_lo=0.1, q_hi=0.9):
        """Per-type trajectories of the FP mixture, with dispersion bands.

        Only summary statistics over agents are returned, so the arrays stay
        tiny even for large K.

        The spread is taken across all K*n_j agents of a type, which mixes two
        sources: different FP blocks play different policies, and agents differ
        in their idiosyncratic draw. That is the informative quantity — a
        converged mixture has every block behaving alike, so the band narrowing
        over FP iterations is itself a convergence signal.

        Returns dict with, per type j:
            state[j]      (T+1, 16)  mean state
            action[j]     (T, 3)     mean action, and action_lo/action_hi
            cum_r[j]      (T,)       mean cumulative running reward, + lo/hi
            emission[j]   (T,)       mean per-step emissions, + lo/hi
            term_r[j]     scalar     mean terminal reward
        """
        idx   = self.dic_indexes
        T_int = int(self.T)
        K     = jax.tree_util.tree_leaves(stacked[0])[0].shape[0]
        X0    = self.population_initial_states(K)
        w     = self.context_weights()
        qs    = jnp.array([q_lo, q_hi])

        step_dyn = vmap(
            vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None)),
            in_axes=(0, 0, 0, None, None),
        )
        norm2   = vmap(vmap(self.normalize_state))
        unnorm2 = vmap(vmap(self.unnormalize_action))
        rew2    = vmap(vmap(self.running_reward))

        def step_fn(carry, t_idx):
            Xs, A_t, cums = carry

            actions = []
            for j in range(N_TYPES):
                a_hat = apply_stacked(stacked[j], statics[j], norm2(Xs[j]))
                actions.append(unnorm2(Xs[j], a_hat))

            cums   = tuple(cums[j] + rew2(Xs[j], actions[j]) for j in range(N_TYPES))
            per_type = jnp.stack([jnp.mean(a[..., 0]) for a in actions])
            A_next   = self.compute_next_A_weighted(A_t, jnp.sum(w * per_type),
                                                    eps0[t_idx])
            P_next = P_traj[t_idx + 1]

            eps_all, new_Xs = [], []
            for j in range(N_TYPES):
                eps_j = jnp.broadcast_to(idio_pop[j][:, t_idx],
                                         (K, self.type_counts[j]))
                eps_all.append(eps_j)
                new_Xs.append(step_dyn(Xs[j], actions[j], eps_j, A_next, P_next))

            # per-step emissions, per agent. Computed directly rather than by
            # differencing a mean, because E[xi*eta] != E[xi]*E[eta] and we want
            # quantiles of the per-agent flow, not differences of quantiles.
            emis = []
            for j in range(N_TYPES):
                a1, a2 = actions[j][..., 1], actions[j][..., 2]
                emis.append(Xs[j][..., idx["ef"]] * a1 * a2
                            + Xs[j][..., idx["eg"]] * a1 * (1.0 - a2)
                            + eps_all[j])

            flat_a = [actions[j].reshape(-1, 3) for j in range(N_TYPES)]
            flat_c = [cums[j].reshape(-1)       for j in range(N_TYPES)]
            flat_e = [emis[j].reshape(-1)       for j in range(N_TYPES)]

            ys = {
                'A':          A_next,
                'state':      tuple(jnp.mean(Xs[j].reshape(-1, 16), axis=0)
                                    for j in range(N_TYPES)),
                'action':     tuple(jnp.mean(flat_a[j], axis=0) for j in range(N_TYPES)),
                'action_q':   tuple(jnp.quantile(flat_a[j], qs, axis=0)
                                    for j in range(N_TYPES)),
                'cum_r':      tuple(jnp.mean(flat_c[j]) for j in range(N_TYPES)),
                'cum_r_q':    tuple(jnp.quantile(flat_c[j], qs) for j in range(N_TYPES)),
                'emission':   tuple(jnp.mean(flat_e[j]) for j in range(N_TYPES)),
                'emission_q': tuple(jnp.quantile(flat_e[j], qs) for j in range(N_TYPES)),
            }
            return (tuple(new_Xs), A_next, cums), ys

        init_cums = tuple(jnp.zeros((K, self.type_counts[j])) for j in range(N_TYPES))
        (Xf, _, _), ys = jax.lax.scan(
            step_fn, (X0, self.A0, init_cums), jnp.arange(T_int)
        )

        final_mean = tuple(jnp.mean(Xf[j].reshape(-1, 16), axis=0)
                           for j in range(N_TYPES))
        term = tuple(jnp.mean(vmap(vmap(self.terminal_reward))(Xf[j]))
                     for j in range(N_TYPES))

        return {
            'A_traj': jnp.concatenate([jnp.atleast_1d(self.A0), ys['A']]),
            'P_traj': P_traj,
            'state':  tuple(jnp.concatenate([ys['state'][j], final_mean[j][None, :]],
                                            axis=0) for j in range(N_TYPES)),
            'action':     ys['action'],
            'action_lo':  tuple(ys['action_q'][j][:, 0, :] for j in range(N_TYPES)),
            'action_hi':  tuple(ys['action_q'][j][:, 1, :] for j in range(N_TYPES)),
            'cum_r':      ys['cum_r'],
            'cum_r_lo':   tuple(ys['cum_r_q'][j][:, 0] for j in range(N_TYPES)),
            'cum_r_hi':   tuple(ys['cum_r_q'][j][:, 1] for j in range(N_TYPES)),
            'emission':   ys['emission'],
            'emission_lo': tuple(ys['emission_q'][j][:, 0] for j in range(N_TYPES)),
            'emission_hi': tuple(ys['emission_q'][j][:, 1] for j in range(N_TYPES)),
            'term_r':     term,
        }

    def agent_batch_trajectory(self, policy, j, A_traj, P_traj, idio):
        """Mean trajectory of a batch of type-j agents against a frozen price
        path. Same shapes as one type of ensemble_trajectory."""
        T_int = int(self.T)
        B     = idio.shape[0]
        X0    = jnp.broadcast_to(self.type_initial_state(j), (B, 16))

        idx = self.dic_indexes

        def step_fn(carry, t_idx):
            X_t, cum = carry
            a_hat  = vmap(policy)(vmap(self.normalize_state)(X_t))
            a_t    = vmap(self.unnormalize_action)(X_t, a_hat)
            cum    = cum + vmap(self.running_reward)(X_t, a_t)
            eps_t  = idio[:, t_idx]
            next_X = vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                X_t, a_t, eps_t, A_traj[t_idx + 1], P_traj[t_idx + 1]
            )
            emis = (X_t[:, idx["ef"]] * a_t[:, 1] * a_t[:, 2]
                    + X_t[:, idx["eg"]] * a_t[:, 1] * (1.0 - a_t[:, 2])
                    + eps_t)
            ys = (jnp.mean(X_t, axis=0), jnp.mean(a_t, axis=0),
                  jnp.mean(cum), jnp.mean(emis))
            return (next_X, cum), ys

        (X_final, _), (states, actions, cums, emis) = jax.lax.scan(
            step_fn, (X0, jnp.zeros(B)), jnp.arange(T_int)
        )
        return {
            'state':    jnp.concatenate([states, jnp.mean(X_final, axis=0)[None, :]],
                                        axis=0),
            'action':   actions,
            'cum_r':    cums,
            'emission': emis,
            'term_r':   jnp.mean(vmap(self.terminal_reward)(X_final)),
        }

    def ensemble_diagnostics(self, stacked, statics, P_traj, eps0, idio_pop):
        """Same rollout as ensemble_price_path, but also returns per-type mean
        rewards and aggregate trade/production series. Used by the sanity
        checks, not by training."""
        T_int = int(self.T)
        K     = jax.tree_util.tree_leaves(stacked[0])[0].shape[0]
        X0    = self.population_initial_states(K)
        w     = self.context_weights()

        step_dyn = vmap(
            vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None)),
            in_axes=(0, 0, 0, None, None),
        )
        norm2   = vmap(vmap(self.normalize_state))
        unnorm2 = vmap(vmap(self.unnormalize_action))
        rew2    = vmap(vmap(self.running_reward))

        def step_fn(carry, t_idx):
            Xs, A_t, cums = carry

            actions = []
            for j in range(N_TYPES):
                a_hat = apply_stacked(stacked[j], statics[j], norm2(Xs[j]))
                actions.append(unnorm2(Xs[j], a_hat))

            cums   = tuple(cums[j] + rew2(Xs[j], actions[j]) for j in range(N_TYPES))
            per_type = jnp.stack([jnp.mean(a[..., 0]) for a in actions])
            A_next   = self.compute_next_A_weighted(A_t, jnp.sum(w * per_type),
                                                    eps0[t_idx])
            P_next = P_traj[t_idx + 1]

            new_Xs = []
            for j in range(N_TYPES):
                eps_j = jnp.broadcast_to(idio_pop[j][:, t_idx],
                                         (K, self.type_counts[j]))
                new_Xs.append(step_dyn(Xs[j], actions[j], eps_j, A_next, P_next))

            # weighted market-wide totals: sum_i f_i = N_total * sum_j w_j *
            # mean_{i in j} f_i, so replacing n_j/N by w_j is the mean-field
            # aggregate at the same scale as the old plain sum
            N_tot = jnp.float32(K * int(self.agent_per_policy))
            buy  = N_tot * jnp.sum(w * jnp.stack(
                [jnp.mean(jnp.clip(a[..., 0], min=0.0)) for a in actions]))
            sell = N_tot * jnp.sum(w * jnp.stack(
                [jnp.mean(-jnp.clip(a[..., 0], max=0.0)) for a in actions]))
            return (tuple(new_Xs), A_next, cums), (A_next, buy, sell)

        init_cums = tuple(jnp.zeros((K, self.type_counts[j])) for j in range(N_TYPES))
        (Xf, _, cums), (A_traj, buy, sell) = jax.lax.scan(
            step_fn, (X0, self.A0, init_cums), jnp.arange(T_int)
        )

        term = tuple(vmap(vmap(self.terminal_reward))(Xf[j]) for j in range(N_TYPES))
        per_type_reward = tuple(
            jnp.mean(cums[j] + term[j]) for j in range(N_TYPES)
        )
        return {
            'A_traj':          jnp.concatenate([jnp.atleast_1d(self.A0), A_traj]),
            'per_type_reward': per_type_reward,
            'trade_buy':       buy,
            'trade_sell':      sell,
            'X_final':         Xf,
        }
