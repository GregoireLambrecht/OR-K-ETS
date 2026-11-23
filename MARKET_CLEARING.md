# Market clearing — implementation instructions

Branch `market-clearing`, cut from `main` at `4b0d6af`.

This branch replaces the reduced-form price impact

$$
A_{t+1} = \max\big(A_t + h(\bar f_t)\,dt + \varepsilon^{0,2}_t \sqrt{dt},\ A^{\text{floor}}\big)
$$

(`compute_next_A_weighted` in `envs/environment.py`) by a **market-clearing price**.
At each day $t$, every agent of type $j$ submits a demand schedule (a trade as a
function of the price), and $A_t$ is the price at which the aggregate trade equals
the exogenous net supply. When no price inside the corridor clears the market, the
government absorbs the imbalance at the floor or at the ceiling.

Why: under the impact rule, net purchases are absorbed by unmodelled market makers.
The cap is therefore not a cap, and the price can carry a predictable drift that
every agent trades on. With clearing, the allowances in circulation are conserved,
so the cap is the total free allocation plus net supply. Fictitious play, the
best-response training and the scenario extension are unchanged. Only the price
step of the population simulation and the trade head of the network change.

Indices: the paper's types $j = 1, 2, 3$ are `j = 0, 1, 2` in the code (private
producer, large public producer, financial institution / market maker). Particle
$i$ of type $j$ following mixture component $l$ is written $(j,i,l)$.

---

## 1. The model

### 1.1 Demand schedule of type $j$

A single network per type $j$ and component $l$, with parameters $\vartheta^{j,l}$, is
evaluated on the **pre-trade** state of particle $(j,i,l)$. Its outputs do **not**
depend on the candidate price $a$:

$$
\big(\alpha^{j,i,l}_t,\ \beta^{j,i,l}_t,\ \hat\xi^{j,i,l}_t,\ \hat\eta^{j,i,l}_t\big)
= \Phi_{\vartheta^{j,l}}\big(\bar x^{j,i,l}_t;\ \bar\gamma\big),
\qquad \beta^{j,i,l}_t = \beta_{\min} + \operatorname{softplus}(\cdot) > 0 .
$$

The trade at a candidate price $a$ is

$$
f^{j,i,l}_t(a) = \operatorname{clip}\Big( C^{f,j}\, \tanh\Big(\alpha^{j,i,l}_t - \beta^{j,i,l}_t\, \frac{a - A^{\text{pre}}_t}{A_{\text{scale}}}\Big),\ -\min\big(\theta^{j,i,l}_t, C^{f,j}\big),\ C^{f,j} \Big),
$$

where $A^{\text{pre}}_t$ is the last cleared price, already in the state.
Centering on $A^{\text{pre}}_t$ is only a reparametrization: $A^{\text{pre}}_t$ is
pre-trade, so the family is still $\tanh(\text{const} - \beta a)$. It gives a neutral
initialization: a freshly initialized network ($\alpha \approx 0$) barely trades at
an unchanged price. The reservation price of the particle is

$$
R^{j,i,l}_t = A^{\text{pre}}_t + A_{\text{scale}}\, \alpha^{j,i,l}_t / \beta^{j,i,l}_t .
$$

For **any** network outputs, $a \mapsto f^{j,i,l}_t(a)$ is continuous and
nonincreasing, since $\beta > 0$ and the clip bounds are pre-trade. Do **not** feed $a$,
$A_t$ or $\tilde A_t$ to the network: that breaks monotonicity and multiplies the
clearing cost by the number of bisection iterations.

Production and the fuel share stay as they are, computed from the pre-trade state
(v1 choice, see §8).

### 1.2 Aggregate excess demand

With the context weights $w_j$ (`context_weights()`, a function of $\mathfrak n_3$ in $\gamma$)
and $k$ components in the mixture,

$$
D_t(a) = \sum_{j} w_j \, \frac{1}{k\, n_j} \sum_{l=0}^{k-1} \sum_{i=1}^{n_j} f^{j,i,l}_t(a) \;-\; S_t ,
\qquad S_t = \sigma_S\, \varepsilon^{0,2}_t \sqrt{dt} .
$$

$S_t$ is the net supply of exogenous traders, in allowances per unit mass of
agents. It replaces the additive price shock: `eps0` keeps its shape and seeds, and
only its meaning changes. $D_t$ is continuous and nonincreasing in $a$.

### 1.3 Clearing condition

The price $A_t \in [A^{\text{floor}}, A^{\text{ceil}}]$ satisfies

$$
D_t(A_t) \in N_{[A^{\text{floor}}, A^{\text{ceil}}]}(A_t),
\qquad
N_{[l,u]}(a) = \begin{cases} \{0\} & l < a < u \\ (-\infty, 0] & a = l \\ [0, \infty) & a = u \end{cases}
$$

- $D_t(A_t) = 0$ in the interior.
- At the floor, $D_t(A_t) \le 0$, and the government buys $G_t = -D_t(A_t) \ge 0$.
- At the ceiling, $D_t(A_t) \ge 0$, and the government sells $-G_t = D_t(A_t) \ge 0$.

**Existence** only needs continuity and a compact corridor, so the ceiling must be
finite. The study has no ceiling, so use an artificial one, `A_CEIL`, and log how
often it binds (it should be about 0).

**Selection.** Let
$a^-_t = \sup\{a : D_t(a) > 0\}$ (equal to $A^{\text{floor}}$ if the set is empty) and
$a^+_t = \inf\{a : D_t(a) < 0\}$ (equal to $A^{\text{ceil}}$ if the set is empty).
By monotonicity, the solution set is exactly $[a^-_t, a^+_t]$. Take

$$
A_t = \operatorname{clip}\big(A^{\text{pre}}_t,\ a^-_t,\ a^+_t\big):
$$

the price does not move if the previous one still clears, and otherwise it moves
as little as possible. This single formula also covers the floor and ceiling cases.

**Conservation (the cap).** With $\bar\theta_t = \sum_j w_j \cdot \text{mean}_{i,l}\, \theta^{j,i,l}_t$,

$$
\bar\theta_{t+1} = \bar\theta_t + S_t + G_t \qquad \text{(exactly, up to the bisection tolerance).}
$$

### 1.4 Timing and indexing (keep the current array conventions)

Keep `A_traj` of shape `(T+1,)`, with a new meaning:

| index | meaning |
|---|---|
| `A_traj[0]` | $A_{\text{ref}}$: the pre-open price, used only as $A^{\text{pre}}_0$ (tie-breaker and network input). Use `A0_BASE` = 6.05. |
| `A_traj[t+1]` | $A_t$, the price **cleared at step $t$**, at which the trades of step $t$ are executed |

Consequences, most of which hold already:

- State slot `A` (`x[5]`) at step $t$ holds `A_traj[t]` = $A^{\text{pre}}_t$. This is
  what `single_step_dynamics(..., next_A=A_traj[t+1], ...)` already writes. No change.
- The running mean in slot `A_avg` already averages `A_traj[1..t+1]`. At $T$ it is the
  mean of the $T$ cleared prices, which is the statutory K-ETS reference. No change.
- **Change:** trade revenue is $-f_t \cdot A_t$ with $A_t$ = `A_traj[t+1]`, not
  $-f_t \cdot$ `x[5]`. `running_reward` must receive the execution price.
- $A_0$ (cleared) is now an **output**. Whether it is close to 6.05 at
  $\gamma^{\text{ref}}$ is a calibration check, not an input.

---

## 2. Code changes, file by file

### 2.1 `envs/base_params.py`

Add:

```python
# ── market clearing ──────────────────────────────────────────────────────────
A_REF        = A0_BASE        # pre-open price: A^pre_0, tie-breaker only
A_CEIL       = 200.0          # artificial ceiling (USD); must essentially never bind
BETA_MIN     = 0.1            # floor on the schedule slope (normalised price units)
CLEAR_ITERS  = 40             # bisection halvings per side; 40 halvings of ~200 -> ~2e-10
SIGMA_S      = 0.05 * CAP_F_BASE * MEAN_MARKET_IMPACT   # exogenous net-supply scale, TO CALIBRATE

def supply_noise_base(key, T):
    """Standard normal draws; scaled by SIGMA_S * sqrt(dt) in the clearing step."""
    return jax.random.normal(key, (T,))
```

Keep `market_impact_base` and `white_noise_A_base` so `main` still runs, but do
not use them on this branch. Comment the old note "deliberately no ceiling" to say
that the ceiling is now an artificial bound for existence only.

### 2.2 `envs/models.py`

`ContextActionNN`: the output layer goes from 3 to **4** outputs, and the squashing
changes:

```python
out = self.trunk_layers[self.n_hidden](h)
return jnp.stack([
    out[0],                                         # alpha   (unbounded)
    BETA_MIN + jax.nn.softplus(out[1]),             # beta    > 0
    jax.nn.sigmoid(out[2]),                         # xi_hat  in [0, 1]
    jax.nn.sigmoid(out[3]),                         # eta_hat in [0, 1]
])
```

- Set `action_dim=4` as the default for `ContextActionNN` / `BoundContextNN`.
  `TypePolicyNN` (rungs 1–2) stays untouched.
- Initialize the bias of `out[1]` so that $\beta \approx 1$ at start, i.e.
  softplus$^{-1}(0.9) \approx 0.37$.
- Checkpoints from `main` are **not loadable** (different output shape). Always use
  new result directories, e.g. `results_mc_floor30/choice`.

### 2.3 `envs/environment.py`

Add three pure functions on the env:

```python
def schedule_params(self, x, out):
    """(16,) state, (4,) net output -> (alpha, beta, lo, hi, xi, eta).
    lo/hi are the pre-trade clip bounds; xi/eta are unnormalised as before."""
    cap_f = x[idx['cap_f']]; theta = x[idx['theta']]
    lo, hi = -jnp.minimum(theta, cap_f), cap_f
    xi  = out[2] * x[idx['cap_xi']]
    eta = jnp.where(x[idx['tech']] >= 0.0, x[idx['tech']], out[3])
    return out[0], out[1], lo, hi, xi, eta

def trade_at(self, x, alpha, beta, lo, hi, a):
    """f(a) for one particle; nonincreasing and continuous in a."""
    z = alpha - beta * (a - x[idx['A']]) / self.A_scale
    return jnp.clip(x[idx['cap_f']] * jnp.tanh(z), lo, hi)

def clear_price(self, D, A_pre):
    """D: scalar a -> excess demand, nonincreasing. Returns (A_t, a_minus, a_plus).
    Two fixed-length bisections (lax.fori_loop, jit/vmap safe)."""
```

Bisection for `clear_price`, with `lo_ = self.Afloor`, `hi_ = A_CEIL`:

```python
def left(_, lh):                    # a_minus = sup{a : D(a) > 0}; invariant D(l) > 0 >= D(h)
    l, h = lh; m = 0.5 * (l + h); pos = D(m) > 0
    return jnp.where(pos, m, l), jnp.where(pos, h, m)
def right(_, lh):                   # a_plus = inf{a : D(a) < 0}; invariant D(l) >= 0 > D(h)
    l, h = lh; m = 0.5 * (l + h); neg = D(m) < 0
    return jnp.where(neg, l, m), jnp.where(neg, m, h)

d_lo, d_hi = D(lo_), D(hi_)
a_minus = jnp.where(d_lo <= 0, lo_, jnp.where(d_hi > 0, hi_,
          jax.lax.fori_loop(0, CLEAR_ITERS, left,  (lo_, hi_))[1]))
a_plus  = jnp.where(d_hi >= 0, hi_, jnp.where(d_lo < 0, lo_,
          jax.lax.fori_loop(0, CLEAR_ITERS, right, (lo_, hi_))[0]))
A_t = jnp.clip(A_pre, a_minus, a_plus)
```

`D(a)` is closed form once $(\alpha, \beta, lo, hi)$ are computed: a weighted mean of
clipped tanh functions over the `(K, n_j)` particles of each type. **One** network
forward pass per step; the bisection is pure arithmetic.

**Factor out one population step.** `ensemble_price_path`, `ensemble_trajectory` and
`ensemble_diagnostics` currently repeat the same step logic. Replace it with a single
private method, so the three cannot drift apart:

```python
def _population_step(self, stacked, statics, Xs, t_idx, P_traj, S_noise, idio_pop, w):
    # 1. one forward pass per type on the pre-trade states
    # 2. schedule_params per particle
    # 3. D(a) = sum_j w_j * mean_{l,i} trade_at(...) - SIGMA_S * S_noise[t] * sqrt(dt)
    # 4. A_t, a_minus, a_plus = clear_price(D, A_pre = Xs[0][0, 0, idx['A']])
    # 5. f = trade_at(..., A_t); actions = (f, xi, eta)
    # 6. new_Xs = step_dyn(Xs, actions, eps, next_A=A_t, next_P=P_traj[t+1])
    # returns new_Xs, A_t, actions, diag = {D(A_t), a_minus, a_plus}
```

Then:

- `ensemble_price_path`: scan `_population_step`, return
  `concat([A_REF], A_cleared)` of shape `(T+1,)`. Add an optional
  `return_diag=False` that also returns per-step `G_t = -D(A_t)`, the floor and
  ceiling flags, and the flat flag `a_plus - a_minus > tol`.
- `ensemble_trajectory`, `ensemble_diagnostics`: same step; the running reward
  uses `A_t`.
- `running_reward(x, a, A_exec)`: `trade_revenue = -a[0] * A_exec`.
- `rollout_agent_batch` and `agent_batch_trajectory` (the deviator): the network
  output gives the schedule; evaluate it at `A_exec = A_traj[t_idx + 1]`, then reward
  and dynamics as before. Shapes and signatures are unchanged, so every caller in
  `utils.py` keeps working.
- `create_initial_state`: slot `A` is $A_{\text{ref}}$ (pass `A_REF`; same value as
  today).
- `compute_next_A`, `compute_next_A_weighted`: no longer called. Keep them, marked
  legacy.
- `reward_scale`: uses `self.A0` = `A0_BASE`, a **constant**, so leave it. It is a
  fixed scale, not the opening price.

### 2.4 `utils.py` (rung 3 only)

- `make_price_fn_r3`, `make_batched_return_fn_r3`, `make_batched_hist_fn_r3`,
  `make_br_stage_fn_r3`: no signature change. They call the env rollouts. In
  `sample_market`, `eps0` now comes from `supply_noise_base`; it goes through
  `env_T.generate_eps0`, so this is a constructor change in the entry points (§2.5).
- `eval_block`: compute floor statistics on `a[:, 1:]` (cleared prices only) and add:
  `A_on_ceil`, `clear_residual_max` (max $|D_t(A_t)|$ over interior steps,
  relative to $\sum_j w_j C^{f,j}$), `clear_flat_frac`, `gov_flow_mean`,
  `price_drift` (mean and std of $A_{t+1} - A_t$), and the mean reservation price per
  type $j$. Use `return_diag=True` on the block's price evaluation.
- `fictitious_play_rung3`: no logic change. `policy_0` is a fresh network, which with
  the centered parametrization is close to "no trade at an unchanged price".

### 2.5 Entry points: `main_rung3.py`, `pretrain_mm_r3.py`, `audit_pmfg.py`, `audit_paths.py`

- Pass `generate_eps0_func=supply_noise_base`. `market_impact_func` becomes
  unused; pass it, but it is ignored.
- Default output directories get the `results_mc_` prefix.
- `pretrain_mm_r3.py`: the target heuristic `target_hat_f` is unchanged. The fit
  becomes: make the schedule evaluated at the pre-trade price match the target,

  $$
  \min_\vartheta \ \mathbb E\Big[\big(\tanh(\alpha_\vartheta) - \hat f_{\text{target}}\big)^2\Big],
  $$

  since at $a = A^{\text{pre}}$ the centered schedule is $\tanh(\alpha)$. Also add a
  weak penalty pulling $\beta$ toward 1. The zero-gradient trap is still there (type 2
  starts at $\theta = 0$, so the clip lower bound is 0 and the gradient dies when
  $\alpha - \beta(\cdot) < 0$), so the warm start remains necessary.
- `helper_plot.py`, `diagnostics.py`: relabel "allowance shock" as "net supply";
  the price panels plot `A_traj[1:]` (cleared prices).

---

## 3. Tests — `tests/test_clearing.py` (pytest, runs on CPU)

1. **Single schedule.** All particles share $\alpha, \beta$, no clip binds, and $S_t = 0$:
   then $A_t = R = A^{\text{pre}} + A_{\text{scale}}\,\alpha/\beta$, to `1e-6`.
2. **Floor.** Every particle sells at every price: $A_t = A^{\text{floor}}$ and
   $G_t = -D_t(A^{\text{floor}}) > 0$.
3. **Ceiling.** Every particle buys at every price: $A_t$ = `A_CEIL` and $G_t < 0$.
4. **Flat region.** Everyone is clipped at the buy cap and $S_t$ equals total demand:
   $a^- < a^+$, and $A_t = \operatorname{clip}(A^{\text{pre}}, a^-, a^+) = A^{\text{pre}}$ when $A^{\text{pre}}$ is inside.
5. **Monotonicity.** For random network outputs, $D_t$ evaluated on a grid of 1000
   prices is nonincreasing.
6. **Conservation.** Over a full `ensemble_trajectory`,
   $\bar\theta_T - \bar\theta_0 = \sum_t (S_t + G_t)$ to relative `1e-5`.
7. **Shapes and jit.** `ensemble_price_path` returns `(T+1,)`, `A_traj[0] == A_REF`;
   `make_price_fn_r3` and `make_br_stage_fn_r3` compile under `vmap` at `T=10`.
8. **Deviator consistency.** A deviator holding the same parameters as a
   single-component mixture gets the same return as that component's particles,
   up to idiosyncratic noise.

## 4. Smoke run before the 30-hour solve

```bash
KETS_FLOOR_MAX=30 python3 main_rung3.py choice --model-dir results_mc_smoke \
    --nb-policy 5 --t-grid 50 --iters 1000 --batch 128 \
    --verify-samples 100 --fixed-samples 50 --eval-samples 16
```

Accept when:
- the tests pass;
- `clear_residual_max` < 1e-6;
- `A_on_ceil` ≈ 0;
- exploitability decreases over the 5 iterations;
- no NaN appears in $\alpha$ or $\beta$.

## 5. Diagnostics to report (progress.json, per block)

| key | meaning | expected |
|---|---|---|
| `A_on_floor`, `A_on_ceil` | fraction of cleared steps at each bound | floor: depends on κ; ceil ≈ 0 |
| `clear_residual_max` | max interior $\lvert D_t(A_t) \rvert$, relative | < 1e-6 |
| `clear_flat_frac` | fraction of steps with $a^- < a^+$ | small |
| `gov_flow_mean` | mean $G_t$ (government purchases) | > 0 only when on the floor |
| `price_drift` | mean $A_{t+1} - A_t$ | ≈ 0 (near-martingale), except from the $C^{f,j}$ caps |
| `R_mean[j]` | mean reservation price of type $j$ | brackets $A_t$ |
| `A0_cleared` | cleared price at $t = 0$ | compare with 6.05 at $\gamma^{\text{ref}}$ |

**Sanity check on the κ switch.** For $\kappa^2 < 1$ the price should sit on the floor.
For $\kappa^2 > 1$, the heuristic
$A_{T-1} \approx \kappa^2 \tilde A_T\, \mathbb P(\text{aggregate short})$ predicts
$\mathbb P(\text{aggregate short}) \approx 1/\kappa^2$. Log the fraction of rows in which
the market as a whole ends short.

## 6. Phases

1. **Core** (§2.1–2.3 and the tests): params, network head, clearing, rollouts.
2. **FP and warm start** (§2.4–2.5): the smoke run of §4, then the full solves on
   `results_mc_floor{20,30,40}`.
3. **Unrestricted audit.** In `audit_pmfg.py`, add `--unrestricted`: the specialist
   deviator of type $j$ is a network with input `(normalised state, A_exec / A_scale)`
   that outputs $f$ directly (tanh times $C^{f,j}$, clipped). It never takes part in
   clearing, so it needs no monotonicity. Compare its gain with the restricted
   specialist: if they match, the schedule form is not binding.
4. **Price bank (performance).** The population is frozen during a best response.
   Once per FP iteration, clear `N_bank` (about 1e5) markets with the chunked
   `price_fn`. Inside `make_br_stage_fn_r3`, sample minibatches of
   (γ, P, A) from the bank instead of re-simulating the population at every
   gradient step. The deviator's idiosyncratic noise stays fresh.
5. **Robustness (optional, for the paper).**
   - $C^{f,j} \times \{0.5, 2\}$ at $\gamma^{\text{ref}}$ and two more scenarios.
   - A $K$-step schedule, $\sum_{k} w_k \tanh(\alpha_k - \beta_k \cdot)$ with $w_k \ge 0$, $\sum_k w_k = 1$, for any type $j$ where the audit of phase 3 shows a gap.
   - A quadratic trading cost variant: a linear schedule $(R^j - a)/\lambda_j$ and position limit $\Theta^3$.

## 7. Paper changes this implies

- §2.1: eq. (price_dyn) and the impact function $h$ are replaced by the clearing condition (§1.3 here); the noise $\varepsilon^{0,2}$ becomes net supply; $A_0$ is endogenous.
- Definition of a control: the trade component is a demand schedule (§1.1).
- §3: the particle step (particle_h) becomes the clearing step. FP and the best responses are unchanged.
- §4: report the diagnostics of §5, and the conservation of allowances.

## 8. Open choices (decide before phase 2)

- **`SIGMA_S`**: the scale of exogenous net supply. Calibrate to K-ETS daily
  volumes, which set the liquidity the old $h$ stood for.
- **`A_CEIL`**: high enough never to bind. If it binds, it is doing economic work
  and must be reported.
- **Production**: v1 decides $\xi, \eta$ from the pre-trade state (one forward pass). v2 would
  feed the cleared $A_t$ to the production heads (a second, cheap forward pass
  after clearing). The trade head must stay price-free either way.
- **`A_REF`**: 6.05 (the observed price) is used only as the pre-open value and
  tie-breaker.
