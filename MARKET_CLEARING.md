# Market clearing — implementation instructions

Branch `market-clearing`, cut from `main` at `4b0d6af`.

This branch replaces the reduced-form price impact

$$
A_{t+1} = \max\big(A_t + h(\bar f_t)\,dt + \varepsilon^{0,2}_t \sqrt{dt},\ A^{\text{floor}}\big)
$$

(`compute_next_A_weighted` in `envs/environment.py`) by a **market-clearing price**.
At each day $t$, every agent of type $j$ submits a demand schedule, i.e. a trade as a
function of the price. Exogenous traders supply a price-elastic net quantity, and the
price is the one at which aggregate trade balances.

The **price floor blocks the price; the government does not trade.** When the
clearing price would fall below the floor, trades happen at the floor, buyers are
served in full and sellers are rationed (short-side rule).

Why: under the impact rule, net purchases are absorbed by unmodelled market makers.
The cap is therefore not a cap, and the price can carry a predictable drift that
every agent trades on. With clearing, the allowances in circulation are conserved
exactly, so the cap is the total free allocation plus the exogenous net supply.
Fictitious play, the best-response training and the scenario extension are unchanged.
Only the price step of the population simulation, the execution of trades and the
trade head of the network change.

Indices: the paper's types $j = 1, 2, 3$ are `j = 0, 1, 2` in the code (private
producer, large public producer, financial institution / market maker). Particle
$i$ of type $j$ following mixture component $l$ is written $(j,i,l)$. Network
parameters are $\vartheta^{j,l}$ (not $\theta$, which is allowance holdings).

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

The order of the particle at a candidate price $a$ is

$$
f^{j,i,l}_t(a) = \operatorname{clip}\Big( C^{f,j}\, \tanh\Big(\alpha^{j,i,l}_t - \beta^{j,i,l}_t\, \frac{a - A^{\text{pre}}_t}{A_{\text{scale}}}\Big),\ -\min\big(\theta^{j,i,l}_t, C^{f,j}\big),\ C^{f,j} \Big),
$$

where $A^{\text{pre}}_t$ is the last price, already in the state. Centering on
$A^{\text{pre}}_t$ is only a reparametrization: $A^{\text{pre}}_t$ is pre-trade, so the
family is still $\tanh(\text{const} - \beta a)$. It gives a neutral initialization: a
fresh network ($\alpha \approx 0$) barely trades at an unchanged price. The
reservation price of the particle is

$$
R^{j,i,l}_t = A^{\text{pre}}_t + A_{\text{scale}}\, \alpha^{j,i,l}_t / \beta^{j,i,l}_t .
$$

For **any** network outputs, $a \mapsto f^{j,i,l}_t(a)$ is continuous and
nonincreasing, since $\beta > 0$ and the clip bounds are pre-trade. Do **not** feed $a$,
$A_t$ or $\tilde A_t$ to the network: that breaks monotonicity and multiplies the
clearing cost by the number of bisection iterations.

Production and the fuel share stay as they are, computed from the pre-trade state
(v1 choice, see §8).

### 1.2 Exogenous traders: price-elastic net supply

$$
S_t(a) = \sigma_S\, \varepsilon^{0,2}_t \sqrt{dt} + s_1\,\big(a - A_{\text{ref}}\big), \qquad s_1 > 0 .
$$

$S_t(a)$ is in allowances per unit mass of agents. It is a net supply: positive means
the exogenous traders sell, negative means they buy. In the rationing step they are
treated as one more participant, with order $f^{S}_t(a) = -S_t(a)$.
`eps0` keeps its shape and seeds; only its meaning changes, from a price shock to a
supply shock. $s_1$ is the market depth: it plays the role the old impact function
$h$ stood for, and is calibrated to K-ETS volumes.

$s_1 > 0$ is **required**. It makes the excess demand strictly decreasing, so the
clearing price is unique, there are no flat regions, and no artificial ceiling is
needed.

### 1.3 Excess demand and the unconstrained clearing price

With the context weights $w_j$ (`context_weights()`, a function of $\mathfrak n_3$ in $\gamma$)
and $k$ components in the mixture,

$$
D_t(a) = \sum_{j} w_j \, \frac{1}{k\, n_j} \sum_{l=0}^{k-1} \sum_{i=1}^{n_j} f^{j,i,l}_t(a) \;-\; S_t(a) .
$$

$D_t$ is continuous and strictly decreasing, with slope at most $-s_1$. The
unconstrained clearing price $A^\star_t$ is its unique root. A bracket is explicit: since
every order is at most $C^{f,j}$,

$$
A^\star_t \le A^{\text{hi}}_t \coloneqq A_{\text{ref}} + \frac{\sum_j w_j C^{f,j} + \sigma_S\, \lvert \varepsilon^{0,2}_t \rvert \sqrt{dt}}{s_1} .
$$

### 1.4 The price floor blocks the price: rationing (short-side rule)

$$
A_t = \max\big(A^{\text{floor}},\ A^\star_t\big).
$$

- **The floor does not bind** ($D_t(A^{\text{floor}}) \ge 0$, i.e. $A^\star_t \ge A^{\text{floor}}$):
  the market clears at $A_t = A^\star_t$ and every order is executed in full.
- **The floor binds** ($D_t(A^{\text{floor}}) < 0$): trades happen at $A_t = A^{\text{floor}}$.
  Every order, including the exogenous traders' order $-S_t$, is evaluated at the
  floor and split into buy and sell volumes:

$$
B_t = \sum_j w_j\, \text{mean}_{l,i}\big(f^{j,i,l}_t(A^{\text{floor}})\big)^+ + \big(S_t(A^{\text{floor}})\big)^- ,
\qquad
O_t = \sum_j w_j\, \text{mean}_{l,i}\big(f^{j,i,l}_t(A^{\text{floor}})\big)^- + \big(S_t(A^{\text{floor}})\big)^+ ,
$$

  where $\text{mean}_{l,i}$ is the average over the $k\,n_j$ particles of type $j$.
  Then $O_t > B_t$. Buyers are served in full; sellers are served pro rata, with the
  **fill ratio**

$$
\rho_t = \frac{B_t}{O_t} \in [0, 1).
$$

Set $\rho_t = 1$ when the floor does not bind.

**Executed trade.** The trade that enters the reward and the dynamics of particle
$(j,i,l)$ is

$$
\hat f^{j,i,l}_t = \begin{cases} f^{j,i,l}_t(A_t) & \text{if } f^{j,i,l}_t(A_t) \ge 0 \quad (\text{buyer}),\\[2pt] \rho_t\, f^{j,i,l}_t(A_t) & \text{if } f^{j,i,l}_t(A_t) < 0 \quad (\text{seller}), \end{cases}
$$

with reward $-\hat f^{j,i,l}_t A_t$ and holdings
$\theta^{j,i,l}_{t+1} = \theta^{j,i,l}_t + \hat f^{j,i,l}_t$. Since $\rho_t = 1$ off the floor,
one formula covers both cases.

Pro rata is the mean-field stand-in for the price–time priority of the exchange
(KRX), with random order arrival.

### 1.5 $G_t$: the blocked volume

The price satisfies the complementarity system

$$
D_t(A_t) + G_t = 0, \qquad G_t \ge 0, \qquad A_t \ge A^{\text{floor}}, \qquad G_t\,\big(A_t - A^{\text{floor}}\big) = 0 ,
$$

so $G_t = -D_t(A^{\text{floor}}) = O_t - B_t$ when the floor binds and $0$ otherwise.
$G_t$ is **not** a trade. It is the volume of sell orders that the floor blocks, and
$\rho_t = 1 - G_t / O_t$. It has two readings, both reported:

1. **Blocked liquidity:** what the participants wanted to sell and could not, in
   total and by type $j$ (type $j$'s blocked volume is $(1 - \rho_t)$ times its sell
   orders).
2. **Cost of a reserve-backed floor:** $G_t$ is what a government would have to buy
   each day to make the floor a clearing price, so $A^{\text{floor}} \sum_t G_t$ is the
   budget such a floor would need.

### 1.6 Floor modes (flag `floor_mode`)

| `floor_mode` | meaning | executed sale at the floor | conservation |
|---|---|---|---|
| `"block"` (default) | trading below the floor is forbidden, sellers are rationed | $\rho_t\, f^{j,i,l}_t$ | $\bar\theta_{t+1} = \bar\theta_t + \hat S_t$ |
| `"support"` | the government buys the excess $G_t$ at the floor | $f^{j,i,l}_t$ | $\bar\theta_{t+1} = \bar\theta_t + \hat S_t - G_t$ |

Here $\bar\theta_t = \sum_j w_j\, \text{mean}_{l,i}\, \theta^{j,i,l}_t$ and $\hat S_t$ is the
executed exogenous net supply (rationed like any seller in `"block"` mode). Both modes
share the clearing; only the execution differs. One solve of each gives the
comparison of a blocking floor against a reserve-backed one. The difference is large
wherever the floor binds, in particular below the κ switch, where almost everyone
sells and $\rho_t \approx 0$ in `"block"` mode.

### 1.7 Timing and indexing (keep the current array conventions)

| array | index | meaning |
|---|---|---|
| `A_traj` `(T+1,)` | `[0]` | $A_{\text{ref}}$: the pre-open price, used only as $A^{\text{pre}}_0$. Use `A0_BASE` = 6.05. |
| | `[t+1]` | $A_t$, the price of step $t$, at which the trades of step $t$ are executed |
| `rho_traj` `(T+1,)` | `[0]` | $1$ (unused) |
| | `[t+1]` | $\rho_t$, the fill ratio of sellers at step $t$ |

Consequences, most of which hold already:

- State slot `A` (`x[5]`) at step $t$ holds `A_traj[t]` = $A^{\text{pre}}_t$. This is
  what `single_step_dynamics(..., next_A=A_traj[t+1], ...)` already writes. No change.
- The running mean in slot `A_avg` already averages `A_traj[1..t+1]`. At $T$ it is the
  mean of the $T$ prices, which is the statutory K-ETS reference. No change.
- **Change:** trade revenue is $-\hat f_t \cdot A_t$ with $A_t$ = `A_traj[t+1]`, not
  $-f_t \cdot$ `x[5]`, and holdings move by $\hat f_t$. `running_reward` and
  `single_step_dynamics` must receive the executed trade and the execution price.
- $A_0$ (the first cleared price) is now an **output**. Whether it is close to 6.05
  at $\gamma^{\text{ref}}$ is a calibration check, not an input.

---

## 2. Code changes, file by file

### 2.1 `envs/base_params.py`

Add:

```python
# ── market clearing ──────────────────────────────────────────────────────────
A_REF        = A0_BASE        # pre-open price A^pre_0, and centre of the elastic supply
BETA_MIN     = 0.1            # floor on the schedule slope (normalised price units)
CLEAR_ITERS  = 40             # bisection halvings on [A_floor, A_hi]
SIGMA_S      = 0.05 * CAP_F_BASE * MEAN_MARKET_IMPACT   # exogenous supply shock scale, TO CALIBRATE
S1           = 0.5 * CAP_F_BASE * MEAN_MARKET_IMPACT / A0_BASE  # supply elasticity, > 0, TO CALIBRATE
FLOOR_MODE   = os.environ.get('KETS_FLOOR_MODE', 'block')        # 'block' | 'support'

def supply_noise_base(key, T):
    """Standard normal draws; scaled by SIGMA_S * sqrt(dt) in the clearing step."""
    return jax.random.normal(key, (T,))
```

`S1` is in allowances per unit mass per dollar. The value above means that a price
move of one $A_0$ changes the exogenous supply by half an average trading capacity:
a placeholder, to be calibrated. Read `FLOOR_MODE` once at import, like `FLOOR_MAX`,
and record it in `progress.json`.

Keep `market_impact_base` and `white_noise_A_base` so `main` still runs, but do not
use them on this branch. Update the old note "deliberately no ceiling": there is
still no ceiling, and none is needed, because $s_1 > 0$ guarantees existence.

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

**Per-particle helpers:**

```python
def schedule_params(self, x, out):
    """(16,) state, (4,) net output -> (alpha, beta, lo, hi, xi, eta).
    lo/hi are the pre-trade clip bounds; xi/eta are unnormalised as before."""
    cap_f = x[idx['cap_f']]; theta = x[idx['theta']]
    lo, hi = -jnp.minimum(theta, cap_f), cap_f
    xi  = out[2] * x[idx['cap_xi']]
    eta = jnp.where(x[idx['tech']] >= 0.0, x[idx['tech']], out[3])
    return out[0], out[1], lo, hi, xi, eta

def order_at(self, x, alpha, beta, lo, hi, a):
    """f(a) for one particle; nonincreasing and continuous in a."""
    z = alpha - beta * (a - x[idx['A']]) / self.A_scale
    return jnp.clip(x[idx['cap_f']] * jnp.tanh(z), lo, hi)

def executed(self, f, rho):
    """Executed trade: buyers in full, sellers scaled by the fill ratio."""
    return jnp.where(f >= 0.0, f, rho * f)

def supply(self, a, eps_t):
    """Exogenous net supply S_t(a); positive = exogenous traders sell."""
    return SIGMA_S * eps_t * jnp.sqrt(self.dt) + S1 * (a - A_REF)
```

**Clearing:**

```python
def clear(self, orders_fn, eps_t, cap_total):
    """orders_fn(a) -> weighted (buy, sell) order volumes of the agents, summed over types.
    cap_total = sum_j w_j * C^{f,j}, the largest possible aggregate buy volume.
    Returns (A_t, rho_t, G_t)."""
    def D(a):
        buy, sell = orders_fn(a)
        return buy - sell - self.supply(a, eps_t)

    fl  = self.Afloor
    hi  = A_REF + (cap_total + SIGMA_S * jnp.abs(eps_t) * jnp.sqrt(self.dt)) / S1
    d_fl = D(fl)

    def bisect(_, lh):                     # invariant D(l) > 0 >= D(h)
        l, h = lh; m = 0.5 * (l + h); pos = D(m) > 0
        return jnp.where(pos, m, l), jnp.where(pos, h, m)
    l, h   = jax.lax.fori_loop(0, CLEAR_ITERS, bisect, (fl, hi))
    A_star = 0.5 * (l + h)

    binds = d_fl < 0
    A_t   = jnp.where(binds, fl, A_star)

    buy, sell = orders_fn(fl)
    s_fl = self.supply(fl, eps_t)
    B = buy  + jnp.maximum(-s_fl, 0.0)     # exogenous traders buying at the floor
    O = sell + jnp.maximum( s_fl, 0.0)     # exogenous traders selling at the floor
    rho = jnp.where(binds, B / jnp.maximum(O, 1e-12), 1.0)
    G   = jnp.where(binds, O - B, 0.0)
    if FLOOR_MODE == 'support':
        rho = jnp.ones_like(rho)           # government buys G: sellers are not rationed
    return A_t, rho, G
```

`orders_fn(a)` is closed form once $(\alpha, \beta, lo, hi)$ are computed: per type $j$,
the weighted means over the `(K, n_j)` particles of the positive and negative parts
of `order_at`. So there is **one** network forward pass per step; the bisection and
the rationing are pure arithmetic. The bisection is skipped in effect when the floor
binds (its result is discarded), so it can run unconditionally for `vmap`.

**One population step, shared by every rollout.** `ensemble_price_path`,
`ensemble_trajectory` and `ensemble_diagnostics` currently repeat the same step
logic. Replace it with a single private method so they cannot drift apart:

```python
def _population_step(self, stacked, statics, Xs, t_idx, P_traj, eps0, idio_pop, w):
    # 1. one forward pass per type on the pre-trade states
    # 2. schedule_params per particle
    # 3. A_t, rho_t, G_t = clear(orders_fn, eps0[t_idx], cap_total=sum_j w_j * C^{f,j})
    # 4. f = order_at(..., A_t);  f_exec = executed(f, rho_t)
    # 5. actions = (f_exec, xi, eta)
    # 6. new_Xs = step_dyn(Xs, actions, eps, next_A=A_t, next_P=P_traj[t+1])
    # returns new_Xs, A_t, rho_t, actions, diag = {G_t, binds, B_t, O_t, D(A_t), R_mean[j]}
```

Then:

- `ensemble_price_path`: scan `_population_step` and return
  `(A_traj, rho_traj)`, both `(T+1,)` (see §1.7). Add `return_diag=False`; when true,
  also return per-step `G_t`, the floor flag, `B_t`, `O_t`, the interior residual
  $D_t(A_t)$ and the mean reservation price by type $j$.
- `ensemble_trajectory`, `ensemble_diagnostics`: same step; rewards use $\hat f$ and $A_t$.
- `running_reward(x, a, A_exec)`: `trade_revenue = -a[0] * A_exec`, where `a[0]` is
  already the executed trade.
- `rollout_agent_batch(policy, j, A_traj, rho_traj, P_traj, idio)` and
  `agent_batch_trajectory` (the deviator): the network output gives the schedule;
  evaluate it at `A_exec = A_traj[t_idx + 1]`, execute it with
  `rho_traj[t_idx + 1]`, then reward and dynamics as before. The deviator is a
  price-taker: it treats both paths as exogenous, and the pathwise gradient flows
  through $\hat f = \rho_t f$ with $\rho_t$ fixed.
- `create_initial_state`: slot `A` is $A_{\text{ref}}$ (pass `A_REF`; same value as today).
- `compute_next_A`, `compute_next_A_weighted`: no longer called. Keep them, marked legacy.
- `reward_scale`: uses `self.A0` = `A0_BASE`, a **constant**, so leave it. It is a
  fixed scale, not the opening price.

### 2.4 `utils.py` (rung 3 only)

- `make_price_fn_r3`, the `prices` closure in `make_br_stage_fn_r3`, and every other
  caller of `ensemble_price_path`: the price function now returns `(A, rho)`. Freeze
  both with `stop_gradient` and pass both to the deviator's rollout.
- `make_batched_return_fn_r3`, `make_batched_hist_fn_r3`, `exploitability_r3`,
  `eval_block`: add the `rho` argument next to `A`. Both terms of an exploitability
  use the same `(A, rho)` per row, so the pairing is unchanged.
- In `sample_market`, `eps0` now comes from `supply_noise_base`; it goes through
  `env_T.generate_eps0`, so this is a constructor change in the entry points (§2.5).
- `eval_block`: compute floor statistics on `a[:, 1:]` (prices only) and add the
  diagnostics of §5, using `return_diag=True` on the block's price evaluation.
- `fictitious_play_rung3`: no logic change. `policy_0` is a fresh network, which with
  the centered parametrization is close to "no trade at an unchanged price".

### 2.5 Entry points: `main_rung3.py`, `pretrain_mm_r3.py`, `audit_pmfg.py`, `audit_paths.py`

- Pass `generate_eps0_func=supply_noise_base`. `market_impact_func` becomes
  unused; pass it, but it is ignored.
- Add `--floor-mode {block,support}` (default `block`), forwarded through
  `KETS_FLOOR_MODE` before import, and written to `progress.json`.
- Default output directories get the `results_mc_` prefix, plus `_support` for the
  support mode.
- `pretrain_mm_r3.py`: the target heuristic `target_hat_f` is unchanged. The fit
  becomes: make the schedule evaluated at the pre-trade price match the target,

  $$
  \min_\vartheta \ \mathbb E\Big[\big(\tanh(\alpha_\vartheta) - \hat f_{\text{target}}\big)^2\Big],
  $$

  since at $a = A^{\text{pre}}$ the centered schedule is $\tanh(\alpha)$. Also add a
  weak penalty pulling $\beta$ toward 1. The zero-gradient trap is still there (type 2
  starts at $\theta = 0$, so the clip lower bound is 0 and the gradient dies when
  $\alpha - \beta(\cdot) < 0$), so the warm start remains necessary.
- `audit_pmfg.py`: the specialist deviator also receives `rho` with the price path.
- `helper_plot.py`, `diagnostics.py`: relabel "allowance shock" as "net supply
  shock"; the price panels plot `A_traj[1:]`; add a panel for $\rho_t$ and $G_t$.

---

## 3. Tests — `tests/test_clearing.py` (pytest, runs on CPU)

1. **Single schedule.** All particles share $\alpha, \beta$, no clip binds, and
   $\sigma_S \varepsilon = 0$: $A_t$ solves $\text{order}(a) = s_1 (a - A_{\text{ref}})$. Check against
   a dense-grid root to `1e-6`.
2. **Monotonicity.** For random network outputs, $D_t$ on a grid of 1000 prices is
   strictly decreasing, with slope at most $-s_1$.
3. **Bracket.** For random outputs and shocks, $D_t(A^{\text{hi}}_t) < 0$.
4. **Floor binds, `block`.** Every particle sells at the floor: $A_t = A^{\text{floor}}$,
   $\rho_t = B_t / O_t < 1$, $G_t = O_t - B_t > 0$, and executed buys equal executed
   sells, exogenous traders included.
5. **Floor binds, `support`.** Same inputs: $\rho_t = 1$, sellers are filled in full,
   and the government's purchases equal $G_t$.
6. **Floor does not bind.** $\rho_t = 1$, $G_t = 0$, $\lvert D_t(A_t) \rvert <$ `1e-8`.
7. **Conservation.** Over a full `ensemble_trajectory`:
   `block` gives $\bar\theta_T - \bar\theta_0 = \sum_t \hat S_t$, and
   `support` gives $\bar\theta_T - \bar\theta_0 = \sum_t (\hat S_t - G_t)$, to relative `1e-5`.
8. **Shapes and jit.** `ensemble_price_path` returns two `(T+1,)` arrays,
   `A_traj[0] == A_REF`, `rho_traj[0] == 1`; `make_price_fn_r3` and
   `make_br_stage_fn_r3` compile under `vmap` at `T=10`.
9. **Deviator consistency.** A deviator holding the same parameters as a
   single-component mixture gets the same return as that component's particles,
   up to idiosyncratic noise, including on paths where the floor binds.

## 4. Smoke run before the 30-hour solve

```bash
KETS_FLOOR_MAX=30 python3 main_rung3.py choice --model-dir results_mc_smoke \
    --floor-mode block --nb-policy 5 --t-grid 50 --iters 1000 --batch 128 \
    --verify-samples 100 --fixed-samples 50 --eval-samples 16
```

Accept when:
- the tests pass;
- the interior residual $\max \lvert D_t(A_t) \rvert$ is below 1e-6 (relative);
- exploitability decreases over the 5 iterations;
- no NaN appears in $\alpha$, $\beta$ or $\rho$.

## 5. Diagnostics to report (progress.json, per block)

| key | meaning | expected |
|---|---|---|
| `A_on_floor` | fraction of steps where the floor binds | high below the κ switch |
| `rho_mean_on_floor`, `rho_min` | fill ratio of sellers when the floor binds | ≈ 0 below the κ switch in `block` mode |
| `blocked_volume`, `blocked_volume[j]` | $\sum_t G_t$, and type $j$'s share $(1 - \rho_t) \times$ its sell orders | 0 in the interior |
| `support_cost` | $A^{\text{floor}} \sum_t G_t$, the budget a reserve-backed floor would need | |
| `clear_residual_max` | max interior $\lvert D_t(A_t) \rvert$, relative to $\sum_j w_j C^{f,j}$ | < 1e-6 |
| `price_drift` | mean and std of $A_{t+1} - A_t$ off the floor | ≈ 0 (near-martingale), except from the $C^{f,j}$ caps |
| `R_mean[j]` | mean reservation price of type $j$ | brackets $A_t$ |
| `A0_cleared` | price at $t = 0$ | compare with 6.05 at $\gamma^{\text{ref}}$ |
| `market_short_frac` | fraction of rows in which the market as a whole ends short | see below |

**Sanity check on the κ switch.** For $\kappa^2 < 1$ the price should sit on the floor
and, in `block` mode, sellers should be heavily rationed. For $\kappa^2 > 1$, the
heuristic $A_{T-1} \approx \kappa^2 \tilde A_T\, \mathbb P(\text{aggregate short})$ predicts
$\mathbb P(\text{aggregate short}) \approx 1/\kappa^2$.

## 6. Phases

1. **Core** (§2.1–2.3 and the tests): params, network head, clearing, rationing, rollouts.
2. **FP and warm start** (§2.4–2.5): the smoke run of §4, then the full solves on
   `results_mc_floor{20,30,40}` in `block` mode.
3. **Support mode.** The same solves with `--floor-mode support`, for the comparison
   of a blocking floor against a reserve-backed one.
4. **Unrestricted audit.** In `audit_pmfg.py`, add `--unrestricted`: the specialist
   deviator of type $j$ is a network with input
   `(normalised state, A_exec / A_scale, rho)` that outputs $f$ directly (tanh times
   $C^{f,j}$, clipped). It never takes part in clearing, so it needs no
   monotonicity. Compare its gain with the restricted specialist: if they match, the
   schedule form is not binding.
5. **Price bank (performance).** The population is frozen during a best response.
   Once per FP iteration, clear `N_bank` (about 1e5) markets with the chunked
   `price_fn`. Inside `make_br_stage_fn_r3`, sample minibatches of
   (γ, P, A, ρ) from the bank instead of re-simulating the population at every
   gradient step. The deviator's idiosyncratic noise stays fresh.
6. **Robustness (optional, for the paper).**
   - $C^{f,j} \times \{0.5, 2\}$ at $\gamma^{\text{ref}}$ and two more scenarios.
   - $s_1 \times \{0.5, 2\}$: sensitivity to market depth.
   - A $K$-step schedule, $\sum_{k} w_k \tanh(\alpha_k - \beta_k \cdot)$ with $w_k \ge 0$, $\sum_k w_k = 1$, for any type $j$ where the audit of phase 4 shows a gap.

## 7. Paper changes this implies

- §2.1: eq. (price_dyn) and the impact function $h$ are replaced by the clearing of
  §1.3 and the floor rule of §1.4; the noise $\varepsilon^{0,2}$ becomes a supply shock of
  price-elastic exogenous traders; $A_0$ is endogenous; there is no government trade.
- Definition of a control: the trade component is a demand schedule (§1.1), and the
  executed trade is $\hat f$.
- §3: the particle step (particle_h) becomes the clearing step, which returns the
  price and the fill ratio. FP and the best responses are unchanged; the deviating
  agent faces frozen $(A_t, \rho_t)$.
- §4: report the diagnostics of §5, the conservation of allowances, and the
  `block` / `support` comparison (blocked liquidity against the cost of support).

## 8. Open choices (decide before phase 2)

- **`SIGMA_S` and `S1`**: the supply-shock scale and the market depth. Calibrate to
  K-ETS daily volumes and price volatility; they set the liquidity the old $h$ stood for.
- **The floor in K-ETS**: check that it is a hard trading limit (orders below it
  rejected). `block` assumes that; `support` is the reserve-backed alternative.
- **Production**: v1 decides $\xi, \eta$ from the pre-trade state (one forward pass).
  v2 would feed the price $A_t$ to the production heads (a second, cheap forward pass
  after clearing). The trade head must stay price-free either way.
- **`A_REF`**: 6.05 (the observed price), used as the pre-open value and as the centre
  of the elastic supply.
- **Not in this document**: the major–minor formulation (paper type 2 as one large
  agent submitting its own schedule) and the merit-order electricity price. Both
  build on this clearing step; write them up separately once it works.
