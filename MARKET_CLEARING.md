# Market clearing — implementation instructions

Branch `market-clearing`, cut from `main` at `4b0d6af`.

This branch replaces the reduced-form price impact

$$
A_{t+1} = \max\big(A_t + h(\bar f_t)\,dt + \varepsilon^{0,2}_t \sqrt{dt},\ A^{\text{floor}}\big)
$$

(`compute_next_A_weighted` in `envs/environment.py`) by a **market-clearing price**.
At each day $t$, every agent of type $j$ submits a demand schedule, i.e. a trade as a
function of the price. Exogenous traders add a random net supply, and the price is
the one at which aggregate trade balances.

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

### 1.2 Exogenous traders: a random net supply

$$
S_t = \sigma_S(\gamma)\, \varepsilon^{0,2}_t \sqrt{dt} .
$$

$S_t$ is in allowances per unit mass of agents. It is a net supply: positive means the
exogenous traders sell, negative means they buy. It does **not** depend on the price.
In the rationing step the exogenous traders are treated as one more participant, with
order $f^{S}_t = -S_t$. `eps0` keeps its shape and seeds; only its meaning changes,
from a price shock to a supply shock. The shocks have mean zero, so over the year the
exogenous traders add no allowances on average, and the cap stays the total free
allocation.

**Scale of the shock: proportional to the allowances of the scenario.** The shocks are
read as coming from the allowances in circulation, so their size scales with the total
free allocation of the market's own scenario $\gamma$:

$$
\Theta(\gamma) = \sum_j w_j(\gamma)\, \theta^j_0(\gamma), \qquad
\sigma_S(\gamma) = \frac{c\, \Theta(\gamma)}{\sqrt{365}}, \qquad c = 1\% .
$$

- The cumulative exogenous supply over the year has standard deviation $c\,\Theta(\gamma)$,
  1% of the cap, whatever the number of steps $T$ (with $dt = 365/T$).
- $\Theta(\gamma)$ depends on the allocation level $\tilde\theta^0$ and on $\mathfrak n_3$ (fewer
  producers, fewer allowances per unit mass). Compute it per market from the context
  weights and the initial holdings (`bind_env`), so it is vmapped like the rest; the
  networks see $\gamma$ through FiLM and learn the scenario-dependent noise as is.
- Orders of magnitude (daily trading capacity of the market about 154 at
  $\gamma^{\text{ref}}$): $\Theta \approx 13{,}900$ gives a daily standard deviation of about 7.3,
  roughly 4.7% of capacity; at $\tilde\theta^0 = 1.2$, about 9.7 a day; at $\tilde\theta^0 = 0$, no
  supply shock (only the electricity price remains random).
- **To state in the paper:** the allocation lever has two effects, more allowances and
  more volatile exogenous supply, so the sensitivity surfaces over $\tilde\theta^0$ include
  its effect on volatility. If the two must be separated, run a second solve with
  $\sigma_S$ fixed at $\sigma_S(\gamma^{\text{ref}})$.

**No price-elastic term.** An earlier version of this document had
$S_t(a) = \sigma_S \varepsilon^{0,2}_t \sqrt{dt} + s_1 (a - A_{\text{ref}})$. It is removed: outside
traders who sell whenever the price rises leak the cap, and with the value that was
proposed ($s_1 \approx 16$) they supplied more than the whole cap over a year and pinned
the price near $A_{\text{ref}} = 6.05$. Any code with `S1` must drop that term.

### 1.3 Excess demand and the clearing price

With the context weights $w_j$ (`context_weights()`, a function of $\mathfrak n_3$ in $\gamma$)
and $n$ simulated agents per type, agent $(j,i)$ following its own drawn component
$l^{j,i}$ of the mixture (§1.8),

$$
D_t(a) = \sum_{j} w_j \, \frac{1}{n} \sum_{i=1}^{n} f^{j,i}_t(a) \;-\; S_t,
\qquad f^{j,i}_t \text{ given by the schedule of } \vartheta^{j, l^{j,i}} .
$$

$D_t$ is continuous and nonincreasing in $a$, since every schedule is. The price lives
in the corridor $[A^{\text{floor}}, A^{\text{ceil}}]$, where $A^{\text{ceil}}$ = `A_CEIL` is an
**artificial ceiling**: K-ETS has none, but a compact corridor is what guarantees that
a price always exists. Set it high enough that it never binds (e.g. 200 USD) and log
how often it does; it should be about never.

**Selection.** Inside the corridor, $D_t$ may be flat at zero over an interval, for
example when every agent is at a clip bound. Let
$a^-_t = \sup\{a \in [A^{\text{floor}}, A^{\text{ceil}}] : D_t(a) > 0\}$ (equal to $A^{\text{floor}}$ if the
set is empty) and $a^+_t = \inf\{a \in [A^{\text{floor}}, A^{\text{ceil}}] : D_t(a) < 0\}$ (equal to
$A^{\text{ceil}}$ if the set is empty). By monotonicity every price in $[a^-_t, a^+_t]$ clears
the market, and we take

$$
A_t = \operatorname{clip}\big(A^{\text{pre}}_t,\ a^-_t,\ a^+_t\big):
$$

the price does not move if the previous one still clears, and otherwise it moves as
little as possible.

### 1.4 The bounds block the price: rationing (short-side rule)

The same formula covers the bounds: $A_t = A^{\text{floor}}$ when $D_t(A^{\text{floor}}) < 0$, and
$A_t = A^{\text{ceil}}$ when $D_t(A^{\text{ceil}}) > 0$. In both cases the market does not clear at
$A_t$, and the long side is rationed. Every order, including the exogenous traders'
order $-S_t$, is evaluated at $A_t$ and split into buy and sell volumes:

$$
B_t = \sum_j w_j\, \text{mean}_{i}\big(f^{j,i}_t(A_t)\big)^+ + \big(S_t\big)^- ,
\qquad
O_t = \sum_j w_j\, \text{mean}_{i}\big(f^{j,i}_t(A_t)\big)^- + \big(S_t\big)^+ ,
$$

where $\text{mean}_{i}$ is the average over the $n$ simulated agents of type $j$.

- **Interior** ($A^{\text{floor}} < A_t < A^{\text{ceil}}$, or a bound where $D_t = 0$): $B_t = O_t$,
  every order is executed in full.
- **The floor binds** ($D_t(A^{\text{floor}}) < 0$): $O_t > B_t$. Buyers are served in full;
  sellers are served pro rata, with the **sellers' fill ratio**
  $\rho^-_t = B_t / O_t \in [0, 1)$.
- **The ceiling binds** ($D_t(A^{\text{ceil}}) > 0$, should not happen): $B_t > O_t$. Sellers are
  served in full; buyers are served pro rata, with the **buyers' fill ratio**
  $\rho^+_t = O_t / B_t \in [0, 1)$.

Set a ratio to 1 whenever its side is not rationed.

**Executed trade.** The trade that enters the reward and the dynamics of agent $(j,i)$ is

$$
\hat f^{j,i}_t = \begin{cases} \rho^+_t\, f^{j,i}_t(A_t) & \text{if } f^{j,i}_t(A_t) \ge 0 \quad (\text{buyer}),\\[2pt] \rho^-_t\, f^{j,i}_t(A_t) & \text{if } f^{j,i}_t(A_t) < 0 \quad (\text{seller}), \end{cases}
$$

with reward $-\hat f^{j,i}_t A_t$ and holdings $\theta^{j,i}_{t+1} = \theta^{j,i}_t + \hat f^{j,i}_t$.
In the interior both ratios are 1, so one formula covers every case.

Pro rata is the mean-field stand-in for the price–time priority of the exchange
(KRX), with random order arrival.

### 1.5 $G_t$: the blocked volume

The price satisfies the complementarity system

$$
D_t(A_t) + G_t = 0, \qquad G_t \in -N_{[A^{\text{floor}}, A^{\text{ceil}}]}(A_t),
$$

that is, $G_t = 0$ in the interior, $G_t = O_t - B_t \ge 0$ when the floor binds and
$G_t = O_t - B_t \le 0$ when the ceiling binds. $G_t$ is **not** a trade. At the floor it
is the volume of sell orders the floor blocks ($\rho^-_t = 1 - G_t / O_t$); at the ceiling,
$-G_t$ is the blocked buy orders. At the floor it has two readings, both reported:

1. **Blocked liquidity:** what the participants wanted to sell and could not, in
   total and by type $j$ (type $j$'s blocked volume is $(1 - \rho^-_t)$ times its sell
   orders).
2. **Cost of a reserve-backed floor:** $G_t$ is what a government would have to buy
   each day to make the floor a clearing price, so $A^{\text{floor}} \sum_t G_t$ is the
   budget such a floor would need.

### 1.6 Floor modes (flag `floor_mode`)

| `floor_mode` | meaning | executed sale at the floor | conservation |
|---|---|---|---|
| `"block"` (default) | trading outside the corridor is forbidden, the long side is rationed | $\rho^-_t\, f^{j,i}_t$ | $\bar\theta_{t+1} = \bar\theta_t + \hat S_t$ |
| `"support"` | the government buys the excess $G_t$ at the floor (and sells $-G_t$ at the ceiling) | $f^{j,i}_t$ | $\bar\theta_{t+1} = \bar\theta_t + \hat S_t - G_t$ |

Here $\bar\theta_t = \sum_j w_j\, \text{mean}_{i}\, \theta^{j,i}_t$ and $\hat S_t$ is the
executed exogenous net supply (rationed like any participant on the long side in
`"block"` mode). Both modes share the clearing; only the execution differs. One solve
of each gives the comparison of a blocking floor against a reserve-backed one. The
difference is large wherever the floor binds, in particular below the κ switch, where
almost everyone sells and $\rho^-_t \approx 0$ in `"block"` mode.

### 1.7 Timing and indexing (keep the current array conventions)

| array | index | meaning |
|---|---|---|
| `A_traj` `(T+1,)` | `[0]` | $A_{\text{ref}}$: the pre-open price, used only as $A^{\text{pre}}_0$. Use `A0_BASE` = 6.05. |
| | `[t+1]` | $A_t$, the price of step $t$, at which the trades of step $t$ are executed |
| `rho_traj` `(T+1, 2)` | `[0]` | $(1, 1)$ (unused) |
| | `[t+1]` | $(\rho^-_t, \rho^+_t)$, the fill ratios of sellers and buyers at step $t$ |

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

### 1.8 Simulating the population: one control per agent

**Current scheme (block).** Each market simulates every one of the $k$ components of
the mixture, with $N_j$ particles each, averaged with weights $1/k$: $k\, N_j$
particles per type. The cost of an FP iteration grows with $k$, so a full solve costs
$O(K^2)$. This is the quadratic runtime of Fig. 3 of the paper.

**New scheme (per agent).** Each market simulates a **fixed** number $n$ of agents per
type $j$, and each agent draws its own component:

$$
l^{j,i} \in \{0, \dots, k-1\}, \qquad i = 1, \dots, n, \qquad
\text{agent } (j,i) \text{ follows } \bar\pi^{j,\, l^{j,i}} \text{ for the whole year.}
$$

This is the direct Monte Carlo of the definition of a population control: each member
of the population draws its own control $U^j \sim \sigma^j$ (Definition 3 of the paper).
The block scheme is the special case $n = k N_j$ with perfect stratification.

- **Cost:** $O(n)$ per step and type instead of $O(k N_j)$, so an FP iteration has a
  constant cost and a full solve costs $O(K)$. This makes 100–200 FP iterations
  affordable; exploitability was still decreasing at $K = 50$.
- **Bias:** none for the mixture, up to the finite-population error of order $1/n$,
  the same order as today.
- **Draw per agent, never per market** (for the minors). Putting all $n$ agents of type
  $j$ on one drawn component would coordinate the whole type in that market. Clearing
  and the dynamics are nonlinear in the population's behaviour, so averaging such
  markets is a biased estimate of a *different* object, not a noisier estimate of the
  mixture. (One draw per market is the right rule for a **major** player only; not on
  this branch.)

**Stratified assignment.** Rather than i.i.d. draws, spread the agents evenly over the
components with a random offset drawn per market:

$$
l^{j,i} = \Big(o^j + \Big\lfloor \frac{i\, k}{n} \Big\rfloor\Big) \bmod k, \qquad o^j \sim \text{Unif}\{0, \dots, k-1\} .
$$

- If $n \ge k$, every component appears in every market, with $\lfloor n/k \rfloor$ or
  $\lceil n/k \rceil$ agents.
- If $n < k$, each market sees an evenly spread subset of the components, and the
  offset makes the scheme unbiased across markets.
- Choose $n \ge K_{\max}$, e.g. $n = 128$ to $256$ per type. With the weights $w_j$
  carrying the proportions, use the **same** $n$ for every type, so the shapes no
  longer depend on $\mathfrak n_3$.
- The formula only needs $k$ as an integer, which can be a traced value. With the
  mixture stored in a fixed buffer of size $K_{\max}$ (unused slots never drawn), the
  population step compiles **once** for the whole solve instead of once per FP
  iteration, as happens today because the stacked mixture changes shape.

**Exploitability.** Keep the average reward $J^j$ of the mixture **exact** at first:
`make_batched_hist_fn_r3` rolls out every component on the frozen blocks. It runs once
per iteration, so its cost is moderate, and the convergence diagnostic stays as
precise as before. If it becomes the bottleneck, sample the evaluation agents'
components the same way, enumerating them exactly when $M_{\text{eval}} \ge k$.

**Flag.** `POP_SAMPLING = 'agent' | 'block'`, default `'agent'`. Keep `'block'` for
the equivalence check of §4.

---

## 2. Code changes, file by file

### 2.1 `envs/base_params.py`

Add:

```python
# ── market clearing ──────────────────────────────────────────────────────────
A_REF        = A0_BASE        # pre-open price A^pre_0 (tie-breaker of the selection rule)
A_CEIL       = 200.0          # artificial ceiling (USD), for existence only; must essentially never bind
BETA_MIN     = 0.1            # floor on the schedule slope (normalised price units)
CLEAR_ITERS  = 40             # bisection halvings per side on [A_floor, A_CEIL]
SHOCK_SHARE  = 0.01           # c: yearly std of exogenous supply = 1% of the scenario's allocation
FLOOR_MODE   = os.environ.get('KETS_FLOOR_MODE', 'block')        # 'block' | 'support'

# ── population sampling (§1.8) ───────────────────────────────────────────────
POP_SAMPLING = os.environ.get('KETS_POP_SAMPLING', 'agent')      # 'agent' | 'block'
N_SIM        = 128            # simulated agents per type and market ('agent' mode); >= K_MAX
K_MAX        = 200            # size of the fixed mixture buffer

def supply_noise_base(key, T):
    """Standard normal draws; scaled by sigma_S(gamma) * sqrt(dt) in the clearing step."""
    return jax.random.normal(key, (T,))
```

There is no constant `SIGMA_S`: the scale $\sigma_S(\gamma) = c\,\Theta(\gamma)/\sqrt{365}$ is computed
per market from the scenario (§1.2, §2.3). There is **no** `S1` either: the exogenous
supply does not depend on the price. Read
`FLOOR_MODE` once at import, like `FLOOR_MAX`, and record it in `progress.json`.

Keep `market_impact_base` and `white_noise_A_base` so `main` still runs, but do not
use them on this branch. Update the old note "deliberately no ceiling": K-ETS has no
ceiling, and `A_CEIL` is only an artificial bound that guarantees a price exists.

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
- Add, next to `apply_stacked`, a gather-based application for the per-agent
  sampling of §1.8:

  ```python
  def apply_gathered(stacked, static, comp, X):
      """stacked: params with leading dim K_MAX.  comp: (n,) component indices.
      X: (n, in_dim) -> (n, 4). Agent i uses the parameters of component comp[i]."""
      params_i = jax.tree_util.tree_map(lambda p: p[comp], stacked)
      return jax.vmap(lambda p, x: eqx.combine(p, static)(x))(params_i, X)
  ```

  The gathered parameters cost $n \times$ one network per type, which is small for a
  3×64 trunk with FiLM.

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

def executed(self, f, rho_sell, rho_buy):
    """Executed trade: each side scaled by its fill ratio (1 when not rationed)."""
    return jnp.where(f >= 0.0, rho_buy * f, rho_sell * f)

def sigma_S(self):
    """sigma_S(gamma) = c * Theta(gamma) / sqrt(365), Theta = total free allocation per unit mass
    of the bound scenario (context weights x initial holdings of each type)."""
    w     = self.context_weights()
    theta = jnp.stack([self.type_initial_state(j)[1] for j in range(N_TYPES)])
    return SHOCK_SHARE * jnp.sum(w * theta) / jnp.sqrt(365.0)

def supply(self, eps_t):
    """Exogenous net supply S_t; positive = exogenous traders sell. Price-independent."""
    return self.sigma_S() * eps_t * jnp.sqrt(self.dt)
```

`sigma_S()` reads the scenario bound by `bind_env` (allocation level and $\mathfrak n_3$), so it
is traced and vmapped per market with no extra argument.

**Clearing:**

```python
def clear(self, orders_fn, eps_t, A_pre):
    """orders_fn(a) -> weighted (buy, sell) order volumes of the agents, summed over types.
    Returns (A_t, rho_sell, rho_buy, G_t)."""
    S = self.supply(eps_t)
    def D(a):
        buy, sell = orders_fn(a)
        return buy - sell - S

    lo_, hi_ = self.Afloor, A_CEIL
    d_lo, d_hi = D(lo_), D(hi_)

    def left(_, lh):                       # a_minus = sup{a : D(a) > 0}; invariant D(l) > 0 >= D(h)
        l, h = lh; m = 0.5 * (l + h); pos = D(m) > 0
        return jnp.where(pos, m, l), jnp.where(pos, h, m)
    def right(_, lh):                      # a_plus = inf{a : D(a) < 0}; invariant D(l) >= 0 > D(h)
        l, h = lh; m = 0.5 * (l + h); neg = D(m) < 0
        return jnp.where(neg, l, m), jnp.where(neg, m, h)

    a_minus = jnp.where(d_lo <= 0, lo_, jnp.where(d_hi > 0, hi_,
              jax.lax.fori_loop(0, CLEAR_ITERS, left,  (lo_, hi_))[1]))
    a_plus  = jnp.where(d_hi >= 0, hi_, jnp.where(d_lo < 0, lo_,
              jax.lax.fori_loop(0, CLEAR_ITERS, right, (lo_, hi_))[0]))
    A_t = jnp.clip(A_pre, a_minus, a_plus)

    # rationing of the long side at a bound (the ratios are 1 in the interior)
    buy, sell = orders_fn(A_t)
    B = buy  + jnp.maximum(-S, 0.0)        # exogenous traders buying
    O = sell + jnp.maximum( S, 0.0)        # exogenous traders selling
    rho_sell = jnp.where(O > B, B / jnp.maximum(O, 1e-12), 1.0)
    rho_buy  = jnp.where(B > O, O / jnp.maximum(B, 1e-12), 1.0)
    G = O - B                              # >= 0 at the floor, <= 0 at the ceiling, ~0 inside
    if FLOOR_MODE == 'support':            # the government takes the other side of G
        rho_sell = jnp.ones_like(rho_sell); rho_buy = jnp.ones_like(rho_buy)
    return A_t, rho_sell, rho_buy, G
```

In the interior $B_t = O_t$ up to the bisection tolerance, so both ratios are 1 (use a
small relative tolerance in the comparisons `O > B` and `B > O` to avoid spurious
rationing from rounding).

`orders_fn(a)` is closed form once $(\alpha, \beta, lo, hi)$ are computed: per type $j$,
the weighted means over the $n$ simulated agents (or the `(K, n_j)` particles in
`'block'` mode) of the positive and negative parts of `order_at`. So there is **one**
network forward pass per step; the bisections and the rationing are pure arithmetic,
and they run unconditionally (results selected with `jnp.where`) so the function
`vmap`s over markets.

**One population step, shared by every rollout.** `ensemble_price_path`,
`ensemble_trajectory` and `ensemble_diagnostics` currently repeat the same step
logic. Replace it with a single private method so they cannot drift apart:

```python
def _population_step(self, stacked, statics, comp, Xs, t_idx, P_traj, eps0, idio_pop, w):
    # 0. 'agent' mode: Xs[j] is (n, 16) and comp[j] is (n,), drawn once per market (§1.8);
    #    'block' mode: Xs[j] is (K, n_j, 16) and comp is unused (today's layout)
    # 1. one forward pass per type on the pre-trade states
    #    ('agent': apply_gathered(stacked[j], statics[j], comp[j], norm(Xs[j])))
    # 2. schedule_params per particle
    # 3. A_t, rho_sell, rho_buy, G_t = clear(orders_fn, eps0[t_idx], A_pre)
    # 4. f = order_at(..., A_t);  f_exec = executed(f, rho_sell, rho_buy)
    # 5. actions = (f_exec, xi, eta)
    # 6. new_Xs = step_dyn(Xs, actions, eps, next_A=A_t, next_P=P_traj[t+1])
    # returns new_Xs, A_t, (rho_sell, rho_buy), actions,
    #         diag = {G_t, on_floor, on_ceil, flat = a_plus - a_minus > tol, B_t, O_t, D(A_t), R_mean[j]}
```

Then:

- `ensemble_price_path`: scan `_population_step` and return
  `(A_traj, rho_traj)` of shapes `(T+1,)` and `(T+1, 2)` (see §1.7). Add
  `return_diag=False`; when true, also return per-step `G_t`, the floor and ceiling
  flags, the flat flag, `B_t`, `O_t`, the interior residual $D_t(A_t)$ and the mean
  reservation price by type $j$.
- `ensemble_trajectory`, `ensemble_diagnostics`: same step; rewards use $\hat f$ and $A_t$.
- `running_reward(x, a, A_exec)`: `trade_revenue = -a[0] * A_exec`, where `a[0]` is
  already the executed trade.
- `rollout_agent_batch(policy, j, A_traj, rho_traj, P_traj, idio)` and
  `agent_batch_trajectory` (the deviator): the network output gives the schedule;
  evaluate it at `A_exec = A_traj[t_idx + 1]`, execute it with
  `rho_traj[t_idx + 1]`, then reward and dynamics as before. The deviator is a
  price-taker: it treats both paths as exogenous, and the pathwise gradient flows
  through $\hat f = \rho^\pm_t f$ with the ratios fixed.
- `population_initial_states`: in `'agent'` mode, return one `(N_SIM, 16)` array per
  type; the type counts no longer depend on `financial_inst` (the proportions enter
  only through `context_weights`). Add
  `population_components(key, k)` returning, per type $j$, the `(N_SIM,)` stratified
  indices $l^{j,i} = (o^j + \lfloor i k / n \rfloor) \bmod k$ of §1.8.
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
- **Per-agent sampling (§1.8):**
  - Store the mixture in a fixed buffer of size `K_MAX` per type and pass the current
    size `k` as a traced integer, so the jitted price and best-response functions
    compile once per solve.
  - `sample_market` draws one component key per market, alongside γ and the common
    noise; `prices` calls `population_components` with it. The population's
    idiosyncratic noise becomes `(N_SIM, T)` per type.
  - Frozen validation blocks: draw the assignment from the block's seed folded with
    `k` (`jax.random.fold_in(seed, k)`), so a block is reproducible at a given
    iteration.
  - `make_batched_hist_fn_r3` (exact $J^j$) is unchanged.

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
  shock"; the price panels plot `A_traj[1:]`; add a panel for $\rho^-_t$, $\rho^+_t$ and $G_t$.

---

## 3. Tests — `tests/test_clearing.py` (pytest, runs on CPU)

1. **Single schedule.** All agents share $\alpha, \beta$, no clip binds, and
   $\sigma_S \varepsilon = 0$: $A_t = R = A^{\text{pre}} + A_{\text{scale}}\,\alpha/\beta$, to `1e-6`.
2. **Monotonicity.** For random network outputs, $D_t$ on a grid of 1000 prices is
   nonincreasing.
3. **Flat region.** Every agent is clipped at its buy cap and $S_t$ equals total demand:
   $a^-_t < a^+_t$, and $A_t = A^{\text{pre}}$ when $A^{\text{pre}} \in [a^-_t, a^+_t]$.
4. **Floor binds, `block`.** Every agent sells at the floor: $A_t = A^{\text{floor}}$,
   $\rho^-_t = B_t / O_t < 1$, $\rho^+_t = 1$, $G_t = O_t - B_t > 0$, and executed buys
   equal executed sells, exogenous traders included.
5. **Floor binds, `support`.** Same inputs: $\rho^-_t = 1$, sellers are filled in full,
   and the government's purchases equal $G_t$.
6. **Interior.** $\rho^-_t = \rho^+_t = 1$, $G_t \approx 0$, $\lvert D_t(A_t) \rvert <$ `1e-8`.
6b. **Ceiling binds.** Every agent buys at `A_CEIL`: $A_t$ = `A_CEIL`, $\rho^+_t = O_t / B_t < 1$,
   $\rho^-_t = 1$, $G_t < 0$.
7. **Conservation.** Over a full `ensemble_trajectory`:
   `block` gives $\bar\theta_T - \bar\theta_0 = \sum_t \hat S_t$, and
   `support` gives $\bar\theta_T - \bar\theta_0 = \sum_t (\hat S_t - G_t)$, to relative `1e-5`.
8. **Shapes and jit.** `ensemble_price_path` returns arrays of shapes `(T+1,)` and
   `(T+1, 2)`, `A_traj[0] == A_REF`, `rho_traj[0] == (1, 1)`; `make_price_fn_r3` and
   `make_br_stage_fn_r3` compile under `vmap` at `T=10`.
9. **Deviator consistency.** A deviator holding the same parameters as a
   single-component mixture gets the same return as that component's particles,
   up to idiosyncratic noise, including on paths where the floor binds.
10. **Stratification.** For $n \ge k$, every component gets $\lfloor n/k \rfloor$ or
    $\lceil n/k \rceil$ agents; over many offsets, each component's frequency is $1/k$.
11. **Agent = block in the stratified limit.** With $n = k N_j$, offset 0 and the same
    idiosyncratic draws, $\lfloor i k / n \rfloor = \lfloor i / N_j \rfloor$ assigns exactly
    $N_j$ agents to each component, so `'agent'` mode reproduces the `'block'` price
    path to floating-point precision.

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

**Sampling check (once, for the paper).** Run the same short configuration with
`KETS_POP_SAMPLING=block` and `KETS_POP_SAMPLING=agent` (e.g. $T = 50$, $K = 10$,
one seed, `N_SIM` ≥ 10). The price paths, the fill ratios $\rho^\pm_t$ and the
exploitability per type $j$ should agree within the seed-to-seed noise. Report it in
one sentence and an appendix figure. Also record the time per FP iteration in both
modes: constant in `'agent'` mode, growing with $k$ in `'block'` mode.

## 5. Diagnostics to report (progress.json, per block)

| key | meaning | expected |
|---|---|---|
| `A_on_floor` | fraction of steps where the floor binds | high below the κ switch |
| `A_on_ceil` | fraction of steps where the artificial ceiling binds | ≈ 0; if not, `A_CEIL` is doing economic work |
| `clear_flat_frac` | fraction of steps with $a^-_t < a^+_t$ (selection rule used) | small |
| `rho_mean_on_floor`, `rho_min` | fill ratio $\rho^-_t$ of sellers when the floor binds | ≈ 0 below the κ switch in `block` mode |
| `blocked_volume`, `blocked_volume[j]` | $\sum_t G_t$ at the floor, and type $j$'s share $(1 - \rho^-_t) \times$ its sell orders | 0 in the interior |
| `exogenous_net_supply` | $\sum_t \hat S_t$ divided by the total free allocation | ≈ 0 (mean-zero shocks) |
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

1. **Core** (§2.1–2.3 and the tests): params, network head, clearing, rationing,
   per-agent population sampling, rollouts.
2. **FP and warm start** (§2.4–2.5): the smoke run and the sampling check of §4, then
   the full solves on `results_mc_floor{20,30,40}` in `block` floor mode, with
   per-agent sampling and more FP iterations (e.g. 100) at the same wall time.
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
   - $c \in \{0.5\%, 2\%\}$: sensitivity to the size of the exogenous supply shocks.
   - $\sigma_S$ fixed at $\sigma_S(\gamma^{\text{ref}})$: separates the effect of the allocation from
     its effect on volatility, if a reviewer asks.
   - A $K$-step schedule, $\sum_{k} w_k \tanh(\alpha_k - \beta_k \cdot)$ with $w_k \ge 0$, $\sum_k w_k = 1$, for any type $j$ where the audit of phase 4 shows a gap.

## 7. Paper changes this implies

- §2.1: eq. (price_dyn) and the impact function $h$ are replaced by the clearing of
  §1.3 and the floor rule of §1.4; the noise $\varepsilon^{0,2}$ becomes a mean-zero supply
  shock of exogenous traders; $A_0$ is endogenous; there is no government trade.
- Definition of a control: the trade component is a demand schedule (§1.1), and the
  executed trade is $\hat f$.
- §3: the particle step (particle_h) becomes the clearing step, which returns the
  price and the fill ratios. FP and the best responses are unchanged; the deviating
  agent faces frozen $(A_t, \rho^-_t, \rho^+_t)$.
- §3, simulating the population: replace eq. (particle_h), which averages all $k$
  components with $N_j$ particles each, by the per-agent scheme of §1.8, presented as
  the Monte Carlo of the definition of a population control. State the complexity:
  $O(n)$ per step and type instead of $O(k N_j)$, hence $O(K)$ for a solve instead of
  $O(K^2)$, and redo the runtime figure.
- §4: report the diagnostics of §5, the conservation of allowances, and the
  `block` / `support` comparison (blocked liquidity against the cost of support).

## 8. Open choices (decide before phase 2)

- **`SHOCK_SHARE`** ($c$ = 1%): the yearly size of the exogenous supply shocks relative to
  the scenario's allocation. Check it against K-ETS daily price volatility (log the
  realized daily volatility at $T = 365$).
- **`A_CEIL`**: high enough never to bind. If it binds, it is doing economic work and
  must be reported.
- **A price-elastic exogenous supply** ($s_1$): deliberately not in this model (§1.2).
  Any such term must be discussed first, with a diagnostic of how much it leaks the cap.
- **The floor in K-ETS**: check that it is a hard trading limit (orders below it
  rejected). `block` assumes that; `support` is the reserve-backed alternative.
- **Production**: v1 decides $\xi, \eta$ from the pre-trade state (one forward pass).
  v2 would feed the price $A_t$ to the production heads (a second, cheap forward pass
  after clearing). The trade head must stay price-free either way.
- **`A_REF`**: 6.05 (the observed price), used only as the pre-open value and
  tie-breaker of the selection rule.
- **Not in this document**: the major–minor formulation (paper type 2 as one large
  agent submitting its own schedule) and the merit-order electricity price. Both
  build on this clearing step; write them up separately once it works.
