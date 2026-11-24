# Major–minor market clearing — design and algorithm

Branch `major-minor-clearing`. This branch is **empty on purpose**: it holds only this
document. It is the design for the next model, built on top of the market-clearing
model of branch `market-clearing` (see its `MARKET_CLEARING.md`). Implement it
starting from `market-clearing` once that branch's code works.

This document is self-contained: it restates the clearing rules, then adds the major
players, the sampling rules and the full algorithm.

---

## 0. Summary

- **Who is major.** Paper type 2 (large public producers: about 11% of agents, but
  about 46% of trading capacity and 60% of production capacity at $\gamma^{\text{ref}}$)
  becomes $N_0$ **major players** (default $N_0 = 1$, the KEPCO group). Types 1
  (private producers) and 3 (financial institutions) stay **minor populations**.
- **Clearing.** Every participant submits a **demand schedule** (a trade as a function of
  the price). Exogenous traders supply a **price-elastic** net quantity. The price is the
  root of aggregate excess demand. The **price floor blocks the price**: when it binds,
  trades happen at the floor, buyers are served in full and sellers are **rationed**
  pro rata. The government never trades.
- **Minors** are price-takers. They see only public information.
- **Majors** internalize their effect on the price and on the fill ratio. They also see
  only their own state and public information.
- **Equilibrium.** Closed-loop Nash between the majors and the minor populations.
- **Algorithm.** Fictitious play. Minors: per-agent sampling of their mixture, best
  response against a frozen price path (as today). Majors: one component **drawn per
  market**, best response with the market **re-cleared inside the rollout** and the
  gradient going through the clearing.

---

## 1. Notation and units

| paper | code (`envs/base_params.py`) | role in this model |
|---|---|---|
| type 1, private producers | `j = 0` | minor population |
| type 2, large public producers | `j = 1` | **majors**, indexed $0$ (and $m = 1, \dots, N_0$) |
| type 3, financial institutions | `j = 2` | minor population |

- Minor types: $j \in \mathcal J = \{1, 3\}$. Simulated agent $i$ of type $j$ is $(j,i)$,
  following mixture component $l^{j,i}$.
- Majors: $m = 1, \dots, N_0$, written $(0,m)$; drop $m$ when $N_0 = 1$.
- Network parameters: $\vartheta^{j,l}$ for minors, $\vartheta^{0,l}$ for majors ($\theta$ is
  allowance holdings).
- **Units: everything is per unit of total mass.** The context weights are
  $\mathfrak n_1, \mathfrak n_2, \mathfrak n_3$ from `context_weights()` (functions of $\mathfrak n_3$ in $\gamma$).
  The majors share the mass $m_0 = \mathfrak n_2$; each major has mass $m_0 / N_0$, and its
  quantities are **aggregate** quantities of that mass:

$$
C^{f,0} = \frac{m_0}{N_0} C^{f,2}, \qquad C^{\xi,0} = \frac{m_0}{N_0} C^{\xi,2}, \qquad
\theta^{0,m}_0 = \frac{m_0}{N_0}\, \theta^{2}_0, \qquad \sigma^{\varepsilon,0} = \frac{m_0}{N_0}\, \sigma^{\varepsilon,2} .
$$

  A major's trade $f^{0,m}_t$ is therefore large compared with one minor's, and enters
  the clearing with weight 1.

---

## 2. Information

| player | observes | does not observe |
|---|---|---|
| minor $(j,i)$ | own state $Y^{j,i}_t$, public state $Z_t$, public history summary $H_t$ | majors' states, other agents' states, components drawn |
| major $(0,m)$ | own state $Y^{0,m}_t$, $Z_t$, $H_t$ | other majors' states, minors' states |

- Public state: $Z_t = \big(t,\ A^{\text{pre}}_t,\ \tilde A_{t-1},\ P_t,\ \gamma\big)$, with
  $A^{\text{pre}}_t$ the last price.
- Public history summary: $H_t = \big(\text{EMA}_t(A_s - A_{s-1}),\ \rho_{t-1},\ \text{EMA}_t(\text{volume}_s)\big)$.
  Price, fill ratio and volume are published by the exchange. These features are
  pre-trade, so schedules stay monotone in the candidate price.
- The majors can compute the minors' conditional distribution (a function of the
  common-noise path and of their own past actions); they need no access to individual
  minors.
- **Common noise** for the minors:

$$
\mathcal F^0_t = \sigma\big(\gamma,\ P_{0..t},\ \varepsilon^{0,2}_{1..t},\ (\varepsilon^{0,m}_{1..t})_m,\ (u^{0,m})_m\big),
$$

  including the majors' emission shocks $\varepsilon^{0,m}$ (one player: they do not
  average out) and the majors' drawn components $u^{0,m}$ (§5).

**Filtering.** Minors cannot see the majors' positions but could partly infer them from
prices. Their true best response may depend on the whole public history; $H_t$ is a
finite summary of it. The audit (§8) uses a recurrent deviator to test whether this
restriction binds.

---

## 3. Orders

### 3.1 Minors (types $j \in \{1,3\}$): price-taking schedules

$$
f^{j,i}_t(a) = \operatorname{clip}\Big( C^{f,j} \tanh\Big(\alpha^{j,i}_t - \beta^{j,i}_t\, \frac{a - A^{\text{pre}}_t}{A_{\text{scale}}}\Big),\ -\min\big(\theta^{j,i}_t, C^{f,j}\big),\ C^{f,j} \Big),
$$

$$
\big(\alpha^{j,i}_t,\ \beta^{j,i}_t,\ \hat\xi^{j,i}_t,\ \hat\eta^{j,i}_t\big) = \Phi_{\vartheta^{j,\,l^{j,i}}}\big(Y^{j,i}_t, Z_t, H_t;\ \gamma\big), \qquad \beta^{j,i}_t = \beta_{\min} + \operatorname{softplus}(\cdot) > 0 .
$$

The candidate price $a$ is **not** an input of the network, so $f^{j,i}_t$ is continuous
and nonincreasing in $a$ for any network output. Reservation price:
$R^{j,i}_t = A^{\text{pre}}_t + A_{\text{scale}}\, \alpha^{j,i}_t / \beta^{j,i}_t$.

### 3.2 Majors: strategic schedules

$$
f^{0,m}_t(a) = \operatorname{clip}\Big( C^{f,0} \tanh\Big(\alpha^{0,m}_t - \beta^{0,m}_t\, \frac{a - A^{\text{pre}}_t}{A_{\text{scale}}}\Big),\ -\min\big(\theta^{0,m}_t, C^{f,0}\big),\ C^{f,0} \Big),
$$

$$
\big(\alpha^{0,m}_t,\ \beta^{0,m}_t,\ \hat\xi^{0,m}_t,\ \hat\eta^{0,m}_t\big) = \Phi_{\vartheta^{0,\,u^{0,m}}}\big(Y^{0,m}_t, Z_t, H_t;\ \gamma\big), \qquad \beta^{0,m}_t = \operatorname{softplus}(\cdot) \ge 0 .
$$

Same form as the minors, with one difference: $\beta^{0,m}_t$ may reach 0, so a pure
quantity (a flat schedule) is reachable.

**Why a schedule and not a quantity.** The supply shock $\varepsilon^{0,2}_t$ is not known
when orders are submitted; it is revealed by the price. A schedule lets the major
condition its trade on the realized price, so it is strictly better than a quantity
(Klemperer and Meyer, 1989). A quantity is the special case $\beta^{0,m} = 0$.

### 3.3 Exogenous traders: price-elastic net supply

$$
S_t(a) = \sigma_S\, \varepsilon^{0,2}_t \sqrt{dt} + s_1\,\big(a - A_{\text{ref}}\big), \qquad s_1 > 0,
$$

positive when they sell. In rationing they are one more participant, with order
$-S_t(a)$. $s_1 > 0$ is required: it makes excess demand strictly decreasing even
when every $\beta^{0,m} = 0$, and it bounds the majors' price impact by $1/s_1$.

---

## 4. Clearing at day $t$

### 4.1 Excess demand

With $n$ simulated agents per minor type (§5),

$$
D_t(a) = \underbrace{\sum_{m=1}^{N_0} f^{0,m}_t(a)}_{\text{majors}} + \underbrace{\sum_{j \in \{1,3\}} \mathfrak n_j\, \frac{1}{n} \sum_{i=1}^{n} f^{j,i}_t(a)}_{\text{minors}} - S_t(a) .
$$

$D_t$ is continuous and strictly decreasing, with slope at most $-s_1$. Its root
$A^\star_t$ is unique and lies below the explicit bound

$$
A^{\text{hi}}_t = A_{\text{ref}} + \frac{N_0\, C^{f,0} + \sum_{j \in \{1,3\}} \mathfrak n_j C^{f,j} + \sigma_S \lvert \varepsilon^{0,2}_t \rvert \sqrt{dt}}{s_1} .
$$

### 4.2 Blocking floor and rationing (short-side rule)

$$
A_t = \max\big(A^{\text{floor}},\ A^\star_t\big).
$$

If $D_t(A^{\text{floor}}) \ge 0$, the floor does not bind: $A_t = A^\star_t$, $\rho_t = 1$, and every
order is executed in full.

If $D_t(A^{\text{floor}}) < 0$, the floor binds: $A_t = A^{\text{floor}}$. Evaluate every order at the
floor and split it into buy and sell volumes, majors and exogenous traders included:

$$
B_t = \sum_m \big(f^{0,m}_t\big)^+ + \sum_{j} \mathfrak n_j\, \frac{1}{n}\sum_i \big(f^{j,i}_t\big)^+ + \big(S_t\big)^-,
\qquad
O_t = \sum_m \big(f^{0,m}_t\big)^- + \sum_{j} \mathfrak n_j\, \frac{1}{n}\sum_i \big(f^{j,i}_t\big)^- + \big(S_t\big)^+,
$$

all at $a = A^{\text{floor}}$. Then $O_t > B_t$, and the fill ratio of sellers is

$$
\rho_t = \frac{B_t}{O_t} \in [0, 1) .
$$

### 4.3 Executed trades

For every participant $p$ (minor agent, major, exogenous traders):

$$
\hat f^{p}_t = \begin{cases} f^{p}_t(A_t) & \text{if } f^{p}_t(A_t) \ge 0, \\ \rho_t\, f^{p}_t(A_t) & \text{if } f^{p}_t(A_t) < 0 . \end{cases}
$$

Reward: $-\hat f^{p}_t A_t$. Holdings: $\theta^{p}_{t+1} = \theta^{p}_t + \hat f^{p}_t$.
Allowances are conserved exactly:

$$
\sum_m \theta^{0,m}_{t+1} + \sum_j \mathfrak n_j\, \tfrac{1}{n}\textstyle\sum_i \theta^{j,i}_{t+1}
= \sum_m \theta^{0,m}_{t} + \sum_j \mathfrak n_j\, \tfrac{1}{n}\textstyle\sum_i \theta^{j,i}_{t} + \hat S_t ,
$$

with $\hat S_t$ the executed exogenous net supply.

### 4.4 Blocked volume

$G_t = O_t - B_t \ge 0$ when the floor binds, $0$ otherwise. It satisfies
$D_t(A_t) + G_t = 0$, $G_t \ge 0$, $G_t (A_t - A^{\text{floor}}) = 0$, and
$\rho_t = 1 - G_t / O_t$. $G_t$ is **not** a trade: it is the volume of sell orders the floor
blocks. $A^{\text{floor}} \sum_t G_t$ is what a reserve-backed floor would cost.

A flag `floor_mode = "block" | "support"` keeps the alternative where the government
buys $G_t$ ($\rho_t = 1$), for comparison.

### 4.5 Timing and arrays

| array | index | meaning |
|---|---|---|
| `A_traj` `(T+1,)` | `[0]` | $A_{\text{ref}}$, the pre-open price (`A0_BASE` = 6.05) |
| | `[t+1]` | $A_t$, the price of day $t$ |
| `rho_traj` `(T+1,)` | `[0]` | 1 (unused) |
| | `[t+1]` | $\rho_t$ |

State slot `A` at day $t$ holds $A^{\text{pre}}_t$ = `A_traj[t]`. The running mean
`A_avg` averages `A_traj[1..t+1]`, so $\tilde A_T$ is the statutory average of the year.

---

## 5. Sampling the population and the majors

**Minors: one component per agent.** In each market, each minor type $j$ simulates the
same number $n$ of agents (e.g. $n = 128 \ge K_{\max}$). Agent $(j,i)$ follows

$$
l^{j,i} = \Big(o^j + \Big\lfloor \frac{i\, k}{n} \Big\rfloor\Big) \bmod k, \qquad o^j \sim \text{Unif}\{0, \dots, k-1\} \text{ per market},
$$

for the whole year. This is the Monte Carlo of the population control (each member
draws its own control) with stratification. The proportions enter only through
$\mathfrak n_j$, never through $n$.

**Majors: one component per major per market.** At $t = 0$, each major $m$ draws

$$
u^{0,m} \sim \text{Unif}\{0, \dots, k-1\}, \quad \text{independently across } m,
$$

and follows $\bar\pi^{0,\,u^{0,m}}$ for the whole year. This is the major's **mixed
strategy**: FP averages its best responses across markets, never inside a market.
Averaging inside a market would turn one player into a population and bring back the
price-taking approximation. The draws are common noise for the minors.

**Never draw a minor type's component per market**: that coordinates the whole type
and estimates a different object.

**Mixture buffers.** For every type (majors included), the mixture is stored in a fixed
buffer of `K_MAX` parameter slots; slot $l$ holds the best response of iteration $l$ and
$k$ (a traced integer) counts the filled slots. Indices are always below $k$, so empty
slots are never used, and the jitted functions compile once per solve.

---

## 6. Objectives and equilibrium

### 6.1 Minor of type $j$

Price-taker: it treats the paths $(A_t, \rho_t)_t$ as exogenous.

$$
V^j(\boldsymbol\pi^j; \Sigma^0, \Sigma, \gamma) = \mathbb E\Big[ \sum_{t=0}^{T-1} \Big( -\hat f^j_t A_t + (P_t \mathbf 1 - c^j) \cdot \xi^j_t \eta^j_t \Big) - \kappa^2 \tilde A_T \big(e^j \cdot \bar\xi^j_T + \bar\varepsilon^j_T - \theta^j_T\big)^+ \Big].
$$

### 6.2 Major $m$

Same reward and penalty in aggregate units, but **the paths are generated with its own
schedule inside the clearing**:

$$
V^{0}(\boldsymbol\pi^{0}; \Sigma^0_{-m}, \Sigma, \gamma) = \mathbb E\Big[ \sum_{t=0}^{T-1} \Big( -\hat f^{0,m}_t A_t + (P_t \mathbf 1 - c^0) \cdot \xi^{0,m}_t \eta^{0,m}_t \Big) - \kappa^2 \tilde A_T \big(e^0 \cdot \bar\xi^{0,m}_T + \bar\varepsilon^{0,m}_T - \theta^{0,m}_T\big)^+ \Big],
$$

where $A_t$, $\rho_t$, $\tilde A_T$ and the minors' states all respond to $\boldsymbol\pi^0$; the
other majors follow components drawn from $\Sigma^0$; the minors keep their feedback
schedules (they react through prices and their own states, but do not re-optimize).

Channels of market power the major can learn:

1. **Price impact:** buying raises $A_t$; it shades its trades.
2. **Fill at the floor:** its executed sale $\rho_t f^{0,m}_t$ grows with its own order and
   crowds the minors out of the limited demand.
3. **The K-ETS penalty index:** $\tilde A_T$ is an average the major moves; a major that
   will be short gains by depressing it.
4. **Information:** the minors cannot see its position.

### 6.3 Equilibrium (closed-loop Nash)

A pair $(\Sigma^0, \Sigma)$, with $\Sigma^0 = (\Upsilon^0, \sigma^0, \bar\pi^0)$ the majors' mixed
control and $\Sigma = (\Sigma^1, \Sigma^3)$ the minors' population controls, is an
equilibrium in scenario $\gamma$ if

1. **minors:** $J^j(\Sigma^0, \Sigma; \gamma) \ge V^j(\boldsymbol\pi^j; \Sigma^0, \Sigma, \gamma)$ for every $j \in \{1,3\}$ and
   every control $\boldsymbol\pi^j$ measurable with respect to the minor's information
   (§2), with the paths frozen;
2. **majors:** $J^0(\Sigma^0, \Sigma; \gamma) \ge V^0(\boldsymbol\pi^0; \Sigma^0_{-m}, \Sigma, \gamma)$ for every
   control $\boldsymbol\pi^0$ measurable with respect to the major's information, with the
   market re-cleared under the deviation.

Exploitabilities:

$$
\mathcal E^j = \sup_{\boldsymbol\pi^j} V^j - J^j \ (\text{frozen paths}), \qquad
\mathcal E^0 = \sup_{\boldsymbol\pi^0} V^0 - J^0 \ (\text{re-cleared market}).
$$

**Scenario extension.** $\gamma$ is drawn per market from $Q$ and every network is
conditioned on it (FiLM). The extended values of all players are $Q$-averages of
nonnegative gaps, so the almost-everywhere argument of the paper applies to the
majors unchanged.

**Number of majors.** $N_0 = 1$: one aggregate major (the KEPCO group), maximal market
power. $N_0 > 1$: competing generators, each internalizing its own, smaller impact.

---

## 7. Gradients through the clearing (majors only)

The minors' best responses use frozen paths, as today. The majors' best responses
differentiate through the clearing.

**Interior ($A^\star_t > A^{\text{floor}}$).** Implicit function theorem at the root: for any input
$x$ of the clearing (the deviating major's $\alpha^{0}_t, \beta^{0}_t$, its state, and, in the
full gradient, the minors' schedule parameters),

$$
\frac{\partial A_t}{\partial x} = -\frac{\partial_x D_t(A_t)}{D_t'(A_t)}, \qquad
D_t'(A_t) = \sum_m f^{0,m\,\prime}_t(A_t) + \sum_j \mathfrak n_j \tfrac{1}{n}\textstyle\sum_i f^{j,i\,\prime}_t(A_t) - s_1 \le -s_1 .
$$

The major's own slope $f^{0,m\,\prime}_t \le 0$ is in the denominator: a steeper schedule
moves the price less, a flatter one moves it more but cannot react to the shock.

**Floor binds.** $A_t = A^{\text{floor}}$ does not move; the gradient goes through
$\rho_t = B_t / O_t$, an explicit function of the orders at the floor. For a selling
major,

$$
\hat f^{0,m}_t = -\frac{B_t}{O^{-m}_t + (f^{0,m}_t)^-}\, (f^{0,m}_t)^-,
$$

increasing and concave in its order. Plain automatic differentiation handles it.

**Implementation.**
- Wrap the bisection in `jax.custom_jvp`. In the JVP rule, compute
  `jax.jvp(lambda x: D(A_t, x), (x,), (dx,))` at the fixed $A_t$ and divide by the
  analytic $D_t'(A_t)$; return 0 when the floor binds. Never differentiate through the
  bisection iterations.
- The switch "floor binds or not" is a `jnp.where`: the gradient is piecewise, which
  is correct.

**Two regimes for the major's gradient.**
- **Myopic:** `stop_gradient` on the minors' schedule parameters and states (and on
  the other majors'). The gradient goes only through the same-day clearing ($A_t$,
  $\rho_t$) and through the major's own state. Cost: about one population simulation
  per step, like a minor best response.
- **Full:** no `stop_gradient`. The gradient also goes through the minors' future
  reactions (their states and their inputs $A^{\text{pre}}$, $H$). Backpropagate through the
  whole scan with `jax.checkpoint` on the step, and a smaller batch.

Train with the myopic gradient; **evaluate** $\mathcal E^0$ by re-simulating the full
dynamics (no gradient needed). If a specialist trained with the full gradient beats the
myopic one clearly, switch to full training.

---

## 8. The algorithm

### Algorithm 1 — Clearing at day $t$ in one market

```
Input : majors' schedule parameters (alpha0_m, beta0_m, lo0_m, hi0_m), m = 1..N0
        minors' schedule parameters (alpha_ji, beta_ji, lo_ji, hi_ji), j in {1,3}, i = 1..n
        weights n_j, supply shock eps_t, floor A_floor, A_pre
Output: A_t, rho_t, G_t

1  D(a) = sum_m f0_m(a) + sum_j n_j * mean_i f_ji(a) - S_t(a)          # closed form
2  if D(A_floor) < 0:                                                   # floor binds
3      A_t   = A_floor
4      B_t   = buy volume at A_floor  (majors, minors, exogenous)
5      O_t   = sell volume at A_floor (majors, minors, exogenous)
6      rho_t = B_t / O_t ;  G_t = O_t - B_t
7  else:
8      A_t   = bisection root of D on [A_floor, A_hi]  (CLEAR_ITERS halvings)
9      rho_t = 1 ;  G_t = 0
10 if floor_mode == "support": rho_t = 1                                # government buys G_t
11 return A_t, rho_t, G_t
```

Steps 1–9 run unconditionally (results selected with `jnp.where`) so the function
`vmap`s over markets. One network forward pass per participant per day; the rest is
arithmetic.

### Algorithm 2 — Simulating one market for one year

```
Input : scenario gamma, common noise (P, eps0), majors' policies (pi0_m), minors' mixtures (buffers, k),
        component offsets o^j, idiosyncratic noise, optional: a deviating major replacing major 1
Output: A_traj, rho_traj, G_traj, rewards of every participant

1  initialise states: minors (n per type, holdings from Lambda^j), majors (aggregate holdings), A_pre = A_ref
2  draw l^{j,i} = (o^j + floor(i k / n)) mod k for every minor agent           # fixed for the year
3  for t = 0 .. T-1:
4      forward pass: majors -> (alpha0, beta0, xi0, eta0); minors -> (alpha, beta, xi, eta) via apply_gathered
5      (A_t, rho_t, G_t) = Algorithm 1
6      executed trades f_hat = f(A_t) for buyers, rho_t * f(A_t) for sellers
7      rewards  += -f_hat * A_t + production revenue
8      states   <- dynamics(state, f_hat, xi, eta, idiosyncratic shock, next_A = A_t, next_P = P_{t+1})
9      update public history H (EMA of price changes, rho_t, volume)
10 rewards -= terminal penalty kappa2 * A_avg_T * shortfall
```

### Algorithm 3 — Fictitious play with majors and minors

```
Input : prior Q, iterations K, batch sizes B (minors) and B0 (majors), gradient steps n_grad, n_grad0
Init  : buffers for j in {1,3} and for the majors; slot 0 = fresh networks
        (optional warm start of the major from the price-taking type-2 policy of `market-clearing`)
        k = 1

for iteration k = 1 .. K:

  # ── (a) minors' best responses: one per type j in {1,3}, price-taking ──────────────
  for step = 1 .. n_grad:
      draw B markets: gamma ~ Q, common noise, majors' components u^{0,m} ~ Unif{0..k-1},
                      minors' offsets o^j, idiosyncratic noise
      (A, rho) = Algorithm 2 for each market, with the current mixtures      # no gradient
      A, rho   = stop_gradient(A, rho)
      for j in {1,3} in parallel:
          roll out one deviating agent of type j per market against frozen (A, rho)
          loss_j = - mean reward / s_j ;  Adam step on vartheta^j

  # ── (b) majors' best response, market re-cleared in the loop ────────────────────────
  for step = 1 .. n_grad0:
      draw B0 markets as above; the deviating major replaces major 1,
                      majors 2..N0 draw u^{0,m}
      run Algorithm 2 with the deviating major inside the clearing
          (gradient through Algorithm 1 by the custom JVP of §7;
           myopic: stop_gradient on minors' and other majors' parameters and states)
      loss_0 = - mean reward of the deviating major / s_0 ;  Adam step on vartheta^0

  # (a) and (b) both best-respond to the mixtures of iteration k-1: run them in parallel.

  # ── (c) update the mixtures ──────────────────────────────────────────────────────────
  write the new networks into slot k of each buffer ;  k <- k + 1

  # ── (d) evaluate on frozen validation blocks ─────────────────────────────────────────
  minors : E^j = V^j(new BR) - J^j, both on the same frozen (A, rho); J^j exact over components
  majors : J^0 = stratified average over u^0 in {0..k-1} (same number of markets per component)
           V^0(new BR) by re-simulating each market with the BR inside the clearing
           E^0 = V^0 - J^0
  log the diagnostics of §9
```

**Notes.**
- The minors' best response is exactly today's, with $(A, \rho)$ instead of $A$.
- The price bank of `market-clearing` (clear many markets once per iteration, sample
  minibatches) applies to step (a) only: in step (b) the prices depend on the
  deviating major.
- $s_0$ is a fixed reward scale for the major (aggregate units), like $s_j$.

### Algorithm 4 — Audit

- **Minors:** specialist deviator at a fixed scenario, frozen paths (as today). Add a
  **recurrent** specialist (GRU over public history) to test the information restriction
  of §2; it never takes part in clearing, so it needs no monotonicity.
- **Majors:** specialist major at a fixed scenario, trained with the **full** gradient,
  evaluated by re-simulation. Its gain over the equilibrium major measures what the
  myopic training leaves on the table.

---

## 9. Diagnostics

| key | meaning |
|---|---|
| `E_major`, `E_minor[j]` | exploitabilities (relative to $s_0$, $s_j$) |
| `major_impact` | mean $\partial A_t / \partial f^{0}_t$ along equilibrium paths, interior days |
| `major_markdown` | mean $R^{0}_t - A_t$: gap between the major's valuation and the price |
| `major_fill_share` | at the floor: share of executed sales captured by the majors vs their share of sell orders |
| `major_profile` | majors' net trades by month, compared with a price-taking major (same networks, frozen-price best response) |
| `penalty_index_effect` | $\tilde A_T$ with the strategic major minus with the price-taking major |
| `A_on_floor`, `rho_mean_on_floor`, `blocked_volume[j]`, `support_cost` | as in `market-clearing` |
| `clear_residual_max`, `price_drift`, `A0_cleared` | as in `market-clearing` |
| `emissions`, `revenue[j]`, `revenue_major` | the outcomes a regulator reads, with and without market power |

---

## 10. Tests

1. **Clearing with majors.** A flat major schedule ($\beta^0 = 0$) shifts $D_t$ by a
   constant; $A^\star_t$ matches a dense-grid root.
2. **IFT gradient.** $\partial A_t / \partial \alpha^0$ and $\partial A_t / \partial \beta^0$ from the custom JVP
   match central finite differences on interior days.
3. **Rationing gradient.** At the floor, $\partial \hat f^0 / \partial \alpha^0$ matches finite differences.
4. **Conservation** with majors, in both floor modes.
5. **Major sampling.** Over many markets, each major's component frequency is $1/k$;
   inside a market, each major follows exactly one component.
6. **Price-taking limit.** With `stop_gradient` on $A_t$ and $\rho_t$ in the major's
   rollout, the major's best response equals a frozen-path best response.
7. **Vanishing mass.** As $m_0 \to 0$, prices converge to those of `market-clearing`
   without type 2.

---

## 11. Roadmap

1. Implement and validate `market-clearing` first (its `MARKET_CLEARING.md`).
2. **Diagnostic.** Train one major best response (myopic, then full) against the
   converged `market-clearing` equilibrium, where type 2 is a price-taking population.
   Its gain over the price-taking type-2 policy measures the market power left on the
   table. If it is small, report it and keep the mean-field model.
3. If the gain is significant: full Algorithm 3 with $N_0 = 1$.
4. Robustness: $N_0 = 5$ competing generators, against $N_0 = 1$.
5. Audits of Algorithm 4.

## 12. Paper changes

- §2: type 2 becomes $N_0$ major players; the information structure of §2; the
  clearing of §4 (schedules, elastic supply, blocking floor with rationing).
- Definitions: majors' mixed controls sampled per market; minors' population controls
  sampled per agent; closed-loop Nash (§6.3); exploitabilities with frozen paths for
  minors and re-clearing for majors.
- §3: Algorithms 1–3, the IFT gradient, myopic vs full training.
- §4: the market-power results — price impact, fill crowding at the floor, manipulation
  of the K-ETS penalty index $\tilde A_T$ — and the comparison $N_0 = 1$ against $N_0 = 5$.

## 13. Open choices

- **$N_0$** and whether the KEPCO generators coordinate (one major) or compete.
- **The majors' objective:** profit, as here, or cost minimization under a supply
  obligation; KEPCO settlement coefficients affect their revenues (check how they apply).
- **Public history features $H_t$:** which summaries, and whether minors also see the
  last fill ratio and volume (exchange data suggest yes).
- **The K-ETS floor:** check it is a hard trading limit (`block`); `support` is the
  alternative.
- **Calibration** of $\sigma_S$ and $s_1$ (market depth): they bound the majors' price
  impact.
- **Not in this document:** the merit-order electricity price; production stays as in
  `market-clearing` v1.
