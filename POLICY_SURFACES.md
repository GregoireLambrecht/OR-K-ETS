# Policy surfaces for the major–minor market (four figures)

Spec for the coding agent. It adds **evaluation code only**: no change to the
training (fictitious play, best responses, clearing) and no change to the
scenario box. It produces the four market figures of the paper, one surface per
figure, from trained rung-3 runs (`main_majmin.py --rung 3`, one run per seed).

Code indexing (do not confuse with the paper's): `MINORS = (0, 2)` are the
private producers (paper type 1) and the financial institutions (paper type 3);
`MAJOR = 1` is the large public producer (the major, paper superscript 0).
The scenario is `g = (init_alloc, kappa, Afloor, mm_share)` (`MajorMinorEnv.bind`).

| Fig. | Metric | x axis | y axis | Fixed at `CTX_REF` | Band |
|---|---|---|---|---|---|
| 1 | average price $\bar A$ | `init_alloc` ∈ [0, 1.2] | `kappa` ∈ [1, 6] | `Afloor`, `mm_share` | seeds |
| 2 | rationed volume at the floor $\mathcal V_{floor}$ | `init_alloc` ∈ [0, 1.2] | `Afloor` ∈ [FLOOR_MIN, FLOOR_MAX] | `kappa`, `mm_share` | seeds |
| 3 | price distortion of the major $\mathcal L$ (v1: blind major) | `init_alloc` ∈ [0, 1.2] | `Afloor` ∈ [FLOOR_MIN, FLOOR_MAX] | `kappa`, `mm_share` | common noise |
| 4 | price volatility $\sigma_A$ | `mm_share` ∈ [0, 0.5] | `Afloor` ∈ [FLOOR_MIN, FLOOR_MAX] | `init_alloc`, `kappa` | common noise |

All four grids are inside the training box (`envs/base_params.py`:
`ALLOC ∈ [0, 1.2]`, `KAPPA ∈ [0.8, 7.2]`, `FLOOR ∈ [FLOOR_MIN = 6, FLOOR_MAX]`,
`MM ∈ [0, 0.5]`), so **nothing is retrained** except the blind major of Fig. 3.
Default resolution 13 × 13 (CLI argument). The fixed coordinates are `CTX_REF`
(`utils.py`) unless overridden on the command line.

---

## 1. Inputs and loading

- `--run-dirs`: one trained rung-3 run per seed (same box, same `T`, same `n`).
- Read `progress.json` of each run and set, **before importing JAX / the envs**,
  exactly as `plot_majmin_scenarios.py` does:
  `KETS_FLOOR_MODE = prog['floor_mode']`, `KETS_N_SIM = prog['n_sim']`,
  `KETS_USE_H = prog['public_history_input']`.
  `KETS_FLOOR_MAX` is read at import in `base_params.py` and is not stored in
  `progress.json`: take it from `--floor-max` and assert it equals the value used
  in training (add it to `meta` of future runs if convenient).
- Assert `floor_mode == 'block'` (rationing). In `'support'` mode there is no
  rationing and Fig. 2 has a different meaning.
- Load the **final mixture** with all saved slots, exactly as in
  `plot_majmin_scenarios.py` (`K = len(prog['records'])`, slots `0..K`,
  `make_buffer` / `buffer_set`, `make_sim`). `k = K + 1` components.
- All runs must have the same `T` and `n`; assert it.

## 2. Common noise protocol (essential)

The bands and the pairings are only meaningful if every simulation sees **the
same states of the world**.

- Draw the markets **once**, with a fixed key `NOISE_KEY = PRNGKey(NOISE_SEED + 7)`
  (any fixed value, distinct from `VERIFY_SEED` and from training keys), with
  `draw_markets(env, NOISE_KEY, S, T, n, prior)` and `S = 512` (CLI). Keep
  `(P, eps0, imaj, imin, ck)`; **discard `gam`** and replace it, at each grid
  point, by the grid scenario broadcast to `(S, 4)`.
- Reuse these same draws:
  1. at **every grid point** (common random numbers: only `g` changes; the supply
     shock is `supply(eps0[t])`, scaled by `sigma_S(g)` inside the env, so the
     standardised `eps0` is shared);
  2. for **every seed** (so the seed band only measures the disagreement of the
     equilibria, and the common-noise aggregation of §5 can average over seeds
     realisation by realisation);
  3. for the **equilibrium and the counterfactual** markets of Fig. 3 (pairing).
- Components: the major follows component `u = s mod k` in market `s`
  (stratified, as in the evaluation of `utils_majmin_fast.py`); the minors follow
  the stratified components built from `ck[s]` inside `simulate`.
- Market `s` is the same state of the world everywhere; index it by `k`
  (realisation) in the formulas below.

## 3. Quantities to record per market

One call of `sim(maj_buf, minor_buf, k, g, P[s], eps0[s], imaj[s], (imin[0][s], imin[1][s]), ck[s], u[s], False)`
returns `(A_traj, rho_traj, H, r0, ys, (Xf, X0f))`, vmapped over `s` in chunks
(as `sim_many` in `plot_majmin_scenarios.py`).

**Traded prices.** `ys['A']` has shape `(T,)`: the prices of days `0..T-1`.
`A_traj[0]` is the pre-open `A_REF` and is **not** a traded price; never use it.

**One diagnostic addition to `MajorMinorEnv.market_step`.** Add the total bids and
offers at the cleared price to the per-day output dict `y`:

```python
'B': B, 'O': O,
```

`B` and `O` are already computed there (rationing block). This is output only:
`y` does not enter any loss, so training is unchanged. Do not modify anything else
in `market_step`.

Per grid point `g`, seed `l`, market `k`, record (as float32 numpy arrays):

| name | shape | definition |
|---|---|---|
| `A` | `(T,)` | `ys['A']` |
| `on_floor` | `(T,)` | `ys['on_floor']` |
| `G` | `(T,)` | `ys['G']` (= `O - B` at a binding bound, 0 otherwise; ≥ 0 at the floor) |
| `O` | `(T,)` | `ys['O']` |
| `r0` | `()` | the major's total reward |

Store raw per-market arrays; all aggregations are done in the plotting script (§6).

## 4. Metrics

### Fig. 1 — average price (the original metric of the paper)

$$
\bar A^{l,k}(g) = \frac1T \sum_{t=0}^{T-1} A^{l,k}_t .
$$

Same definition as the paper (`eq:metric_price`), with the traded prices
`ys['A']`.

### Fig. 2 — rationed volume at the floor

When the floor binds, supply exceeds demand at the floor price and the offers
not executed are $G_t = O_t - B_t \ge 0$ (sellers filled at $\rho^-_t = B_t/O_t$).
The metric is the share of all offers of the year left unexecuted at the floor:

$$
\mathcal V^{l,k}_{floor}(g) = \frac{\sum_{t} G^{l,k}_t\,\mathbb 1\{A^{l,k}_t = A^{floor}\}}{\sum_{t} O^{l,k}_t},
\qquad
\bar{\mathcal V}^{\,l}_{floor}(g) = \frac{\sum_k \sum_t G^{l,k}_t\,\mathbb 1\{\cdot\}}{\sum_k \sum_t O^{l,k}_t}
$$

(the surface uses the ratio of sums over markets, $\bar{\mathcal V}^{\,l}$, not the
mean of the per-market ratios). Use `on_floor` for the indicator, not a float
comparison of `A` with `Afloor`. Guard the denominator (`max(·, 1e-12)`).

Also report, for the text (not as a surface):
- the **time spent on the floor**, the share of the year at which the floor binds,
  $$
  \tau^{l,k}_{floor}(g) = \frac1T\sum_{t=0}^{T-1}\mathbb 1\{A^{l,k}_t = A^{floor}\}
  $$
  (indicator = `on_floor`; one step is `365/T` days, so $\tau$ is also the
  number of days on the floor divided by 365): its mean over markets and seeds,
  and the 2.5 / 50 / 97.5 % quantiles over `k` of $\hat\tau^{\,k} = \frac1L\sum_l \tau^{l,k}$
  (seeds averaged first, as in §5). It replaces the probability of hitting the
  floor, which saturates at 0 or 1 over most of the grid and ignores how long
  the floor binds;
- the 2.5 / 50 / 97.5 % quantiles over `k` of the per-market $\mathcal V^{l,k}_{floor}$ (seeds averaged first).

Note: on day 0 the price is the root closest to the floor (`sel = Afloor` at
`t = 0` in `market_step`); `on_floor` is true only if the floor actually binds
(excess supply at the floor), so day 0 is not counted on the floor by
construction. Use `on_floor`, never `A == Afloor`.

### Fig. 3 — price distortion of the major, **version 1: blind major** (quick check)

$$
\mathcal L^{l,k}(g) = \frac{\bar A^{l,k}(g) - \bar A^{\,blind,l,k}(g)}{\bar A^{\,blind,l,k}(g)} ,
$$

where $\bar A^{\,blind,l,k}(g)$ is the average price of the **same market**
(same `P, eps0, imaj, imin, ck`, same minors' mixture and components) in which the
major's mixture is replaced by the blind major $\pi^{0,blind}$ of seed `l`
(§4.1), the market being **re-cleared** with it:
`sim(blind_net, minor_buf, k, g, ..., u[s], True)` (`use_br=True`, no gradient).

- $\mathcal L > 0$: the strategic major raises the price compared with a major
  that ignores its price impact (withholding as a net seller).
- $\mathcal L < 0$: it lowers the price (holding back purchases as a net buyer).
- Expected (Hahn 1984): sign change where the major's allocation equals its
  emissions, i.e. near `init_alloc = 1` if the theoretical need is met; and a
  floor that truncates the negative side.

**Status.** v1 is not an equilibrium counterfactual: the minors keep the mixture
they learnt against the strategic major. It is a quick check of the sign and the
shape of the surface. Version 2 (§4.2) replaces it later.

Also record, on the same markets, the major's value in both markets (re-cleared
in both): $\Delta J^{0} = r_0^{\,mix} - r_0^{\,blind}$ (value of internalizing the
price impact, minors fixed), reported in the text.

#### 4.1 Training the blind major (v1)

The blind major is the major's best response **as a price taker** against the
final equilibrium: it treats $(A_t, \rho_t)_t$ as given, exactly as a minor does.

- **New env method** `MajorMinorEnv.rollout_major(policy, A_traj, rho_traj, H_traj, P_traj, idio_maj, ctx)`,
  mirroring `rollout_minor` for one aggregate player:
  - initial state `self.major_initial_state()` (aggregate units, × `m0`);
  - `idio_maj` in aggregate units: `imaj_raw * self.major_mass()` (as in `make_sim`);
  - at day `t`: `A_exec = A_traj[t + 1]`, `rho = rho_traj[t + 1]`,
    `out = policy(self.net_input(X, H_traj[t]), ctx)`,
    `a = self.act_at(X, out, A_exec)`, executed trade
    `a[0] * rho[1]` if buying, `a[0] * rho[0]` if selling,
    reward `self.running_reward(X, a, A_exec)`, next state
    `self.single_step_dynamics(X, a, idio_maj[t], A_exec, P_traj[t + 1])`;
  - plus `self.terminal_reward(X_T)`. Returns the scalar total reward.
- **Consistency test (required, add to `tests/`).** For a market cleared with
  major component `u`, `rollout_major(component u, A_traj, rho_traj, H, P, imaj_raw * m0, ctx)`
  must return the major's `r0` of that market up to float tolerance (the major's
  executed trades at the cleared price are what the rollout reproduces). This
  also checks that `act_at` and `_order(sched(·))` give the same trade for the
  major's state. If it fails, stop and report: do not adjust the test.
- **Training data: the price bank.** Build the bank as in
  `utils_majmin_fast.py` (`bank_chunk_fn` / `build_bank`) with the **final**
  buffers and `k = K + 1`, prior `env.generate_context` (the whole box, so the
  blind major is FiLM-conditioned like every other network), major component
  `u ~ U{0..k-1}` per market. Bank fields: `gam, P, A, rho` (+ `H` if `USE_H`).
  The blind major's own idiosyncratic noise is drawn fresh at every step.
- **Loss.** `-mean(rollout_major(...)) / s0`, with `s0 = prog['s0']` (the scale
  used in training), batch `B0 = 256` markets per step, `iters0 = 4000` steps,
  cosine lr `1e-3 → 1e-4` (same optimiser as `make_opt(n_it, warm=False)`).
  Gradients flow only through the blind major's own network and states; `A`, `rho`
  are data.
- **Initialisation.** From the run's last major best response
  `policy_{K}_type1.eqx` (the policy class is the same; only the price awareness
  differs). Optional check (flag `--fresh`): also train from a fresh `major_policy`
  and report both values; they should agree.
- **Output.** `<run-dir>/counterfactual/blind_major.eqx` and
  `<run-dir>/counterfactual/blind_major.json` (losses: first and last 5 % mean;
  settings; the bank size; the seed).
- **Check on the evaluation block** (`'prior'` block of `utils_majmin_fast.py`,
  frozen paths): frozen-path value of the blind major vs. frozen-path value of the
  major's mixture (`rollout_major` of each component, averaged with `u = s mod k`).
  The blind major must be ≥ the mixture up to one standard error (it is a best
  response on frozen paths). Report the paired difference and its standard error.

Script: `blind_major.py --run-dir <dir> [--iters 4000 --batch 256 --bank 65536 --fresh]`.

#### 4.2 Version 2 (later — do not implement now)

The proper counterfactual C0: a **full fictitious play** in which the major is a
price taker. Same code as `fictitious_play_majmin_fast` except the major's stage,
which trains on the price bank with `rollout_major` (like the minors), and the
major's exploitability, which is evaluated on frozen paths (a price-taking
deviation). Same box, same seeds. Fig. 3 v2 then uses $\bar A^{C0}$ computed on the
same noises (§2). The metric and the plotting are unchanged.

### Fig. 4 — price volatility

On market `k`, with the traded prices $A_0, \dots, A_{T-1}$ and the time step
$\Delta t = 1/T$ year (`env.dt = 365/T` days):

$$
r^{l,k}_t = \log\frac{A^{l,k}_t}{A^{l,k}_{t-1}},\ t = 1..T-1,
\qquad
\sigma^{l,k}_A(g) = \sqrt{T}\ \operatorname{std}\big(r^{l,k}_1, \dots, r^{l,k}_{T-1}\big)\quad(\texttt{ddof=1}).
$$

Annualised volatility: $\sigma_A = 0.4$ means a one-standard-deviation price move of
about ±40 % over a year. Prices are ≥ `Afloor` ≥ `FLOOR_MIN` > 0, so the log is
defined (assert `A > 0`). Days on the floor give $r_t = 0$: a binding floor lowers
$\sigma_A$ mechanically (to be read with Fig. 2; nothing to correct in the code).

## 5. Aggregations and bands

Two aggregations, as in the paper (`subsec:seed-consistency`,
`subsec:common-noise-sensitivity`):

- **Seed band** (Figs. 1, 2): per seed, aggregate over the markets
  ($\bar o^{\,l}(g)$: the mean over `k` for $\bar A$; the ratio of sums for
  $\mathcal V_{floor}$); then the surface is the mean over seeds and the envelope
  is ± 1 standard deviation over seeds. Message: the equilibria of the seeds agree.
- **Common-noise band** (Figs. 3, 4): per market, average over seeds first,
  $\hat o^{\,k}(g) = \frac1L\sum_l o^{l,k}(g)$ (valid because market `k` is the same
  state of the world for every seed, §2); then the surface is the mean over `k`
  and the envelope is the empirical 2.5 % and 97.5 % quantiles of
  $(\hat o^{\,k}(g))_k$. Message: different events occur under the same rules.
  For Fig. 3, $\mathcal L^{l,k}$ is computed market by market (paired) before any
  averaging.
- Appendix: the seed envelopes of Figs. 3 and 4 (same files, other aggregation).

## 6. Scripts and outputs

1. `blind_major.py` (§4.1), once per run.
2. `policy_surfaces.py --run-dirs <d1> ... <dL> --out <study>/surfaces [--S 512 --res 13 --floor-max ...]`:
   for each figure, each grid point, each seed, simulate the `S` markets (and the
   blind-major markets for Fig. 3) and save
   `<study>/surfaces/fig{1..4}.npz` with
   - `x`, `y` (grid axes), `fixed` (the fixed coordinates), `seeds`, `S`, `T`;
   - the raw per-market arrays of §3 needed by that figure, shape
     `(L, nx, ny, S, ...)` (Fig. 3: both the equilibrium and the blind `A`, and
     both `r0`).
   Fig. 2 and Fig. 3 share the grid: simulate it once and write both files.
3. `plot_policy_surfaces.py --surfaces <study>/surfaces`: computes the metrics and
   the aggregations of §5 from the npz, draws one surface per figure with the
   style of `plot_sensitivity_seeds.py` (`_surface`), and writes
   `fig{1..4}.{pdf,png}` and `summary.json` (the text numbers: $\tau_{floor}$, the
   quantiles of $\mathcal V_{floor}$, $\Delta J^0$ at `CTX_REF`, the blind-major
   check of §4.1).

Cost: 169 grid points × 512 markets × L seeds per figure, forward simulation only;
chunk the vmap (as `sim_many`, chunks of 100–500 markets).

## 7. Sanity checks (print them; fail loudly)

- `tests/`: the `rollout_major` consistency test (§4.1).
- At `CTX_REF`, $\bar A$ (mean over markets of `ys['A']`) agrees with `A_mean` of
  `plot_majmin_scenarios.py` up to Monte Carlo error.
- Clearing quality at every grid point: `max |ys['resid']|` and
  `max |ys['cons_err']|` small; print the worst grid point.
- $\mathcal V_{floor} = 0$ wherever `on_floor` is never true; $0 \le \mathcal V_{floor} \le 1$;
  $0 \le \tau_{floor} \le 1$, and $\tau_{floor} = 0 \Leftrightarrow \mathcal V_{floor} = 0$ market by market.
- $\sigma_A$ finite everywhere.
- The noise arrays are bitwise identical across grid points, seeds and the two
  markets of Fig. 3 (hash them once and assert).

## 8. Do not

- Do not modify the training code, the clearing, the box or `CTX_REF`.
- Do not use `A_traj[0]` (pre-open) in any metric.
- Do not redraw the noise per grid point, per seed or per counterfactual.
- Do not implement version 2 of Fig. 3 yet.
