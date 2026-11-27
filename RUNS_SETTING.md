# Runs for the K-ETS paper: scenario levers, the four jobs, and the LP benchmark

Self-contained specification for the coding agent. Code lives in `PAPER/OR-K-ETS/` (JAX, see its `README.md`). Context and paper-level decisions are in `REVISION_PLAN_OR.md` (same folder); this file only specifies what to run. Written 2026-11-27.

Two findings are targeted: (1) the market power of the major (the large public producer), (2) the effect of market makers. Jobs 1–2 serve the findings; jobs 3–4 are the exact validation.

---

## 1. Participants

Three labels, total mass 1:

| Label | Role | Mass | Production capacity C^ξ per unit of mass |
|---|---|---|---|
| private producers | minor type (mean field) | 𝔫^priv = 0.80 − mm_share | 100 |
| public producer | the **major**, one aggregate player | m⁰ = 0.20 (fixed) | 700 (aggregate capacity 700·m⁰ = 140) |
| market makers | minor type, no production | 𝔫^mm = mm_share | 0 |

Market makers emit nothing: C^ξ = 0 **and** uncontrolled emission noise ν^mm = δ₀. They receive no free allocation.

Penalty: K-ETS rule, κ(Ã) = κ²·Ã with κ¹ = 0, Ã the running mean of the allowance price over the compliance year (`app:penalty` of the paper).

---

## 2. Scenario levers (jobs 1 and 2)

γ = (c, s, A^floor, mm_share, κ²). The prior Q is uniform on the box. Each market of a training batch draws its own γ; the networks receive γ through FiLM.

| Lever | Meaning | Range |
|---|---|---|
| c | coverage ratio of the total cap | [0.3, 1] |
| s | share of the total cap given to the major | [0, 1] |
| A^floor | price floor | [0, 15] |
| mm_share | mass of market makers | [0, 0.5] |
| κ² | penalty multiplier | [1, 6] |

**Total cap** (emissions at full capacity with the dirtiest technology):

  Θ_tot = c · T · ( 700·m⁰·e^{0,fuel} + 100·𝔫^priv·e^{priv,fuel} ),

with e^{ℓ,fuel} the emission rate of the dirtiest technology of label ℓ.

**Allocation.**
- Major: θ⁰ = s·Θ_tot (in the aggregate units of the major's state).
- Private producers: (1 − s)·Θ_tot in total, i.e. (1 − s)·Θ_tot / 𝔫^priv per unit of mass; use the intra-type law Λ^priv of γ_ref rescaled to this mean.
- Market makers: 0.

**Fixed at γ_ref.** Ceiling, trading caps C^f per label, technologies (costs c^{ℓ,k}, emission rates e^{ℓ,k}), good-price dynamics, emission noise of the producers, exogenous supply, horizon T.

**Plotting variable (not a lever).** The major's capacity share

  s_cap(mm_share) = 140·e^{0,fuel} / ( 140·e^{0,fuel} + 100·𝔫^priv·e^{priv,fuel} )

(= 0.64 at mm_share = 0 and 0.82 at mm_share = 0.5 if the two fuel rates are equal). Plot against x = s / s_cap: x < 1 the major is relatively short (net buyer), x > 1 relatively long (net seller), x = 1 same coverage as the private producers.

**Checks on the box before training.**
- At s = 0, T·C^{f,0} must cover the major's need (it receives nothing); at s = 1, the same for the private producers. Report if not.
- Report the share of the box where the ceiling binds most days (expected at small c).

---

## 3. The four jobs

| # | Job | Setting | Seeds | Output |
|---|---|---|---|---|
| 1 | Strategic FP | full model, prior of Section 2; major trained through the clearing (current algorithm) | per seed | the equilibrium; Findings 1 and 2 |
| 2 | Competitive-benchmark FP | full model, **same prior, same seeds, same validation markets** as job 1; the major is a **price taker** (pathwise gradient at fixed market paths, as for minor types; no gradient through the clearing) | per seed | competitive counterfactual; distortion Δ_A = mean price (job 1) − mean price (job 2) at the same γ and common noise |
| 3 | Deterministic FP, **no major** | Section 4 | per seed | compared with job 4 |
| 4 | LP | Section 5 | once (deterministic) | exact competitive equilibrium |

### Validation grids for jobs 1 and 2 (evaluation, not training)

Reference values for the levers not varied in a slice: c_ref, s_ref, A^floor_ref, mm_share_ref = 0.4, κ²_ref = 3 (c_ref, s_ref, A^floor_ref from the calibration of γ_ref).

| Slice | Varied | Values | Purpose |
|---|---|---|---|
| A | s × mm_share | s ∈ {0, 0.1, …, 1} (11) × mm_share ∈ {0, 0.1, …, 0.5} (6) | Hahn curve (Δ_A vs x) and market makers × major heatmap |
| B | κ² × s | κ² ∈ {1, 2, 3, 4, 5, 6} × s ∈ {0, 0.2, 0.4} (short major) | penalty multiplier and market power |
| C | A^floor × s | A^floor ∈ {0, 5, 10, 15} × s ∈ {0, 0.1, …, 1} | does the floor contain market power |
| D | c × mm_share | c ∈ {0.3, 0.4, …, 1.0} (8) × mm_share ∈ {0, 0.1, …, 0.5} | scarcity regimes; price drift and liquidity vs market makers |

At each grid point: 100 validation markets, metrics with 95% intervals clustered by market. Metrics: mean price and Δ_A (overall and by day); major's terminal position, shortfall and cumulative trade path; price drift E[A_{t+k} − A_t | F⁰_t] for k ∈ {1, 10, 30}; traded volume, days at the floor, blocked volume, fill ratios; profit per label, penalties paid, total emissions. Also the single-scenario exploitability at the 16 corners of the (c, s, A^floor, mm_share) box with κ² = 3, and at κ² ∈ {1, 6} for the reference point.

**Invariance check on job 2:** at fixed (c, A^floor, mm_share, κ²), the job-2 price must be flat in s (allocation invariance under competition). Report it on slice A.

---

## 4. Job 3: deterministic setting, no major

Everything as in jobs 1–2 except:

- **No major.** Remove the public producer from the market (no label (0,1), no major order in the clearing). Masses: 𝔫^priv = 0.80 − mm_share, 𝔫^mm = mm_share (they sum to 0.8; only relative weights matter in the clearing).
- **Cap and allocation:** Θ_tot = c · T · 100 · 𝔫^priv · e^{priv,fuel}, all of it to the private producers, **deterministic**: Λ^priv = δ at Θ_tot / 𝔫^priv per unit of mass.
- **No noise at all:**
  - good prices follow their mean path, P_t := E[P_t] (deterministic);
  - no uncontrolled emissions: ε ≡ 0 for every label;
  - no exogenous supply: S_t ≡ 0.
- **Trading caps C^f large enough never to bind** (check after the run that no order sits at ±C^f).
- **Ceiling high enough never to bind.**
- **Floor fixed at A^floor = 1** (must stay below the LP price λ*(c) on the whole c range; see Section 5). It removes the zero-price equilibrium that the indexed penalty with κ¹ = 0 always allows.
- **κ² ∈ [1.5, 6].** The LP equivalence needs κ² > 1 strictly (being short must cost more than buying); 1.5 keeps a margin.

**Prior for job 3:** uniform on c ∈ [c_lo, c_hi] × mm_share ∈ [0, 0.5] × κ² ∈ [1.5, 6], where [c_lo, c_hi] is the range on which the LP price satisfies λ*(c) ≥ 2 (twice the floor) and the cap binds (λ*(c) > 0). **Run job 4 first to get this range.**

**Validation grid for job 3:** c ∈ 7 equally spaced values in [c_lo, c_hi] × mm_share ∈ {0, 0.25, 0.5} × κ² ∈ {1.5, 3, 6} (63 points), 20 markets each (the market is deterministic except the draw of mixture components by the particles).

**Expected invariances (they are checks):** the job-3 equilibrium must depend on c only. mm_share changes nothing (market makers end with no allowances; the private producers' problem is per unit of mass), and κ² > 1 changes nothing (nobody is short in equilibrium).

---

## 5. Job 4: the LP

### Why it is the exact equilibrium of job 3
With no noise, a floor, and κ² > 1, a short position costs κ²·Ã > A, so no firm ends short and the cap is met exactly; the competitive equilibrium maximizes total production profit under the cap, and the allowance price is the dual of the cap constraint, constant in t. Mass cancels (one producer type), so the LP is written **per unit of mass of private producers** and depends on c only.

### Formulation
Technologies k of the private type, days t ∈ {0, …, T−1}, variables q^k_t ≥ 0 (production with technology k on day t, per unit of mass):

  maximize   Σ_{k,t} (P_t − c^{priv,k}) · q^k_t
  subject to Σ_k q^k_t ≤ 100  for every t   (capacity)
             Σ_{k,t} e^{priv,k} · q^k_t ≤ c · T · 100 · e^{priv,fuel}   (cap; dual variable λ)

P_t is the deterministic good-price path of job 3; c^{priv,k}, e^{priv,k}, T as in the code (γ_ref).

Solve with HiGHS: `scipy.optimize.linprog(method="highs")`; read the cap's dual from `res.ineqlin.marginals` (sign convention: λ* = −marginal for a maximization written as minimization of the negative objective).

### Cross-check (merit order, no solver)
For a price λ ≥ 0, each day choose the technology maximizing P_t − c^{priv,k} − λ·e^{priv,k}; produce 100 if that margin is > 0, else 0. Emissions E(λ) are a nonincreasing step function. Then λ* = inf{λ ≥ 0 : E(λ) ≤ c·T·100·e^{priv,fuel}}, found by bisection. It must agree with the LP dual. If the cap falls exactly at a step of E, the dual is an interval [λ⁻, λ⁺] = the two prices bounding the step; report the interval.

### Values to run
- c ∈ {0.30, 0.31, …, 1.00} (71 values).
- Everything else fixed at the job-3 values (private technologies, P_t path, T, capacity 100).

### Outputs per c
- λ*(c) (or [λ⁻, λ⁺]);
- total production per technology over the year, Σ_t q^k_t (daily schedules are not unique when margins tie across days);
- total emissions (= cap when it binds) and total production profit (LP objective).

Use the curve λ*(c) to set [c_lo, c_hi] for job 3 (λ* ≥ 2 and cap binding).

---

## 6. Comparison of jobs 3 and 4 (benchmark B0 of the paper)

At each point of the job-3 validation grid:
- **Price:** job-3 mean price over days and markets vs λ*(c); it must lie in [λ⁻, λ⁺] up to a stated tolerance, and be flat in t.
- **Production:** job-3 total production per technology (per unit of mass) vs the LP.
- **Compliance:** total emissions vs the cap; fraction of firms short (should be ≈ 0).
- **Profit:** job-3 total production profit vs the LP objective.
- **Quality:** exploitability of job 3 at the same points.
- **Invariances:** results flat in mm_share and in κ².

Deliverable: one table or figure with relative errors against c, with separate lines for mm_share ∈ {0, 0.25, 0.5} and κ² ∈ {1.5, 3, 6}.
