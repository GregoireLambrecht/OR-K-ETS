# K-ETS Bayesian-extended mean-field game — fictitious play and its audit

**Authors:** Soh Young In, Heeyeon Kim, Grégoire Lambrecht, Mathieu Laurière,
Luis Seco. Algorithms developed by G. Lambrecht and M. Laurière; code
developed by G. Lambrecht.

Three agent types (private producers, large public producers, market makers)
trade emission allowances over T = 365 days in the Korean ETS. The market is a
multi-type mean-field game solved by **fictitious play**: at each iteration a
fresh best response is trained against the empirical mixture of all past best
responses, and the quantity that must decrease is the **exploitability**
E_j = V(best response) − V(mixture).

The game is *Bayesian-extended*: every market in a training batch draws its own
regulatory context

    (init_alloc, kappa, Afloor, financial_inst)

and the policy is conditioned on it through a FiLM network, so one solve covers
a whole family of regulatory settings. Three studies differ only in the width
of the price-floor box, `FLOOR_MAX` ∈ {20, 30, 40}; each has two modes,
`choice` (producers pick their technology) and `bau` (technology fixed).

The **PMFG audit** re-measures exploitability at fixed policy scenarios against
a free, unconditional specialist trained at that one context. The gap
E_PMFG − E_BE is the *price of generality* paid by the conditional policy.

The folder is self-contained (`envs/`, `utils.py`, `helper_plot.py`). Run
everything from inside it.

## 1. Install

Python 3.14, JAX on an NVIDIA GPU (CUDA 13). A GPU is needed for every step;
step 3 is the only long one.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # jax[cuda13], equinox, optax, numpy, pandas, matplotlib, seaborn
```

`requirements.txt` pins the core packages; `requirements-freeze.txt` is the
full `pip freeze` of the environment the paper was produced with.

## Shipped snapshot: trained policies and audit

You do not have to run the 30-hour solves. `artifacts/` holds the outputs the
paper was produced from (141 MB):

| archive | restores | produced by |
|---|---|---|
| `mm_warmstart.tar.gz` | `mm_warmstart_r3/mm_policy.eqx` | step 2 |
| `policies_floor{20,30,40}.tar.gz` | `results_floor*/{choice,bau}/` — 51 policy triples, `progress.json`, convergence and deviator sheets | step 3 |
| `audit.tar.gz` | `results_floor*/*/audit/<scenario>/b12000/k*/` — the PMFG audit cells, and `paths.json` | step 4 |

```bash
bash artifacts/unpack.sh                 # restores everything into place
```

Then pick up the pipeline wherever you want:

```bash
# (a) only redraw the audit tables and figures from the shipped cells
python3 collect_audit.py results_floor20/choice results_floor20/bau \
        results_floor30/choice results_floor30/bau results_floor40/choice results_floor40/bau
python3 plot_audit_sheet.py results_floor20 results_floor30 results_floor40

# (b) audit the shipped policies at a new scenario (add it to audit_scenarios() in audit_pmfg.py)
python3 audit_pmfg.py --run-dir results_floor30/choice --scenario my_scenario --checkpoint 50 --iters 12000

# (c) continue fictitious play from the shipped policies (resumes at iteration 51)
KETS_FLOOR_MAX=30 python3 main_rung3.py choice --model-dir results_floor30/choice --nb-policy 80 \
        --iters 12000 --lr 0.001 --lr-schedule cosine --lr-min 0.0001
```

The solve resumes from the last `policy_<k>_type*.eqx` found in `--model-dir`,
so (c) appends iterations 51–80; with `--nb-policy 50` it exits immediately.

## 2. Market-maker warm start (~5 min) — skip if you unpacked the snapshot

```bash
python3 pretrain_mm_r3.py --steps 6000 --batch 4096 --lr 1e-3
```

Writes `mm_warmstart_r3/mm_policy.eqx`. The market maker starts at θ₀ = 0
against a clip, a zero-gradient trap; this regresses its FiLM net onto a
buy-then-sell heuristic first. One warm start serves every study and mode.

## 3. Fictitious play (~30 h per run on one GPU) — skip if you unpacked the snapshot

`FLOOR_MAX` is read from the environment variable `KETS_FLOOR_MAX` **once, at
import**, by `envs/base_params.py`; set it before Python starts, and use a
different `--model-dir` per study.

```bash
for F in 20 30 40; do
  for MODE in choice bau; do
    KETS_FLOOR_MAX=$F python3 main_rung3.py $MODE --model-dir results_floor$F/$MODE \
        --nb-policy 50 --t-grid 365 --iters 12000 --lr 0.001 --lr-schedule cosine --lr-min 0.0001 \
        --batch 256 --verify-samples 500 --fixed-samples 200 --eval-samples 32
  done
done
```

50 fictitious-play iterations; each trains a best response for 12 000 Adam
steps on fresh common noise, batch 256, cosine learning rate 1e-3 → 1e-4.
Exploitability is measured every iteration on frozen blocks: 500 draws from
the context prior plus 200 draws at each of nine fixed contexts (the reference
and eight one-at-a-time deviations).

Writes `results_floor<F>/<mode>/`: one policy triple per iteration
(`policy_<k>_type<j>.eqx`), `progress.json` (exploitability per block and
iteration, price-path movement), `losses/`, and `figs/` (convergence sheet,
deviator sheet, per-iteration sheets). The run is resumable: rerun the same
command and it continues from the last saved iteration.
`python3 main_rung3.py -h` lists the other options.

## 4. PMFG audit (~30 min per cell on one GPU) — skip if you unpacked the snapshot

One cell = one (run, scenario, checkpoint). The scenarios are defined in
`audit_scenarios()` in `audit_pmfg.py`:

| scenario | (init_alloc, kappa, Afloor, financial_inst) |
|---|---|
| `current_kets` | (0.9, 3, 6, 0.4) |
| `low_alloc` | (0.1, 3, 6, 0.4) |
| `high_tax` | (0.9, 5, 6, 0.4) |
| `no_market_maker` | (0.9, 3, 6, 0.0) |
| `high_floor_mid_alloc` | (0.5, 3, FLOOR_MAX − 5, 0.4) — a different market in each study |

```bash
# every scenario x checkpoints 5, 10, ..., 50 for one run (50 cells; parallelise as you like)
for S in current_kets low_alloc high_tax no_market_maker high_floor_mid_alloc; do
  for K in 5 10 15 20 25 30 35 40 45 50; do
    python3 audit_pmfg.py --run-dir results_floor30/choice --scenario $S --checkpoint $K --iters 12000
  done
done
python3 audit_paths.py --run-dir results_floor30/choice      # price-path movement at each scenario, ~4 min
```

Checkpoint k means: mixture = triples 0…k−1, BE deviator = `policy_k`. The
audit trains an unconditional specialist against that mixture with the same
budget (12 000 steps, batch 256, fresh common noise each step), then scores
both deviators on one frozen block of 250 draws. `audit_pmfg.py` reads
`FLOOR_MAX` from the run's own `progress.json` before importing anything, so
no environment variable is needed. Writes
`results_floor<F>/<mode>/audit/<scenario>/b12000/k<NNN>/{result.json, loss.npz, policy_type*.eqx}`
and `audit/<scenario>/paths.json`.

## 5. Tables and figures (~5 min)

```bash
python3 collect_audit.py results_floor20/choice results_floor20/bau \
        results_floor30/choice results_floor30/bau results_floor40/choice results_floor40/bau
python3 plot_audit_sheet.py results_floor20 results_floor30 results_floor40
```

`collect_audit.py` prints, per run, the true exploitability and the price of
generality at every checkpoint and per type at the last one, flags cells whose
audit best response is under-trained (negative gap), and writes
`figs_audit/audit_scenarios.{pdf,png}` (final-checkpoint bars across runs).
`plot_audit_sheet.py` writes one sheet per study,
`figs_audit/audit_sheet_results_floor<F>.{pdf,png}`: rows = the BE-MFG solve
on its prior block then each audit scenario; columns = exploitability per type,
population-weighted exploitability (PMFG and BE deviators), price-path
movement. Every figure comes as PDF and PNG.

## Layout

| path | role |
|---|---|
| `envs/` | environment (`environment.py`), parameters (`base_params.py`), networks (`models.py`) |
| `utils.py` | frozen-noise blocks, jitted best-response stages, exploitability, the FP loop (rungs 1–3; rung 3 is the study) |
| `helper_plot.py` | house style, per-iteration and convergence sheets |
| `main_rung3.py`, `pretrain_mm_r3.py`, `audit_pmfg.py`, `audit_paths.py`, `collect_audit.py`, `plot_audit_sheet.py` | entry points, one per step |
| `diagnostics.py` | rollout diagnostics for the per-iteration sheet |
| `artifacts/` | compressed snapshot of the trained policies and audit, plus `unpack.sh` |
| `figs_audit/` | the paper's audit figures (PDF + PNG) |

`results_floor*/`, `mm_warmstart_r3/` and `logs/` are outputs and are
git-ignored; `artifacts/` is the tracked copy of the ones the paper used, and
`figs_audit/` holds the final figures as committed.
