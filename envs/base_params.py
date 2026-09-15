import jax
import jax.numpy as jnp


KRW_TO_USD = 0.0007
GJ_TO_MHW_COAL = 8
GJ_TO_MHW_GAS = 6

EF_BASE = 0.85
EG_BASE = 0.37
CF_BASE = 47500*KRW_TO_USD
CG_BASE = 96600*KRW_TO_USD
KAPPA_BASE = 3

BM_FUEL = 0.8
BM_GREEN = 0.4
BM_UNIFORM = 0.4

BM_RATIO = {'large': 0.75, 'base':0.13}


CAP_XI_BASE = 100
CAP_F_BASE = CAP_XI_BASE*EF_BASE*1.1
SIGMA_EPS_BASE = CAP_XI_BASE*EF_BASE/20

A0_BASE = 8637*KRW_TO_USD

THETA_P = 0.01
MU_P = 79
SIGMA_P = 6
P0 = 75


def generate_prices_ou(key, T):
    # D4: the shock must scale as sqrt(dt), like the drift scales as dt.
    # Without it the total variance of the path is 365 * sigma^2 / T instead of
    # 365 * sigma^2, so the good-price volatility changes with the discretisation
    # and the multigrid rungs T in {10, 30, 50} are not the same game.
    dt = 365/T
    shocks = jax.random.normal(key, (int(T),))

    def step(current_p, epsilon):
        next_p = (current_p
                  + THETA_P * (MU_P - current_p) * dt
                  + SIGMA_P * epsilon * jnp.sqrt(dt))
        next_p = jnp.maximum(next_p, 0.0)
        return next_p, next_p

    _, price_path = jax.lax.scan(step, P0, shocks)
    return jnp.concatenate([jnp.array([P0]), price_path])


P_SCALE_BASE = P0
A_SCALE_BASE = A0_BASE


THETA0_100_BASE = CAP_XI_BASE*EF_BASE
A_FLOOR_NULL = 0
A_FLOOR_BASE = A0_BASE

BIG_ON_PRIVATE = 18/100
BIG_MARKET_IMPACT = 7
MEAN_MARKET_IMPACT = 7*BIG_ON_PRIVATE + 1*(1-BIG_ON_PRIVATE)

def market_impact_base(f):
    coeff = 2
    return f/(CAP_F_BASE*MEAN_MARKET_IMPACT*coeff)

def white_noise_A_base(key, T):
    return jax.random.normal(key, (T,)) * 0.25

def idiosyncratic_noise_base(key, sigma_eps, T):
    return jnp.abs(jax.random.normal(key, (T,)) * sigma_eps)

#Theta0,ef,eg,cf,cg,cap_f,cap_xi,sigma_eps,tec_control
PRIVATE_GENERATOR = [THETA0_100_BASE, EF_BASE, EG_BASE, CF_BASE, CG_BASE, CAP_F_BASE, CAP_XI_BASE, SIGMA_EPS_BASE, -1.0]
TEC_PRIVATE = 21/154

BIG_PUBLIC_GENERATOR = [7*THETA0_100_BASE, EF_BASE, EG_BASE, CF_BASE, CG_BASE, CAP_F_BASE*7, CAP_XI_BASE*7, SIGMA_EPS_BASE*7, -1.0]
TEC_PUBLIC = 141/188

MARKET_MAKER = [0, EF_BASE, EG_BASE, CF_BASE, CG_BASE, CAP_F_BASE, 1e-5, 1e-5, 0]

agent_templates = {
    'base': PRIVATE_GENERATOR,
    'large': BIG_PUBLIC_GENERATOR,
    'mm': MARKET_MAKER
}

import json
import os
from itertools import product

def generate_scenarios(scenarios_folder, base_config, sweep_params):
    os.makedirs(scenarios_folder, exist_ok=True)
    keys = sweep_params.keys()
    values = sweep_params.values()
    combinations = list(product(*values))
    for i, combo in enumerate(combinations):
        scenario = base_config.copy()
        name_parts = []
        for key, value in zip(keys, combo):
            scenario[key] = value
            name_parts.append(f"{key}_{value}")
        scenario_name = "_".join(name_parts)
        file_path = os.path.join(scenarios_folder, f"{scenario_name}.json")
        with open(file_path, 'w') as f:
            json.dump(scenario, f, indent=4)
    print(f"Successfully generated {len(combinations)} scenarios in '{scenarios_folder}'")


FP_WINDOW = 200   # sliding window size for fictitious play mixture

# ── kappa sweep range for the article study ──────────────────────────────────
# Training range is wider than the study range so kappa=0 is interior
# (not a boundary), forcing the network to learn the sell-vs-buy crossover:
# at kappa=0 allowances have no terminal value so the optimum is to dump them
# and burn fossil (P-cf = 41.75 vs P-cg = 7.38); above some kappa the firm
# holds allowances and switches green. Study range [0, 6].
# Narrowed from -1.2 to -0.5: padding of 0.5 already makes 0 interior, while
# -1.2 spent 14.3% of the prior (vs 6.5%) on a regime that is never studied.
KAPPA_MIN = -0.5
KAPPA_MAX =  7.2

# ── init_alloc training range ─────────────────────────────────────────────────
# D6: must NOT go below 0. theta_0 = 34*T*init_alloc for type 1 (238*T for
# type 2), so init_alloc < -2.75/T makes theta_0 < -cap_f and the admissible
# set [-min(theta, cap_f), cap_f] EMPTY. Unlike kappa < 0, which is a padded
# regime, init_alloc < 0 is an undefined one.
ALLOC_MIN = 0.0
ALLOC_MAX = 1.2

# ── Afloor training range ─────────────────────────────────────────────────────
# The price floor is a regulatory instrument, so it joins the context. The lower
# end is A0 (= A_FLOOR_BASE = 6.05): a floor below the initial price never binds
# at t=0 and the market is free to fall, which is the status quo. The upper end
# is the experiment — the study widens it in stages (20, then 30, then 40) and
# keeps the widest range whose exploitability still converges.
FLOOR_MIN = 6.0
# Set once, at import, from KETS_FLOOR_MAX. It must NOT be patched after import:
# `utils._G_HI` (the FiLM normaliser) and FIXED_SCENARIOS['floor_hi'] both
# capture it at import time, while environment.generate_context reads it at call
# time — patching later would leave the sampler and the normaliser disagreeing,
# silently training on a box the conditioner does not map to [-1, 1].
FLOOR_MAX = float(os.environ.get('KETS_FLOOR_MAX', 20.0))

# ── financial_inst (market-maker proportion) training range ───────────────────
# Enters the market as an aggregation WEIGHT, not as a population size — see
# ExogenousMarketEnvJAX.context_weights. Capped at half the block: past that the
# producers it is supposed to intermediate for stop being the market.
MM_MIN = 0.0
MM_MAX = 0.5

# ═════════════════════════════════════════════════════════════════════════════
# Rung 1 configuration  (see plan.md §6)
# Everything frozen except the policies: one scenario, one common-noise
# realisation, one population idiosyncratic set.
# ═════════════════════════════════════════════════════════════════════════════
GAMMA_REF = (0.5, 3.0)      # (init_alloc, kappa) — the single frozen PMFG
RUNG1_T_GRID    = (10, 30, 50)      # multigrid horizons; final horizon is 50
RUNG1_T_ITERS   = (2000, 2000, 2000)
RUNG1_T_LR      = (1e-3, 5e-4, 5e-4)
RUNG1_B         = 256       # BR training batch, per type, per gradient step
RUNG1_M         = 256       # paired rollouts per type for the exploitability
RUNG1_K         = 5         # FP iterations
RUNG1_WINDOW    = 200       # >= K, so the sliding window never binds
RUNG1_AGENTS    = 100       # agents per policy-block (type 2 gets 11, not 2)
NOISE_SEED      = 20260101  # seed for the frozen common noise / population idio
