"""Market clearing, VERSION 2 (MARKET_CLEARING.md, revision 7be619c).

Kept apart from envs/base_params.py so the v1 runs (A_CEIL corridor, block
population) keep running unchanged on the same branch.

What changes from v1:
  * exogenous supply S_t = sigma_S(gamma) eps sqrt(dt), sigma_S(gamma) = c Theta(gamma)/sqrt(365),
    Theta(gamma) = sum_j w_j theta^j_0(gamma) (c = SUPPLY_C = 1 %): the yearly cumulative
    shock has std c Theta whatever T; price-independent (rev. 7be619c
    removed the price-elastic S1 term: it leaked the cap and pinned the price);
  * corridor [floor, A_CEIL_V2], clearing interval [a-, a+] by two bisections,
    A_t = clip(A_pre, a-, a+);
  * the bounds BLOCK the price: the long side is rationed pro rata (sellers
    rho_sell at the floor, buyers rho_buy at the ceiling); G = O - B = blocked
    volume, not a government trade ('block'); 'support': the government takes G;
  * per-agent population sampling: N_SIM agents per type and market, agent i
    follows the stratified component l = (o + floor(i k / n)) mod k of the
    mixture, which sits in a fixed buffer of size K_MAX (compile once per solve).
"""
import os
from envs.base_params import A0_BASE, CAP_F_BASE, MEAN_MARKET_IMPACT

A_REF        = A0_BASE        # pre-open price A^pre_0 (tie-breaker of the selection rule)
A_CEIL_V2    = float(os.environ.get('KETS_A_CEIL', 200.0))   # artificial ceiling (USD), existence only; must essentially never bind
BETA_MIN_V2  = 0.1            # floor on the schedule slope (normalised price units)
CLEAR_ITERS_V2 = int(os.environ.get('KETS_CLEAR_ITERS', 40))   # bisection halvings per side on [A_floor, A_CEIL_V2]; float32 saturates at ~24
SIGMA_S      = 0.05 * CAP_F_BASE * MEAN_MARKET_IMPACT              # LEGACY fixed scale (runs before the scenario scaling)
SUPPLY_C     = float(os.environ.get('KETS_SUPPLY_C', 0.01))        # sigma_S(gamma) = c Theta(gamma) / sqrt(365)
FLOOR_MODE   = os.environ.get('KETS_FLOOR_MODE', 'block')          # 'block' | 'support'
assert FLOOR_MODE in ('block', 'support'), FLOOR_MODE

POP_SAMPLING = 'agent'        # v2 always samples per agent; 'block' = agent with n = k N_j, offset 0
N_SIM        = int(os.environ.get('KETS_N_SIM', 128))               # simulated agents per type and market
K_MAX        = int(os.environ.get('KETS_K_MAX', 200))               # size of the fixed mixture buffer
