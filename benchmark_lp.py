"""Exact benchmark B0 (benchmark_lp.tex): the deterministic competitive market and
its planner LP, and the exact best response along a computed market.

Under Assumption 1 (no majors, no noise, fixed fine kappa^1, floor 0, large trading
limits) the equilibrium price a* is the dual variable of the cap in the planner's
LP (P), computed exactly as the smallest minimiser of the convex piecewise-affine
Phi(a) = a Theta^c + sum_j n_j C^xi_j sum_t mubar^j_t(a) over [0, kappa^1]
(Phi is evaluated on its breakpoints; `planner_lp` cross-checks with HiGHS).
`best_response_lp` is the LP of Remark 3: the value of the best response of a
type-j agent to given deterministic prices and fill ratios (A_t, rho_t).

Conventions (as the environment, envs/environment.py):
  day t = 0..T-1 uses the good price P_traj[t] and trades at the price cleared
  that day, A_traj[t + 1]; production q_t = xi_t (eta_t, 1 - eta_t) over
  (fossil, gas), xi_t <= C^xi; a type with a fixed technology share eta (the
  financial institutions: eta = 0) has the single technology eta e_f + (1-eta) e_g;
  holdings theta_{t+1} = theta_t + executed trade; the order lies in
  [-min(theta_t, C^f), C^f], executed trade = rho^+ f^+ - rho^- f^-; terminal
  penalty kappa^1 (E_T - theta_T)^+.
Everything is numpy / scipy; the environment is only read.
"""
import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


# ── data of one scenario, read from the environment ──────────────────────────

def market_data(env, g, types=(0, 2)):
    """Scenario g = (init_alloc, kappa^1, Afloor, mm_share) -> dict of numpy arrays.
    env: a MajorMinorEnv built with KETS_DETERMINISTIC=1, KETS_PENALTY=fixed."""
    import jax
    import jax.numpy as jnp
    from envs.base_params import DETERMINISTIC, PENALTY_MODE
    assert DETERMINISTIC and PENALTY_MODE == 'fixed', 'the benchmark needs Assumption 1'
    eb = env.bind(jnp.asarray(g, jnp.float32))
    T = int(env.T)
    P = np.asarray(eb.generate_P(jax.random.PRNGKey(0), T), np.float64)
    w = np.asarray(eb.context_weights(), np.float64)
    ap = np.asarray(env.agent_params, np.float64)
    out = {'T': T, 'p_traj': P, 'kappa': float(g[1]), 'floor': float(g[2]), 'g': tuple(float(x) for x in g),
           'types': []}
    for j in types:
        if w[j] <= 0:
            continue
        ef, eg, cf, cg, capf, capxi, tech = ap[j, 1], ap[j, 2], ap[j, 3], ap[j, 4], ap[j, 5], ap[j, 6], ap[j, 8]
        if tech < 0:                    # chooses its technology mix
            e, c = np.array([ef, eg]), np.array([cf, cg])
        else:                           # fixed share tech of fossil
            e = np.array([tech * ef + (1 - tech) * eg])
            c = np.array([tech * cf + (1 - tech) * cg])
        theta = float(np.asarray(eb.type_initial_state(j))[1])
        out['types'].append(dict(j=j, w=float(w[j]), theta=theta, Cxi=float(capxi), Cf=float(capf),
                                 e=e, c=c, eps=0.0, tech=float(tech), scale=float(env.reward_scale(j))))
    out['Theta0'] = sum(d['w'] * d['theta'] for d in out['types'])
    out['Theta'] = out['Theta0']                                # no exogenous supply: s_t = 0
    out['Theta_c'] = out['Theta'] - T * sum(d['w'] * d['eps'] for d in out['types'])
    return out


def p_days(m):
    return m['p_traj'][:m['T']]


# ── the planner: Phi, its breakpoints, a*, V* ────────────────────────────────

def margins(d, p, a):
    """mu^{j,k}_t(a), (T, m_j)."""
    return p[:, None] - d['c'][None, :] - a * d['e'][None, :]


def rent(d, p, a):
    """mubar^j_t(a) = max(0, max_k mu), (T,)."""
    return np.maximum(0.0, margins(d, p, a).max(axis=1))


def phi(m, a):
    p = p_days(m)
    return a * m['Theta_c'] + sum(d['w'] * d['Cxi'] * rent(d, p, a).sum() for d in m['types'])


def breakpoints(m):
    """Prices in [0, kappa^1] where some cell (j, t) changes its merit order, plus 0, kappa^1."""
    p, K = p_days(m), m['kappa']
    pts = [0.0, K]
    for d in m['types']:
        e, c = d['e'], d['c']
        for k in range(len(e)):
            if e[k] > 0:
                pts += list((p - c[k]) / e[k])                 # shutdown of k
            for k2 in range(len(e)):
                if k2 != k and e[k] != e[k2]:
                    pts.append((c[k2] - c[k]) / (e[k] - e[k2]))  # switch k -> k2
    pts = np.unique(np.clip(np.asarray(pts, np.float64), 0.0, K))
    return pts


def merit_order(m, a):
    """Production per agent of every type at price a (ties: the cleaner technology /
    no production, i.e. the right limit), its emissions e^mo(a) (aggregate, controlled)."""
    p = p_days(m)
    qs, emo = [], 0.0
    for d in m['types']:
        mu = margins(d, p, a)
        best = mu.argmax(axis=1)
        on = mu.max(axis=1) > TOL
        q = np.zeros_like(mu)
        q[np.arange(len(p)), best] = np.where(on, d['Cxi'], 0.0)
        qs.append(q)
        emo += d['w'] * float((q * d['e'][None, :]).sum())
    return qs, emo


def value_star(d, p, a, T):
    """V^j_*(a) = a theta^j + C^xi sum_t mubar^j_t(a) - a T eps^j  (Theorem 1 (c))."""
    return a * d['theta'] + d['Cxi'] * rent(d, p, a).sum() - a * T * d['eps']


def planner(m, rtol=1e-9):
    """a* (smallest minimiser of Phi), the interval I*, the regime, V*_j(a*), Phi(a*)."""
    pts = breakpoints(m)
    vals = np.array([phi(m, a) for a in pts])
    vmin = vals.min()
    tol = rtol * max(1.0, abs(vmin))
    arg = pts[vals <= vmin + tol]
    lo, hi = float(arg.min()), float(arg.max())
    a = lo
    K = m['kappa']
    regime = 'cap_slack' if a <= TOL else ('fine' if a >= K - 1e-9 * max(K, 1) else 'interior')
    p = p_days(m)
    V = {d['j']: value_star(d, p, a, m['T']) for d in m['types']}
    return {'a_star': a, 'I_star': (lo, hi), 'unique': hi - lo <= 1e-7 * max(1.0, K),
            'regime': regime, 'phi': float(vmin), 'V_star': V,
            'total_reward': float(vmin)}        # sum_j n_j V_j = Phi(a*) - a sum_t s_t, s = 0


def planner_lp(m):
    """(P) by HiGHS: value and the cap's dual (the price), as a cross-check of `planner`."""
    p, T = p_days(m), m['T']
    cols, obj, cap_row, n = [], [], [], 0
    blocks = []
    for d in m['types']:
        mj = len(d['e'])
        blocks.append((n, mj))
        for t in range(T):
            for k in range(mj):
                obj.append(-d['w'] * (p[t] - d['c'][k]))       # minimise the negative
                cap_row.append(d['w'] * d['e'][k])
        n += T * mj
    obj.append(m['kappa']); cap_row.append(-1.0)                # shortfall sigma
    nv = n + 1
    A_ub = [np.asarray(cap_row)]
    b_ub = [m['Theta_c']]
    for (start, mj), d in zip(blocks, m['types']):
        for t in range(T):
            r = np.zeros(nv); r[start + t * mj: start + (t + 1) * mj] = 1.0
            A_ub.append(r); b_ub.append(d['Cxi'])
    res = linprog(np.asarray(obj), A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
                  bounds=[(0, None)] * nv, method='highs')
    assert res.status == 0, res.message
    return {'value': -res.fun, 'a_dual': -float(res.ineqlin.marginals[0]), 'x': res.x}


def equilibrium_paths(m):
    """Trajectories of the equilibrium of Theorem 1 (i), per type (dict by j):
    q (T, 2) production per agent over (fossil, gas) from the primal of (P) (the
    split of the marginal cells is the solver's); E (T+1,) cumulative emissions;
    theta (T+1,) post-trade holdings of the existence proof: terminal holdings
    theta_T^j (= E^j in the interior regime), reached by the uniform trades
    F^j / T (w_t = 1/T, s = 0). Only the price, the production of the non-marginal
    cells and theta_T are determined; the timing of the trades is not."""
    T = m['T']
    x = planner_lp(m)['x']
    out, n = {}, 0
    for d in m['types']:
        mj = len(d['e'])
        q = x[n:n + T * mj].reshape(T, mj)
        n += T * mj
        q2 = np.zeros((T, 2))
        if d['tech'] < 0:
            q2[:] = q
        else:                           # fixed share: split the single technology
            q2[:, 0], q2[:, 1] = d['tech'] * q[:, 0], (1 - d['tech']) * q[:, 0]
        e_day = q @ d['e'] + d['eps']
        out[d['j']] = {'q': q2, 'E': np.concatenate([[0.0], np.cumsum(e_day)])}
    Estar = sum(d['w'] * out[d['j']]['E'][-1] for d in m['types'])
    for d in m['types']:
        Ej = out[d['j']]['E'][-1]
        thT = Ej + (m['Theta'] - Estar) if Estar <= m['Theta'] else m['Theta'] / Estar * Ej
        out[d['j']]['theta'] = d['theta'] + (thT - d['theta']) * np.arange(T + 1) / T
    return out


# ── the exact best response along a computed market (Remark 3) ───────────────

def best_response_lp(m, d, A, rho_sell, rho_buy, return_path=False):
    """Value of the best response of type d along deterministic prices A (T,) (the price
    cleared on day t) and fill ratios rho_sell = rho^-, rho_buy = rho^+ (T,).
    Variables: q (T x m_j), executed trade fh (T), holdings th (T+1), shortfall vs."""
    p, T, mj = p_days(m), m['T'], len(d['e'])
    iq = lambda t, k: t * mj + k
    ifh = lambda t: T * mj + t
    ith = lambda t: T * mj + T + t
    ivs = T * mj + T + T + 1
    nv = ivs + 1
    c = np.zeros(nv)
    for t in range(T):
        for k in range(mj):
            c[iq(t, k)] = -(p[t] - d['c'][k])
        c[ifh(t)] = A[t]
    c[ivs] = m['kappa']
    Aeq, beq = [], []
    r = np.zeros(nv); r[ith(0)] = 1.0; Aeq.append(r); beq.append(d['theta'])
    for t in range(T):
        r = np.zeros(nv); r[ith(t + 1)] = 1.0; r[ith(t)] = -1.0; r[ifh(t)] = -1.0
        Aeq.append(r); beq.append(0.0)
    Aub, bub = [], []
    for t in range(T):
        r = np.zeros(nv); r[[iq(t, k) for k in range(mj)]] = 1.0
        Aub.append(r); bub.append(d['Cxi'])                     # capacity
        r = np.zeros(nv); r[ifh(t)] = -1.0; r[ith(t)] = -rho_sell[t]
        Aub.append(r); bub.append(0.0)                          # fh >= -rho^- theta_t
    r = np.zeros(nv)
    for t in range(T):
        for k in range(mj):
            r[iq(t, k)] = d['e'][k]
    r[ith(T)] = -1.0; r[ivs] = -1.0
    Aub.append(r); bub.append(-T * d['eps'])                    # vs >= E_T - theta_T
    bounds = ([(0, None)] * (T * mj)
              + [(-rho_sell[t] * d['Cf'], rho_buy[t] * d['Cf']) for t in range(T)]
              + [(None, None)] * (T + 1) + [(0, None)])
    res = linprog(c, A_ub=np.asarray(Aub), b_ub=np.asarray(bub), A_eq=np.asarray(Aeq),
                  b_eq=np.asarray(beq), bounds=bounds, method='highs')
    assert res.status == 0, res.message
    val = -res.fun
    if not return_path:
        return val
    x = res.x
    q = x[:T * mj].reshape(T, mj)
    return val, {'q': q, 'fh': x[T * mj:T * mj + T], 'theta': x[T * mj + T:T * mj + 2 * T + 1],
                 'shortfall': x[ivs]}


def path_reward(env, g, j, q, fh, A_traj):
    """Total reward of a type-j agent with production q (T, m_j) and executed trades fh (T,)
    along A_traj (T+1,), computed with the ENVIRONMENT's own reward and dynamics: the
    units check between the LPs and the simulator (tests/test_benchmark_lp.py)."""
    import jax
    import jax.numpy as jnp
    eb = env.bind(jnp.asarray(g, jnp.float32))
    T = int(env.T)
    P = eb.generate_P(jax.random.PRNGKey(0), T)
    x = eb.type_initial_state(j)
    tech = float(env.agent_params[j, 8])
    tot = 0.0
    for t in range(T):
        xi = float(q[t].sum())
        eta = (float(q[t, 0]) / xi if xi > 0 else 0.0) if tech < 0 else tech
        a = jnp.array([fh[t], xi, eta], jnp.float32)
        tot += float(eb.running_reward(x, a, A_traj[t + 1]))
        x = eb.single_step_dynamics(x, a, 0.0, A_traj[t + 1], P[t + 1])
    return tot + float(eb.terminal_reward(x))
