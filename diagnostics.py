"""Collect the per-iteration trajectories that plot_rung1 draws.

Separated from utils.py so the FP loop stays about the algorithm and the plots
can also be regenerated offline from saved checkpoints.
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from envs.environment import N_TYPES
from envs.models import stack_policies


def collect_iteration_diagnostics(env_T, history, new_triple, noise, statics=None):
    """Mixture vs new-best-response trajectories, at env_T's horizon.

    Both sides are rolled against the SAME aggregate price path — the one the
    mixture itself generates — because that is exactly the comparison the
    exploitability makes: the best response is a deviation against the
    incumbent population, not a separate world.

    Returns a dict of numpy arrays ready for plot_rung1.plot_iteration.
    """
    if statics is None:
        statics = tuple(stack_policies([h[j] for h in history])[1]
                        for j in range(N_TYPES))
    stacked = tuple(stack_policies([h[j] for h in history])[0]
                    for j in range(N_TYPES))

    mix = env_T.ensemble_trajectory(
        stacked, statics, noise['P_traj'], noise['eps0'], noise['idio_pop']
    )
    A_traj = mix['A_traj']

    br = [
        env_T.agent_batch_trajectory(
            new_triple[j], j, A_traj, noise['P_traj'], noise['idio_eval'][j]
        )
        for j in range(N_TYPES)
    ]

    to_np = lambda x: np.asarray(jax.device_get(x))

    return {
        'A_traj': to_np(A_traj),
        'P_traj': to_np(noise['P_traj']),
        'Afloor': float(env_T.Afloor),
        'T':      int(env_T.T),
        'mix': {
            'state':       [to_np(mix['state'][j])       for j in range(N_TYPES)],
            'action':      [to_np(mix['action'][j])      for j in range(N_TYPES)],
            'action_lo':   [to_np(mix['action_lo'][j])   for j in range(N_TYPES)],
            'action_hi':   [to_np(mix['action_hi'][j])   for j in range(N_TYPES)],
            'cum_r':       [to_np(mix['cum_r'][j])       for j in range(N_TYPES)],
            'cum_r_lo':    [to_np(mix['cum_r_lo'][j])    for j in range(N_TYPES)],
            'cum_r_hi':    [to_np(mix['cum_r_hi'][j])    for j in range(N_TYPES)],
            'emission':    [to_np(mix['emission'][j])    for j in range(N_TYPES)],
            'emission_lo': [to_np(mix['emission_lo'][j]) for j in range(N_TYPES)],
            'emission_hi': [to_np(mix['emission_hi'][j]) for j in range(N_TYPES)],
            'term_r':      [float(mix['term_r'][j])      for j in range(N_TYPES)],
        },
        'br': {
            'state':    [to_np(br[j]['state'])    for j in range(N_TYPES)],
            'action':   [to_np(br[j]['action'])   for j in range(N_TYPES)],
            'cum_r':    [to_np(br[j]['cum_r'])    for j in range(N_TYPES)],
            'emission': [to_np(br[j]['emission']) for j in range(N_TYPES)],
            'term_r':   [float(br[j]['term_r'])   for j in range(N_TYPES)],
        },
    }
