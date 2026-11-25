"""Networks for market clearing v2 (rung 2: unconditional per-type policy)."""
from typing import List
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from envs.models import STATE_DIM_DYNAMIC
from envs.params_v2 import BETA_MIN_V2


class TypePolicyNNV2(eqx.Module):
    """(alpha, beta, xi_hat, eta_hat) on the PRE-TRADE state, beta > BETA_MIN_V2.
    Same body as envs.models.TypePolicyNN; only the slope floor differs
    (0.1 instead of 0.5). beta ~ 1 at initialisation."""

    layers: list
    in_dim:     int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int = STATE_DIM_DYNAMIC,
                 architecture: List[int] = [64, 64, 64], action_dim: int = 4, key=None):
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))
        self.in_dim, self.action_dim = in_dim, action_dim
        dims = [in_dim] + list(architecture) + [action_dim]
        keys = jax.random.split(key, len(dims) - 1)
        layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
                  for i in range(len(dims) - 1)]
        b = layers[-1].bias.at[1].set(float(np.log(np.expm1(1.0 - BETA_MIN_V2))))
        layers[-1] = eqx.tree_at(lambda l: l.bias, layers[-1], b)
        self.layers = layers

    def __call__(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        out = self.layers[-1](h)
        return jnp.stack([out[0],
                          BETA_MIN_V2 + jax.nn.softplus(out[1]),
                          jax.nn.sigmoid(out[2]),
                          jax.nn.sigmoid(out[3])])


def apply_gathered(stacked, static, comp, X):
    """stacked: params with leading dim K_MAX. comp: (n,) component indices.
    X: (n, in_dim) -> (n, 4). Agent i uses the parameters of component comp[i]."""
    params_i = jax.tree_util.tree_map(lambda p: p[comp], stacked)
    return jax.vmap(lambda p, x: eqx.combine(p, static)(x))(params_i, X)


def make_buffer(policy, k_max):
    """Fixed mixture buffer: k_max copies of `policy`'s parameters (unused slots
    are never drawn). Returns (buffer params, static)."""
    params, static = eqx.partition(policy, eqx.is_array)
    buf = jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p, (k_max,) + p.shape), params)
    return buf, static


def buffer_set(buf, i, policy):
    """Write `policy`'s parameters into slot i of the buffer."""
    params, _ = eqx.partition(policy, eqx.is_array)
    return jax.tree_util.tree_map(lambda b, p: b.at[i].set(p), buf, params)
