"""Networks for the major–minor market (MAJOR_MINOR_CLEARING.md).

One class for every player: input = own normalised state (8) + public history
H_t (3), conditioned on the normalised scenario gamma (4) through FiLM. The
head is the demand schedule (alpha, beta) + production (xi_hat, eta_hat):

    minors : beta = BETA_MIN_MINOR + softplus(.) > 0
    major  : beta = softplus(.) >= 0          (a flat schedule = a quantity order)

The candidate price is never an input, so every schedule is nonincreasing in it.
Rung 2 uses the same class with gamma fixed at the reference scenario.
"""
from typing import List
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from envs.models import FiLMLayer, STATE_DIM_DYNAMIC

H_DIM = 3                          # (EMA price change, last sellers' fill ratio, EMA volume)
IN_DIM = STATE_DIM_DYNAMIC + H_DIM
CTX_DIM = 4
BETA_MIN_MINOR = 0.1
BETA_MIN_MAJOR = 0.0


class SchedulePolicy(eqx.Module):
    trunk: list
    film: list
    beta_min: float = eqx.field(static=True)
    n_hidden: int = eqx.field(static=True)

    def __init__(self, beta_min=BETA_MIN_MINOR, architecture: List[int] = [64, 64, 64],
                 film_hidden: int = 32, key=None):
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))
        self.beta_min = float(beta_min)
        self.n_hidden = len(architecture)
        dims = [IN_DIM] + list(architecture) + [4]
        ks = jax.random.split(key, len(dims) - 1 + self.n_hidden)
        trunk = [eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i]) for i in range(len(dims) - 1)]
        # beta ~ 1 at initialisation
        b = trunk[-1].bias.at[1].set(float(np.log(np.expm1(1.0 - self.beta_min))))
        trunk[-1] = eqx.tree_at(lambda l: l.bias, trunk[-1], b)
        self.trunk = trunk
        self.film = [FiLMLayer(CTX_DIM, architecture[i], film_hidden, key=ks[len(dims) - 1 + i])
                     for i in range(self.n_hidden)]

    def __call__(self, x, ctx):
        h = x
        for i in range(self.n_hidden):
            g, b = self.film[i](ctx)
            h = jax.nn.silu(g * self.trunk[i](h) + b)
        out = self.trunk[self.n_hidden](h)
        return jnp.stack([out[0], self.beta_min + jax.nn.softplus(out[1]),
                          jax.nn.sigmoid(out[2]), jax.nn.sigmoid(out[3])])


def minor_policy(key):
    return SchedulePolicy(beta_min=BETA_MIN_MINOR, key=key)


def major_policy(key):
    return SchedulePolicy(beta_min=BETA_MIN_MAJOR, key=key)


def make_buffer(policy, k_max):
    params, static = eqx.partition(policy, eqx.is_array)
    return jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p, (k_max,) + p.shape), params), static


def buffer_set(buf, i, policy):
    params, _ = eqx.partition(policy, eqx.is_array)
    return jax.tree_util.tree_map(lambda b, p: b.at[i].set(p), buf, params)


def gather_apply(buf, static, comp, X, ctx):
    """Agent i uses component comp[i] of the buffer. X (n, IN_DIM), ctx (4,) -> (n, 4)."""
    params_i = jax.tree_util.tree_map(lambda p: p[comp], buf)
    return jax.vmap(lambda p, x: eqx.combine(p, static)(x, ctx))(params_i, X)


def component(buf, static, u):
    """The single network of component u (traced index) of a buffer."""
    return eqx.combine(jax.tree_util.tree_map(lambda p: p[u], buf), static)
