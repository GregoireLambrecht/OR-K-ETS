import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import List


# ═════════════════════════════════════════════════════════════════════════════
# Rung 1: one plain MLP per agent type.
#
# The type descriptors (ef, eg, cf, cg, cap_f, cap_xi, sigma_eps, tech) are
# constant within a network, so they carry no information and are dropped from
# the input: 8 dynamic dims instead of 15.  They stay in the 16-vector state,
# because unnormalize_action still needs cap_f / cap_xi / tech to build the
# admissible set A(j, x).
#
# No FiLM here: the scenario gamma is frozen in Rung 1, so there is nothing to
# condition on.  ContextActionNN below is kept for Rung 3.
# ═════════════════════════════════════════════════════════════════════════════

STATE_DIM_DYNAMIC = 8   # t, theta, xi_f, xi_g, eps, A, P, A_avg


class TypePolicyNN(eqx.Module):
    """Deterministic policy for a single agent type. Returns (f, xi, eta) in
    normalised units: f in [-1,1], xi in [0,1], eta in [0,1]."""

    layers: list
    in_dim:     int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim:       int       = STATE_DIM_DYNAMIC,
        architecture: List[int] = [64, 64, 64],
        action_dim:   int       = 3,
        key = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))

        self.in_dim     = in_dim
        self.action_dim = action_dim

        dims = [in_dim] + list(architecture) + [action_dim]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        out = self.layers[-1](h)
        return jnp.stack([
            jnp.tanh(out[0]),          # f   in [-1, 1]
            jax.nn.sigmoid(out[1]),    # xi  in [0, 1]
            jax.nn.sigmoid(out[2]),    # eta in [0, 1] (may be overridden by tech)
        ])


# ── policy stacking ──────────────────────────────────────────────────────────
# The ensemble rollout applies K policies of the same type, one per FP block.
# Doing that with a Python `for k in range(K)` unrolls the traced graph K times
# per scan step (the old rollout_market did exactly this, which is why compile
# time blew up at K=200).  Instead we stack the K parameter pytrees along a new
# leading axis and vmap over it: one traced policy application, K-way batched.

def stack_policies(policies):
    """[Model] * K  ->  (stacked_params with leading dim K, shared static)."""
    parts  = [eqx.partition(p, eqx.is_array) for p in policies]
    params = [p for p, _ in parts]
    static = parts[0][1]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *params)
    return stacked, static


def apply_stacked(stacked, static, X):
    """stacked: params with leading dim K.  X: (K, n, in_dim) -> (K, n, 3)."""
    def one(params, xs):
        model = eqx.combine(params, static)
        return jax.vmap(model)(xs)
    return jax.vmap(one)(stacked, X)


# ═════════════════════════════════════════════════════════════════════════════
# Rung 3 (BE-MFG): FiLM-conditioned policy.  Unused in Rung 1, kept intact.
# ═════════════════════════════════════════════════════════════════════════════

class FiLMLayer(eqx.Module):
    fc1: eqx.nn.Linear
    fc2_gamma: eqx.nn.Linear
    fc2_beta: eqx.nn.Linear

    def __init__(self, context_dim: int, hidden_dim: int, film_hidden: int, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.fc1       = eqx.nn.Linear(context_dim, film_hidden, key=k1)
        self.fc2_gamma = eqx.nn.Linear(film_hidden, hidden_dim,  key=k2)
        self.fc2_beta  = eqx.nn.Linear(film_hidden, hidden_dim,  key=k3)

    def __call__(self, context):
        h     = jax.nn.silu(self.fc1(context))
        gamma = 1.0 + self.fc2_gamma(h)
        beta  = self.fc2_beta(h)
        return gamma, beta


class ContextActionNN(eqx.Module):
    trunk_layers: list
    film_layers:  list

    state_dim:   int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    action_dim:  int = eqx.field(static=True)
    n_hidden:    int = eqx.field(static=True)

    def __init__(
        self,
        state_dim:    int       = STATE_DIM_DYNAMIC,
        context_dim:  int       = 4,
        action_dim:   int       = 3,
        architecture: List[int] = [64, 64, 64],
        film_hidden:  int       = 32,
        key = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))

        self.state_dim   = state_dim
        self.context_dim = context_dim
        self.action_dim  = action_dim
        self.n_hidden    = len(architecture)

        dims     = [state_dim] + list(architecture) + [action_dim]
        n_layers = len(dims) - 1

        keys       = jax.random.split(key, n_layers + self.n_hidden)
        trunk_keys = keys[:n_layers]
        film_keys  = keys[n_layers:]

        self.trunk_layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=trunk_keys[i])
            for i in range(n_layers)
        ]
        self.film_layers = [
            FiLMLayer(context_dim, architecture[i], film_hidden, key=film_keys[i])
            for i in range(self.n_hidden)
        ]

    def __call__(self, x, context):
        h = x
        for i in range(self.n_hidden):
            h = self.trunk_layers[i](h)
            gamma, beta = self.film_layers[i](context)
            h = jax.nn.silu(gamma * h + beta)

        out = self.trunk_layers[self.n_hidden](h)
        return jnp.stack([
            jnp.tanh(out[0]),
            jax.nn.sigmoid(out[1]),
            jax.nn.sigmoid(out[2]),
        ])


class BoundContextNN(eqx.Module):
    """A ContextActionNN with its scenario attached, exposing `(8,) -> (3,)`.

    This is the whole trick that lets Rung 3 reuse `envs/environment.py`
    untouched: every rollout there calls `policy(x)`, so the scenario has to
    travel with the network rather than through a second argument. `ctx` is an
    ordinary array leaf, so it can be vmapped over — one market per scenario —
    and rebound with `eqx.tree_at` at zero cost.

    `ctx` holds the NORMALISED scenario, in [-1, 1]^2. Normalisation lives in
    the caller (utils_rung3.norm_gamma) so this module needs no parameter
    ranges; raw kappa spans [-0.5, 7.2] against init_alloc's [0, 1.2], and
    unnormalised the FiLM conditioning would be all kappa.

    The bound value is always overwritten before use, so the gradient w.r.t.
    `ctx` is identically zero and Adam leaves it alone.
    """

    net: ContextActionNN
    ctx: jnp.ndarray

    def __init__(self, net=None, ctx=None, key=None, **kwargs):
        self.net = ContextActionNN(key=key, **kwargs) if net is None else net
        self.ctx = (jnp.zeros(self.net.context_dim, dtype=jnp.float32)
                    if ctx is None else ctx)

    def __call__(self, x):
        return self.net(x, self.ctx)
