from typing import Optional

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu


def zclip(
    alpha: float = 0.97,
    z_threshold: float = 2.5,
    eps: float = 1e-6,
    warmup_steps: int = 25,
    stats_dtype: Optional[jax.typing.DTypeLike] = None,
):
    """"""

    def init_fn(params):
        mu_0 = jnp.zeros((), dtype=stats_dtype)
        m2_0 = jnp.zeros((), dtype=stats_dtype)
        var_0 = jnp.zeros((), dtype=stats_dtype)
        step_count_0 = jnp.zeros((), jnp.uint32)
        return (mu_0, m2_0, var_0, step_count_0)

    def update_fn(updates, state, params=None):
        del params
        _, _, _, step_count = state

        # We need XLA to handle branching for warmup as warmup_steps can be
        # arbitrarily large.
        return jax.lax.cond(
            step_count < warmup_steps,
            _update_fn_warmup,
            _update_fn_general,
            updates,
            state,
        )

    def _update_fn_warmup(updates, state):
        # During warmup, we don't touch the gradients, only update statistics.
        # Also we do not use moving averages but actual empirical estimators.

        grad_norm = otu.tree_l2_norm(updates)
        mu_t, m2_t, _, step_count = state

        new_mu_t = jax.lax.select(
            step_count == 0,
            grad_norm,
            mu_t + (grad_norm - mu_t) / (step_count + 1),
        )
        new_m2_t = jax.lax.select(
            step_count == 0,
            grad_norm**2,
            m2_t + (grad_norm**2 - m2_t) / (step_count + 1),
        )
        new_var_t = new_m2_t - new_mu_t**2

        return updates, (
            new_mu_t,
            new_m2_t,
            new_var_t,
            step_count + 1,
        )

    def _update_fn_general(updates, state):
        grad_norm = otu.tree_l2_norm(updates)
        mu_t, m2_t, var_t, step_count = state
        std_t = jnp.sqrt(var_t)

        # If the *un-normalized*  gradient norm is below threshold, we keep it.
        # Otherwise we use clipping on the norm itself.
        z_score = (grad_norm - mu_t) / (std_t + eps)
        clipped_grad_norm = jax.lax.select(
            z_score <= z_threshold,
            grad_norm,
            mu_t + (z_threshold**2 / z_score) * std_t,
        )
        print(z_score)

        # Note: this does not perform clipping, but rather it scales the gradient
        # such that its norm is below clipped_gradient_norm.
        # There seems to be a confusion in the paper and in the implementation from
        # the authors. We follow their implementation in that we rescale the gradients
        # to make sure that their overall norm is not greater than clipped_grad_norm.
        new_updates = otu.tree_scalar_mul(clipped_grad_norm / grad_norm, updates)

        new_mu_t = alpha * mu_t + (1 - alpha) * clipped_grad_norm
        new_m2_t = alpha * m2_t + (1 - alpha) * clipped_grad_norm**2
        new_var_t = alpha * var_t + (1 - alpha) * (clipped_grad_norm - new_mu_t) ** 2

        return new_updates, (
            new_mu_t,
            new_m2_t,
            new_var_t,
            step_count + 1,
        )

    return optax.GradientTransformation(init_fn, update_fn)
