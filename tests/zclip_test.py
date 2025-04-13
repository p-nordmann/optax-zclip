import jax.numpy as jnp
import optax.tree_utils as otu
import pytest

from optax_zclip import zclip

STEPS = 50
WARMUP_STEPS = 3


@pytest.fixture
def params():
    return (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))


@pytest.fixture
def warmup_updates():
    return [
        (jnp.array([10.0, -4.0]), jnp.array([2.0, 6.0])),
        (jnp.array([10.0, 10.0]), jnp.array([10.0, 10.0])),
        (jnp.array([-7.0, -1.0]), jnp.array([24.0, 1.0])),
    ]


@pytest.fixture
def per_step_updates():
    return (jnp.array([500.0, 5.0]), jnp.array([300.0, 3.0]))


def test_zclip_init(params):
    opt = zclip(warmup_steps=WARMUP_STEPS)
    state = opt.init(params)

    assert len(state) == 4
    assert state[0].shape == ()
    assert state[1].shape == ()
    assert state[2].shape == ()
    assert state[0] == 0.0
    assert state[1] == 0.0
    assert state[2] == 0.0
    assert state[3] == 0


def test_zclip_step_count(params, warmup_updates, per_step_updates):
    opt = zclip(warmup_steps=WARMUP_STEPS)
    state = opt.init(params)

    for i, updates in enumerate(warmup_updates + [per_step_updates] * 10):
        updates, state = opt.update(updates, state, params)

        assert state[3] == i + 1


def test_zclip_warmup(params, warmup_updates):
    opt = zclip(warmup_steps=WARMUP_STEPS)
    state = opt.init(params)

    for updates in warmup_updates:
        updates, state = opt.update(updates, state, params)

    (mu_t, m2_t, var_t, step_count) = state
    grad_norms = jnp.array([otu.tree_l2_norm(updates) for updates in warmup_updates])

    assert jnp.isclose(step_count, len(warmup_updates))
    assert jnp.isclose(mu_t, jnp.mean(grad_norms))
    assert jnp.isclose(m2_t, jnp.mean(grad_norms**2))
    assert jnp.isclose(var_t, jnp.var(grad_norms))


def test_zclip_general(params, warmup_updates, per_step_updates):
    opt = zclip(warmup_steps=WARMUP_STEPS)
    state = opt.init(params)

    for updates in warmup_updates:
        updates, state = opt.update(updates, state, params)

    for i in range(STEPS):
        updates, state = opt.update(per_step_updates, state, params)

    # TODO perform checks
    assert False
