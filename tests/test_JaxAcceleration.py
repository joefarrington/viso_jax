import pytest
import jax


@pytest.mark.jax()
def test_jax_acceleration():
    assert jax.default_backend() != "cpu"
