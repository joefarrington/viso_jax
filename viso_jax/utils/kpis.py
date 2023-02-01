from viso_jax.registration import make
from typing import Callable


def get_kpi_function(env_id: str, **env_kwargs) -> Callable:
    """Return the KPI function for the given environment ID."""
    env, _ = make(env_id, **env_kwargs)
    return env.calculate_kpis
