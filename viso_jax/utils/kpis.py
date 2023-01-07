from viso_jax.environments import (
    DeMoorPerishableGymnax,
    HendrixPerishableOneProductGymnax,
    HendrixPerishableSubstitutionTwoProductGymnax,
    MirjaliliPerishablePlateletGymnax,
)

# Each environment has a KPI functionuse this function to return it


def get_kpi_function(env_id, **env_kwargs):
    if env_id == "DeMoorPerishable":
        env = DeMoorPerishableGymnax(env_kwargs)
    elif env_id == "HendrixPerishableOneProduct":
        env = HendrixPerishableOneProductGymnax(**env_kwargs)
    elif env_id == "HendrixPerishableSubstitutionTwoProduct":
        env = HendrixPerishableSubstitutionTwoProductGymnax(**env_kwargs)
    elif env_id == "MirjaliliPerishablePlatelet":
        env = MirjaliliPerishablePlateletGymnax(**env_kwargs)

    else:
        raise ValueError("Environment ID is not registered.")
    return env.calculate_kpis
