# This is based on gymnax.registration by Robert T. Lange
# https://github.com/RobertTLange/gymnax/blob/main/gymnax/registration.py
# Modified from commit b9f4795

from gymnax.environments.environment import Environment, EnvParams
from viso_jax.environments import (
    DeMoorPerishableGymnax,
    HendrixPerishableOneProductGymnax,
    HendrixPerishableSubstitutionTwoProductGymnax,
    MirjaliliPerishablePlateletGymnax,
    MirjaliliPerishablePlateletDeterministicUsefulLifeGymnax,
)


def make(env_id: str, **env_kwargs) -> tuple[Environment, EnvParams]:
    """Version of gymnax.make/OpenAI gym.make for our envs"""

    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered gymnax environments.")
    if env_id == "DeMoorPerishable":
        env = DeMoorPerishableGymnax(**env_kwargs)
    elif env_id == "HendrixPerishableOneProduct":
        env = HendrixPerishableOneProductGymnax(**env_kwargs)
    elif env_id == "HendrixPerishableSubstitutionTwoProduct":
        env = HendrixPerishableSubstitutionTwoProductGymnax(**env_kwargs)
    elif env_id == "MirjaliliPerishablePlatelet":
        env = MirjaliliPerishablePlateletGymnax(**env_kwargs)
    elif env_id == "MirjaliliPerishablePlateletDeterministicUsefulLife":
        env = MirjaliliPerishablePlateletDeterministicUsefulLifeGymnax(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered.")

    return env, env.default_params


registered_envs = [
    "DeMoorPerishable",
    "HendrixPerishableOneProduct",
    "HendrixPerishableSubstitutionTwoProduct",
    "MirjaliliPerishablePlatelet",
    "MirjaliliPerishablePlateletDeterministicUsefulLife",
]
