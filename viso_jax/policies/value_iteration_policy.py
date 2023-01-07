import pandas as pd
from typing import Optional
import jax.numpy as jnp
import viso_jax
from viso_jax.registration import registered_envs


class VIPolicy(object):
    def __init__(
        self,
        env_id: str,
        env_kwargs: Optional[dict] = {},
        env_params: Optional[dict] = {},
        policy_params_filepath: Optional[str] = None,
    ):
        if env_id not in registered_envs:
            raise ValueError("Environment ID is not registered.")
        if policy_params_filepath:
            env, all_env_params = viso_jax.make(env_id, **env_kwargs)
            all_env_params = all_env_params.replace(**env_params)
            self.obs_space_shape = jnp.hstack(
                [
                    env.observation_space(all_env_params).high
                    - env.observation_space(all_env_params).low
                    + 1,
                    -1,  # Shape of action, will be squeezed out if 1
                ]
            )
            self.policy_params = self.load_policy_params(policy_params_filepath)

    @classmethod
    def forward(cls, policy_params, obs, rng):
        order = policy_params[tuple(obs)]
        return order

    def load_policy_params(self, filepath):
        params_df = pd.read_csv(filepath, index_col=0)
        policy_params = jnp.array(
            params_df.values.reshape(self.obs_space_shape)
        ).squeeze()
        return policy_params
