import pandas as pd
from typing import Optional, Dict, Any
import jax.numpy as jnp
import viso_jax
from viso_jax.registration import registered_envs


class VIPolicy(object):
    def __init__(
        self,
        env_id: str,
        env_kwargs: Optional[Dict[str, Any]] = {},
        env_params: Optional[Dict[str, Any]] = {},
        policy_params_filepath: Optional[str] = None,
        policy_params_df: Optional[pd.DataFrame] = None,
    ):
        if env_id not in registered_envs:
            raise ValueError("Environment ID is not registered.")
        if (policy_params_filepath is not None) and (policy_params_df is not None):
            raise ValueError(
                "Supply policy parameters using only one of policy_params_filepath or policy_params_df"
            )
        elif (policy_params_filepath is not None) or (policy_params_df is not None):
            env, default_env_params = viso_jax.make(env_id, **env_kwargs)
            env_params = default_env_params.create_env_params(**env_params)
            self.obs_space_shape = jnp.hstack(
                [
                    env.observation_space(env_params).high
                    - env.observation_space(env_params).low
                    + 1,
                    -1,  # Shape of action, will be squeezed out if 1
                ]
            )
            if policy_params_filepath:
                policy_params_df = self._load_policy_params(policy_params_filepath)
            self.policy_params = self._policy_params_df_to_array(policy_params_df)

    @classmethod
    def forward(cls, policy_params, obs, rng):
        order = policy_params[tuple(obs)]
        return order

    def _load_policy_params(self, filepath):
        policy_params_df = pd.read_csv(filepath, index_col=0)

        return policy_params_df

    def _policy_params_df_to_array(self, policy_params_df):
        policy_params = jnp.array(
            policy_params_df.values.reshape(self.obs_space_shape)
        ).squeeze()
        return policy_params
