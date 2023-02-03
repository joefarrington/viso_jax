import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict, Any, List
import chex
import pandas as pd
import viso_jax
from viso_jax.utils.yaml import from_yaml
from gymnax.environments.environment import Environment


class HeuristicPolicy:
    def __init__(
        self,
        env_id: str,
        env_kwargs: Optional[Dict[str, Any]] = {},
        env_params: Optional[Dict[str, Any]] = {},
        policy_params_filepath: Optional[str] = None,
    ):

        # As in utils/rollout.py env_kwargs and env_params arguments are dicts to
        # override the defaults for an environment.

        # Instantiate an internal envinronment we'll use to access env kwargs/params
        # These are not stored, just used to set up param_col_names, param_row_names and forward
        self.env_id = env_id
        env, default_env_params = viso_jax.make(self.env_id, **env_kwargs)
        all_env_params = default_env_params.create_env_params(**env_params)

        self.param_col_names = self._get_param_col_names(env_id, env, all_env_params)
        self.param_row_names = self._get_param_row_names(env_id, env, all_env_params)
        self.forward = self._get_forward_method(env_id, env, all_env_params)

        if self.param_row_names != []:
            self.param_names = np.array(
                [
                    [f"{p}_{r}" for p in self.param_col_names]
                    for r in self.param_row_names
                ]
            )
        else:
            self.param_names = np.array([self.param_col_names])

        self.params_shape = self.param_names.shape

        if policy_params_filepath:
            self.policy_params = self.load_policy_params(policy_params_filepath)

    def _get_param_col_names(
        self, env_id: str, env: Environment, env_params: Dict[str, Any]
    ) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        raise NotImplementedError

    def _get_param_row_names(
        self, env_id: str, env: Environment, env_params: Dict[str, Any]
    ) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        raise NotImplementedError

    def _get_forward_method(
        self, env_id: str, env: Environment, env_params: Dict[str, Any]
    ) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        raise NotImplementedError

    def load_policy_params(self, filepath: str) -> chex.Array:
        """Load policy parameters from a yaml file"""
        params_dict = from_yaml(filepath)["policy_params"]
        if self.param_row_names is None:
            params_df = pd.DataFrame(params_dict, index=[0])
        else:
            params_df = pd.DataFrame(params_dict)
        policy_params = jnp.array(params_df.values)
        assert (
            policy_params.shape == self.params_shape
        ), f"Parameters in file do not match expected shape: found {policy_params.shape} and expected {self.params_shape}"
        return policy_params
