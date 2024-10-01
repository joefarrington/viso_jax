import jax.numpy as jnp
import numpy as np
from typing import Optional, List
from functools import partial
import chex
import pandas as pd
from viso_jax.utils.yaml import from_yaml
from viso_jax.policies.heuristic_policy import HeuristicPolicy
from gymnax.environments.environment import Environment, EnvParams


class sQPolicy(HeuristicPolicy):
    def _get_param_col_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["s", "Q"]

    def _get_param_row_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if env_id == "RajendranPerishablePlatelet":
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        else:
            return []

    def _get_forward_method(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> callable:
        if env_id == "RajendranPerishablePlatelet":
            return rajendran_perishable_platelet_sQ_policy
        else:
            raise ValueError(f"No (s,Q) policy defined for Environment ID {env_id}")


def rajendran_perishable_platelet_sQ_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """(s,Q) policy for RajendranPerishablePlatelet environment"""
    # NOTE: As with other policies for this scenario only, only place an
    # order when stock is less than s (instead of stock <= s which we
    # used for other scenarios)

    # policy_params = [[s_Mon, Q_Mon], ..., [s_Sun, Q_Sun]]
    weekday = obs[0]
    s = policy_params[weekday][0]
    Q = policy_params[weekday][1]
    total_stock = jnp.sum(obs[1:])  # First element of ob
    s is weekday
    order = jnp.where((total_stock < s), Q, 0)
    return order