import jax.numpy as jnp
import numpy as np
from typing import Optional, List
from functools import partial
import chex
import pandas as pd
from viso_jax.utils.yaml import from_yaml
from viso_jax.policies.heuristic_policy import HeuristicPolicy
from gymnax.environments.environment import Environment, EnvParams


class sSaQPolicy(HeuristicPolicy):
    def _get_param_col_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["s", "S", "a", "Q"]

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
            return rajendran_perishable_platelet_sSaQ_policy
        else:
            raise ValueError(f"No (s,S,a,Q) policy defined for Environment ID {env_id}")


def rajendran_perishable_platelet_sSaQ_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """(s,S,a,Q) policy for RajendranPerishablePlatelet environment"""
    # NOTE: As with other policies for this scenario only, only place an
    # order when stock is less than s (instead of stock <= s which we
    # used for other scenarios)

    # NOTE: Here, we're just applying the constraint by clipping the parameters
    # So, we can't jus tnaively report the paramters reported by Optuna

    # policy_params = [[s_Mon, S_Mon, a_Mon, Q_Mon], ..., [s_Sun, S_Sun, a_Sun, Q_Sun]]
    weekday = obs[0]
    S = policy_params[weekday][1]
    s = jnp.minimum(policy_params[weekday][0], S - 1)
    a = jnp.minimum(policy_params[weekday][2], s - 1)
    Q = policy_params[weekday][3]
    total_stock = jnp.sum(obs[1:])  # First element of ob
    policy_params[:, 2] < policy_params[:, 0]
    order = jnp.where(
        (total_stock < s),
        jnp.where(total_stock < a, S - total_stock, Q),
        0,
    )
    return order
