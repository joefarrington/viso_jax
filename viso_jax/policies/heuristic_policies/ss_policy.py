import jax.numpy as jnp
import numpy as np
from typing import Optional
from functools import partial
import chex
import pandas as pd
from viso_jax.utils.yaml import from_yaml
from viso_jax.policies.heuristic_policy import HeuristicPolicy


class sSPolicy(HeuristicPolicy):
    def _get_param_col_names(self, env_id: str, env_kwargs: dict) -> list[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["s", "S"]

    def _get_param_row_names(self, env_id: str, env_kwargs: dict) -> list[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if env_id == "HendrixPerishableSubstitutionTwoProduct":
            return ["a", "b"]
        elif env_id == "MirjaliliPerishablePlatelet":
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        else:
            return []


def base_sS_policy(s, S, total_stock, policy_params):
    """Basic (s, S) policy for all environments"""
    # s should be less than S
    # Enforce that constraint here, order only made when constraint met
    constaint_met = jnp.all(policy_params[:, 0] < policy_params[:, 1])
    return jnp.where((total_stock <= s) & (constaint_met), S - total_stock, 0)


# Different environments have different observation spaces so we need
# one of each if the policy depends on an calculated feature, e.g total stock
# for (s,S)
def de_moor_perishable_sS_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """(s,S) policy for DeMoorPerishable environment"""
    # policy_params = [[s,S]]
    s = policy_params[0][0]
    S = policy_params[1][1]
    total_stock = obs.sum()
    order = base_sS_policy(s, S, total_stock, policy_params)
    return jnp.array(order)


def hendrix_perishable_one_product_sS_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """(s,S) policy for HendrixPerishableOneProduct environment"""
    # policy_params = [[s,S]]
    s = policy_params[0][0]
    S = policy_params[1][1]
    total_stock = obs.sum()
    order = base_sS_policy(s, S, total_stock, policy_params)
    return jnp.array(order)


# Calculating the total stock for each product depends on the max_useful_life
# Use partial() in the policy_constructor to set it
def hendrix_perishable_substitution_two_product_sS_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey, max_useful_life: int
) -> chex.Array:
    """(s,S) policy for HendrixPerishableSubstitutionTwoProduct environment"""
    # policy_params = [[s_a, S_a], [s_b, S_b]]
    s_a = policy_params[0, 0]
    S_a = policy_params[0, 1]
    s_b = policy_params[1, 0]
    S_b = policy_params[1, 1]

    total_stock_a = jnp.sum(obs[0:max_useful_life])
    total_stock_b = jnp.sum(obs[max_useful_life : 2 * max_useful_life])

    order_a = base_sS_policy(s_a, S_a, total_stock_a, policy_params)
    order_b = base_sS_policy(s_b, S_b, total_stock_b, policy_params)
    return jnp.array([order_a, order_b])


def mirjalili_perishable_platelet_sS_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """(s,S) policy for MirjaliliPerishablePlatelet environment"""
    # policy_params = [[s_Mon, S_Mon], ..., [s_Sun, S_Sun]]
    weekday = obs[0]
    s = policy_params[weekday][0]
    S = policy_params[weekday][1]
    total_stock = jnp.sum(obs[1:])  # First element of obs is weekday
    order = base_sS_policy(s, S, total_stock, policy_params)
    return jnp.array(order)
