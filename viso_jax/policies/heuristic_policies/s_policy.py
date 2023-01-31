import jax.numpy as jnp
from functools import partial
import chex
from viso_jax.utils.yaml import from_yaml
from viso_jax.policies.heuristic_policy import HeuristicPolicy
from gymnax.environments.environment import Environment, EnvParams


class SPolicy(HeuristicPolicy):
    def _get_param_col_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> list[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["S"]

    def _get_param_row_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> list[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if env_id == "HendrixPerishableSubstitutionTwoProduct":
            return ["a", "b"]
        elif env_id == "MirjaliliPerishablePlatelet":
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        else:
            return []

    def _get_forward_method(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        if env_id == "DeMoorPerishable":
            return de_moor_perishable_S_policy
        elif env_id == "HendrixPerishableOneProduct":
            return hendrix_perishable_one_product_S_policy
        elif env_id == "HendrixPerishableSubstitutionTwoProduct":
            return partial(
                hendrix_perishable_substitution_two_product_S_policy,
                max_useful_life=env.max_useful_life,
            )
        elif env_id == "MirjaliliPerishablePlatelet":
            return mirjalili_perishable_platelet_S_policy
        else:
            raise NotImplementedError(
                f"No (S) policy defined for Environment ID {env_id}"
            )


def base_S_policy(S: int, total_stock: int, policy_params: chex.Array) -> chex.Array:
    """Basic (S) policy for all environments"""
    return jnp.where((total_stock < S), S - total_stock, 0)


# Different environments have different observation spaces so we need
# one of each if the policy depends on an calculated feature, e.g total stock
# for (S)
def de_moor_perishable_S_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """(S) policy for DeMoorPerishable environment"""
    # policy_params = [[S]]
    S = policy_params[0, 0]
    total_stock = obs.sum()
    order = base_S_policy(S, total_stock, policy_params)
    return jnp.array(order)


def hendrix_perishable_one_product_S_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """(S) policy for HendrixPerishableOneProduct environment"""
    # policy_params = [[S]]
    S = policy_params[0, 0]
    total_stock = obs.sum()
    order = base_S_policy(S, total_stock, policy_params)
    return jnp.array(order)


# Calculating the total stock for each product depends on the max_useful_life
# Use partial() in the policy_constructor to set it
def hendrix_perishable_substitution_two_product_S_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey, max_useful_life: int
) -> chex.Array:
    """(S) policy for HendrixPerishableSubstitutionTwoProduct environment"""
    # policy_params = [[S_a], [S_b]]
    S_a = policy_params[0, 0]
    S_b = policy_params[1, 0]

    total_stock_a = jnp.sum(obs[0:max_useful_life])
    total_stock_b = jnp.sum(obs[max_useful_life : 2 * max_useful_life])

    order_a = base_S_policy(S_a, total_stock_a, policy_params)
    order_b = base_S_policy(S_b, total_stock_b, policy_params)
    return jnp.array([order_a, order_b])


def mirjalili_perishable_platelet_S_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """(S) policy for MirjaliliPerishablePlatelet environment"""
    # policy_params = [[S_Mon], ..., [S_Sun]]
    weekday = obs[0]
    S = policy_params[weekday][0]
    total_stock = jnp.sum(obs[1:])  # First element of obs is weekday
    order = base_S_policy(S, total_stock, policy_params)
    return jnp.array(order)
