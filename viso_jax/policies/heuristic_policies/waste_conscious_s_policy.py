import jax.numpy as jnp
from functools import partial
from typing import List
import chex
from viso_jax.utils.yaml import from_yaml
from viso_jax.policies.heuristic_policy import HeuristicPolicy
from gymnax.environments.environment import Environment, EnvParams


# NOTE: Coercing mean demand to an integer
# Mean demand is an integer in all experiments
# from Hendrix et al (2019) so this is fine for our purposes

# Policy based on Equation 10 in Hendrix et al (2019)


class WasteConsciousSPolicy(HeuristicPolicy):
    def _get_param_col_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["S"]

    def _get_param_row_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if env_id == "HendrixPerishableSubstitutionTwoProduct":
            return ["a", "b"]
        else:
            return []

    def _get_forward_method(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        if env_id == "DeMoorPerishable":
            return partial(
                de_moor_perishable_waste_conscious_S_policy,
                mean_demand=jnp.array(env_params.demand_gamma_mean, dtype=jnp.int32),
            )
        elif env_id == "HendrixPerishableOneProduct":
            return partial(
                hendrix_perishable_one_product_waste_conscious_S_policy,
                mean_demand=jnp.array(env_params.demand_poisson_mean, dtype=jnp.int32),
            )
        elif env_id == "HendrixPerishableSubstitutionTwoProduct":
            return partial(
                hendrix_perishable_substitution_two_product_waste_conscious_S_policy,
                max_useful_life=env.max_useful_life,
                mean_demand_a=jnp.array(
                    env_params.demand_poisson_mean_a, dtype=jnp.int32
                ),
                mean_demand_b=jnp.array(
                    env_params.demand_poisson_mean_b, dtype=jnp.int32
                ),
            )
        else:
            raise NotImplementedError(
                f"No waste-conscious (S) policy defined for Environment ID {env_id}"
            )


def base_waste_conscious_S_policy(
    S: int, total_stock: int, stock_expiring_next_period: int, mean_demand: int
) -> chex.Array:
    return jnp.where(
        (total_stock < S),
        S - total_stock + (stock_expiring_next_period - mean_demand).clip(0),
        jnp.array(0, dtype=jnp.int32),
    )


# Different environments have different observation spaces so we need
# one of each if the policy depends on an calculated feature, e.g total stock
# for (s,S)
def de_moor_perishable_waste_conscious_S_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey, mean_demand: float
) -> chex.Array:
    """Waste-conscious (S) policy for DeMoorPerishable environment"""
    # policy_params = [[S]]
    S = policy_params[0, 0]
    total_stock = obs.sum()
    stock_expiring_next_period = obs[-1]
    order = base_waste_conscious_S_policy(
        S, total_stock, stock_expiring_next_period, mean_demand
    )
    return jnp.array(order)


def hendrix_perishable_one_product_waste_conscious_S_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey, mean_demand: float
) -> chex.Array:
    """Waste-conscious (S) policy for HendrixPerishableOneProduct environment"""
    # policy_params = [[S]]
    S = policy_params[0, 0]
    total_stock = obs.sum()
    stock_expiring_next_period = obs[-1]
    order = base_waste_conscious_S_policy(
        S, total_stock, stock_expiring_next_period, mean_demand
    )
    return jnp.array(order)


# Calculating the total stock for each product depends on the max_useful_life
# Use partial() in the policy_constructor to set it
def hendrix_perishable_substitution_two_product_waste_conscious_S_policy(
    policy_params: chex.Array,
    obs: chex.Array,
    rng: chex.PRNGKey,
    max_useful_life: int,
    mean_demand_a: float,
    mean_demand_b: float,
) -> chex.Array:
    """Waste conscious (S) policy for HendrixPerishableSubstitutionTwoProduct environment"""
    # policy_params = [[S_a], [S_b]]
    S_a = policy_params[0, 0]
    S_b = policy_params[1, 0]

    total_stock_a = jnp.sum(obs[0:max_useful_life])
    total_stock_b = jnp.sum(obs[max_useful_life : 2 * max_useful_life])

    stock_expiring_next_period_a = obs[max_useful_life - 1]
    stock_expiring_next_period_b = obs[-1]

    order_a = base_waste_conscious_S_policy(
        S_a, total_stock_a, stock_expiring_next_period_a, mean_demand_a
    )
    order_b = base_waste_conscious_S_policy(
        S_b, total_stock_b, stock_expiring_next_period_b, mean_demand_b
    )
    return jnp.array([order_a, order_b])
