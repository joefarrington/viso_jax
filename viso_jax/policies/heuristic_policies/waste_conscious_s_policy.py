import jax.numpy as jnp
import numpy as np
from typing import Optional
from functools import partial
import pandas as pd

# TODO: Currently coercing demand to an integer, work out whether better way to round


class WasteConsciousSPolicy:
    def __init__(
        self,
        env_id: str,
        env_kwargs: Optional[dict] = {},
        env_params: Optional[dict] = {},
        policy_params_filepath: Optional[str] = None,
    ):

        self.param_col_names = np.array(["S"])
        self.param_row_names = None
        if env_id == "DeMoorPerishable":
            self.forward = partial(
                de_moor_perishable_waste_conscious_S_policy,
                mean_demand=jnp.array(env_params["demand_gamma_mean"], dtype=jnp.int32),
            )
        elif env_id == "HendrixPerishableOneProduct":
            self.forward = partial(
                hendrix_perishable_one_product_waste_conscious_S_policy,
                mean_demand=jnp.array(
                    env_params["demand_poisson_mean"], dtype=jnp.int32
                ),
            )
        elif env_id == "HendrixPerishableSubstitutionTwoProduct":
            self.forward = partial(
                hendrix_perishable_substitution_two_product_waste_conscious_S_policy,
                max_useful_life=env_kwargs["max_useful_life"],
                mean_demand_a=jnp.array(
                    env_params["demand_poisson_mean_a"], dtype=jnp.int32
                ),
                mean_demand_b=jnp.array(
                    env_params["demand_poisson_mean_b"], dtype=jnp.int32
                ),
            )
            self.param_row_names = ["a", "b"]
        elif env_id == "MirjaliliPerishablePlatelet":
            self.forward = mirjalili_perishable_platelet_waste_conscious_S_policy
            self.param_row_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        else:
            raise ValueError(f"No (S) policy defined for Environment ID {env_id}")
        if self.param_row_names is not None:
            self.param_names = np.array(
                [
                    [f"{p}_{r}" for p in self.param_col_names]
                    for r in self.param_row_names
                ]
            )
        else:
            # If no specified row names, use 0 as row name/index
            self.param_row_names = [0]
            self.param_names = np.array([self.param_col_names])

        self.params_shape = self.param_names.shape

        if policy_params_filepath:
            self.policy_params = self.load_policy_params(policy_params_filepath)

    def load_policy_params(self, filepath):
        params_df = pd.read_csv(filepath, index_col=0)
        params = jnp.array(params_df.value)
        assert (
            params.shape == self.params_shape
        ), f"Paramters in file do not match expected shape: found {params.shape} and expected {self.params_shape}"
        return


def base_waste_conscious_S_policy(
    S, total_stock, stock_expiring_next_period, mean_demand
):
    return jnp.where(
        (total_stock < S),
        S - total_stock + (stock_expiring_next_period - mean_demand).clip(0),
        jnp.array(0, dtype=jnp.int32),
    )


# Different environments have different observation spaces so we need
# one of each if the policy depends on an calculated feature, e.g total stock
# for (s,S)
def de_moor_perishable_waste_conscious_S_policy(policy_params, obs, rng, mean_demand):
    # policy_params = [[S]]
    S = policy_params[0, 0]
    total_stock = obs.sum()
    stock_expiring_next_period = obs[-1]
    order = base_waste_conscious_S_policy(
        S, total_stock, stock_expiring_next_period, mean_demand
    )
    return jnp.array(order)


def hendrix_perishable_one_product_waste_conscious_S_policy(
    policy_params, obs, rng, mean_demand
):
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
    policy_params, obs, rng, max_useful_life, mean_demand_a, mean_demand_b
):
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


# TODO
def mirjalili_perishable_platelet_waste_conscious_S_policy(policy_params, obs, rng):
    # policy_params = [[S_Mon], ..., [S_Sun]]
    # weekday = obs[0]
    # S = policy_params[weekday][0]
    # total_stock = jnp.sum(obs[1:])  # First element of obs is weekday
    # order = base_S_policy(S, total_stock, policy_params)
    # return jnp.array(order)
    raise NotImplementedError
