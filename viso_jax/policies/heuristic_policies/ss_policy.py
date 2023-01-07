import jax.numpy as jnp
import numpy as np
from typing import Optional
from functools import partial
import pandas as pd


class sSPolicy:
    def __init__(
        self,
        env_id: str,
        env_kwargs: Optional[dict] = {},
        env_params: Optional[dict] = {},
        policy_params_filepath: Optional[str] = None,
    ):

        self.param_col_names = np.array(["s", "S"])
        self.param_row_names = None
        if env_id == "DeMoorPerishable":
            self.forward = de_moor_perishable_sS_policy
        elif env_id == "HendrixPerishableOneProduct":
            self.forward = hendrix_perishable_one_product_sS_policy
        elif env_id == "HendrixPerishableSubstitutionTwoProduct":
            self.forward = partial(
                hendrix_perishable_substitution_two_product_sS_policy,
                max_useful_life=env_kwargs["max_useful_life"],
            )
            self.param_row_names = ["a", "b"]
        elif env_id == "MirjaliliPerishablePlatelet":
            self.forward = mirjalili_perishable_platelet_sS_policy
            self.param_row_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        else:
            raise ValueError(f"No (s,S) policy defined for Environment ID {env_id}")
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


def base_sS_policy(s, S, total_stock, policy_params):
    # s should be less than S
    # Enforce that constraint here, order only made when constraint met
    constaint_met = jnp.all(policy_params[:, 0] < policy_params[:, 1])
    return jnp.where((total_stock <= s) & (constaint_met), S - total_stock, 0)


# Different environments have different observation spaces so we need
# one of each if the policy depends on an calculated feature, e.g total stock
# for (s,S)
def de_moor_perishable_sS_policy(policy_params, obs, rng):
    # policy_params = [[s,S]]
    s = policy_params[0][0]
    S = policy_params[1][1]
    total_stock = obs.sum()
    order = base_sS_policy(s, S, total_stock, policy_params)
    return jnp.array(order)


def hendrix_perishable_one_product_sS_policy(policy_params, obs, rng):
    # policy_params = [[s,S]]
    s = policy_params[0][0]
    S = policy_params[1][1]
    total_stock = obs.sum()
    order = base_sS_policy(s, S, total_stock, policy_params)
    return jnp.array(order)


# Calculating the total stock for each product depends on the max_useful_life
# Use partial() in the policy_constructor to set it
def hendrix_perishable_substitution_two_product_sS_policy(
    policy_params, obs, rng, max_useful_life
):
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


def mirjalili_perishable_platelet_sS_policy(policy_params, obs, rng):
    # policy_params = [[s_Mon, S_Mon], ..., [s_Sun, S_Sun]]
    weekday = obs[0]
    s = policy_params[weekday][0]
    S = policy_params[weekday][1]
    total_stock = jnp.sum(obs[1:])  # First element of obs is weekday
    order = base_sS_policy(s, S, total_stock, policy_params)
    return jnp.array(order)
