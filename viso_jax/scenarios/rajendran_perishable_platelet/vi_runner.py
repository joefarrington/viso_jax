import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import itertools
import logging
import viso_jax
from viso_jax.value_iteration.base_vi_runner import (
    ValueIterationRunner,
)
from omegaconf import OmegaConf
from pathlib import Path
from jax import tree_util
import numpyro


# Enable logging
log = logging.getLogger("ValueIterationRunner")

# For now, assume all arrives fresh and 3 periods of useful life
# Note that, unlike the other examples, oldest stock is on the LHS
# of the vector for consistency with prev stochastic programming and RL
# work.


class OnePerishablePeriodicDemandVIR(ValueIterationRunner):
    def __init__(
        self,
        min_demand,
        max_demand,
        weekday_demand_params,  # [M, T, W, T, F, S, S]
        max_useful_life,
        lead_time,
        max_order_quantity,
        variable_order_cost,
        fixed_order_cost,
        shortage_cost,
        wastage_cost,
        holding_cost,
        batch_size,
        epsilon,
        gamma=1,
        checkpoint_frequency=1,  # Zero for no checkpoints, otherwise every x iterations
        resume_from_checkpoint=False,  # Set to checkpoint file path to restore
        use_pmap=False,
    ):
        self.min_demand = min_demand
        self.max_demand = max_demand
        # Roll weekday demands - e.g. in transition for Monday state we're interested in Tuesday's demand
        self.weekday_demand_params = {
            k: jnp.array(np.roll(v, -1)) for k, v in weekday_demand_params.items()
        }
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantity = max_order_quantity
        self.costs = jnp.array(
            [
                variable_order_cost,
                fixed_order_cost,
                shortage_cost,
                wastage_cost,
                holding_cost,
            ]
        )
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        self.weekdays = {
            0: "monday",
            1: "tuesday",
            2: "wednesday",
            3: "thursday",
            4: "friday",
            5: "saturday",
            6: "sunday",
        }

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = Path("checkpoints_op/")
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint

        self.use_pmap = use_pmap

        self.setup()
        pass

    def generate_states(self):
        possible_orders = range(0, self.max_order_quantity + 1)
        product_arg = [possible_orders] * (self.max_useful_life + self.lead_time - 1)
        stock_states = list(itertools.product(*product_arg))
        state_tuples = [
            (w, *stock) for w, stock in (itertools.product(np.arange(7), stock_states))
        ]

        state_component_idx_dict = {}
        state_component_idx_dict["weekday"] = 0
        state_component_idx_dict["stock_start"] = 1
        state_component_idx_dict["stock_len"] = self.max_useful_life - 1
        state_component_idx_dict["stock_stop"] = (
            state_component_idx_dict["stock_start"]
            + state_component_idx_dict["stock_len"]
        )
        state_component_idx_dict["in_transit_start"] = state_component_idx_dict[
            "stock_stop"
        ]
        state_component_idx_dict["in_transit_len"] = self.lead_time
        state_component_idx_dict["in_transit_stop"] = (
            state_component_idx_dict["in_transit_start"]
            + state_component_idx_dict["in_transit_len"]
        )

        return state_tuples, state_component_idx_dict

    def create_state_to_idx_mapping(self):
        state_to_idx = np.zeros(
            (
                len(self.weekdays.keys()),
                *[self.max_order_quantity + 1]
                * (self.max_useful_life + self.lead_time - 1),
            )
        )
        for idx, state in enumerate(self.state_tuples):
            state_to_idx[state] = idx
        state_to_idx = jnp.array(state_to_idx, dtype=jnp.int32)
        return state_to_idx

    def generate_actions(self):
        actions = jnp.arange(0, self.max_order_quantity + 1)
        action_labels = ["order_quantity"]
        return actions, action_labels

    def generate_possible_random_outcomes(self):
        possible_random_outcomes = jnp.arange(
            self.min_demand, self.max_demand + 1
        ).reshape(-1, 1)
        pro_component_idx_dict = {}
        pro_component_idx_dict["demand"] = 0

        return possible_random_outcomes, pro_component_idx_dict

    def deterministic_transition_function(self, state, action, random_outcome):
        demand = random_outcome[self.pro_component_idx_dict["demand"]]

        in_transit = state[
            self.state_component_idx_dict[
                "in_transit_start"
            ] : self.state_component_idx_dict["in_transit_stop"]
        ]
        in_transit = jnp.hstack([in_transit, action])
        units_received = in_transit[0]
        in_transit = in_transit[1 : 1 + self.state_component_idx_dict["in_transit_len"]]

        opening_stock = jnp.hstack(
            [
                state[
                    self.state_component_idx_dict[
                        "stock_start"
                    ] : self.state_component_idx_dict["stock_stop"]
                ],
                0,
            ]
        ) + jnp.array([0, 0, units_received])

        stock_after_issue = self._issue_oufo(opening_stock, demand)

        # Compute variables required to calculate the cost
        variable_order = action
        fixed_order = action > 0
        shortage = jnp.where(
            demand - jnp.sum(opening_stock) > 0, demand - jnp.sum(opening_stock), 0
        )
        expiries = stock_after_issue[0]
        closing_stock = stock_after_issue[1 : self.max_useful_life]

        holding = jnp.sum(closing_stock)

        # Update the weekday
        next_weekday = (state[self.state_component_idx_dict["weekday"]] + 1) % 7

        next_state = jnp.hstack([next_weekday, closing_stock, in_transit]).astype(
            jnp.int32
        )
        # These components must be in the same order as self.costs
        transition_function_reward_output = jnp.hstack(
            [variable_order, fixed_order, shortage, expiries, holding]
        )

        single_step_reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output
        )

        return next_state, single_step_reward

    def get_probabilities(self, state, action, possible_random_outcomes):
        weekday = state[self.state_component_idx_dict["weekday"]]
        mean_demand = self.weekday_demand_params["mean"][
            weekday
        ]  # Due to roll in __init__(), pulling demand for next day
        return jax.scipy.stats.poisson.pmf(
            jnp.arange(self.min_demand, self.max_demand + 1), mean_demand
        )

    def calculate_initial_values(self):
        return jnp.zeros(len(self.states))

    def check_converged(self, iteration, min_iter, V, V_old):
        period = len(self.weekdays.keys())
        if iteration < (period + 1):
            log.info(
                f"Iteration {iteration} complete, but fewer iterations than periodicity so cannot check for convergence yet"
            )
            return False
        else:
            if self.gamma == 1:
                (
                    min_period_delta,
                    max_period_delta,
                ) = self._calculate_period_deltas_without_discount(V, iteration, period)
            else:
                (
                    min_period_delta,
                    max_period_delta,
                ) = self._calculate_period_deltas_with_discount(
                    V, iteration, period, self.gamma
                )
            delta_diff = max_period_delta - min_period_delta
            if (
                delta_diff
                <= 2
                * self.epsilon
                * jnp.array(
                    [jnp.abs(min_period_delta), jnp.abs(max_period_delta)]
                ).min()
            ):
                if iteration >= min_iter:
                    log.info(f"Converged on iteration {iteration}")
                    log.info(f"Max period delta: {max_period_delta}")
                    log.info(f"Min period delta: {min_period_delta}")
                    return True
                else:
                    log.info(
                        f"Difference below epsilon on iteration {iteration}, but min iterations not reached"
                    )
                    return False
            else:
                log.info(f"Iteration {iteration}, period delta diff: {delta_diff}")
                return False

    ##### Supporting function for deterministic transition function ####
    def _issue_oufo(self, opening_stock, demand):
        _, remaining_stock = jax.lax.scan(self._issue_one_step, demand, opening_stock)
        return remaining_stock

    def _issue_one_step(self, remaining_demand, stock_element):
        remaining_stock = jnp.where(
            stock_element - remaining_demand > 0, stock_element - remaining_demand, 0
        )
        remaining_demand = jnp.where(
            remaining_demand - stock_element > 0, remaining_demand - stock_element, 0
        )
        return remaining_demand, remaining_stock

    def _calculate_single_step_reward(
        self, state, action, transition_function_reward_output
    ):
        # Minus one to reflect the fact that they are costs
        cost = transition_function_reward_output.dot(self.costs)
        reward = -1 * cost
        return reward

    ##### Supporting functions for convergence check

    def _calculate_period_deltas_without_discount(self, V, current_iteration, period):
        # If there's no discount factor, just subtract Values one period ago
        # from current value estimate
        fname = self.cp_path / f"values_{current_iteration - period}.csv"
        V_one_period_ago_df = pd.read_csv(fname, index_col=0)
        V_one_period_ago = jnp.array(V_one_period_ago_df.values).reshape(-1)
        max_period_delta = jnp.max(V - V_one_period_ago)
        min_period_delta = jnp.min(V - V_one_period_ago)
        return min_period_delta, max_period_delta

    def _calculate_period_deltas_with_discount(
        self, V, current_iteration, period, gamma
    ):
        # If there is a discount factor, we need to sum the differences between
        # each step in the period and adjust for the discount factor
        values_dict = self._read_multiple_previous_values(current_iteration, period)
        values_dict[current_iteration] = V
        period_deltas = jnp.zeros_like(V)
        for p in range(period):
            period_deltas += (
                values_dict[current_iteration - p]
                - values_dict[current_iteration - p - 1]
            ) / (gamma ** (period - p))
        min_period_delta = jnp.min(period_deltas)
        max_period_delta = jnp.max(period_deltas)
        return min_period_delta, max_period_delta

    def _read_multiple_previous_values(self, current_iteration, period):
        values_dict = {}
        for p in range(1, period + 1):
            j = current_iteration - p
            fname = self.cp_path / f"values_{j}.csv"
            values_dict[j] = jnp.array(pd.read_csv(fname)["V"].values)
        return values_dict

    def _tree_flatten(self):
        children = (
            self.weekday_demand_params,
            self.costs,
            self.state_to_idx_mapping,
            self.states,
            self.actions,
            self.possible_random_outcomes,
            self.V_old,
            self.iteration,
        )  # arrays / dynamic values
        aux_data = {
            "min_demand": self.min_demand,
            "max_demand": self.min_demand,
            "max_useful_life": self.max_useful_life,
            "batch_size": self.batch_size,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "checkpoint_frequency": self.checkpoint_frequency,
            "cp_path": self.cp_path,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "state_tuples": self.state_tuples,
            "action_labels": self.action_labels,
            "state_component_idx_dict": self.state_component_idx_dict,
            "pro_component_idx_dict": self.pro_component_idx_dict,
            "n_batches": self.n_batches,
        }

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    OnePerishablePeriodicDemandVIR,
    OnePerishablePeriodicDemandVIR._tree_flatten,
    OnePerishablePeriodicDemandVIR._tree_unflatten,
)
