import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import itertools
import logging
from viso_jax.value_iteration.base_vi_runner import (
    ValueIterationRunner,
)
from omegaconf import OmegaConf
from pathlib import Path
from jax import tree_util
import numpyro
from typing import Union
import chex

# Enable logging
log = logging.getLogger("ValueIterationRunner")

# This is based on MM PhD thesis, Chapter 6
# Demand is Negative Binomial
# Uncertainity distribution for age on arrival depends on order quantity


class MirjaliliPerishablePlateletVIR(ValueIterationRunner):
    def __init__(
        self,
        max_demand: int,
        weekday_demand_negbin_n: list[float],  # [M, T, W, T, F, S, S]
        weekday_demand_negbin_delta: list[float],  # [M, T, W, T, F, S, S]
        max_useful_life: int,
        shelf_life_at_arrival_distribution_c_0: list[float],
        shelf_life_at_arrival_distribution_c_1: list[float],
        max_order_quantity: int,
        variable_order_cost: float,
        fixed_order_cost: float,
        shortage_cost: float,
        wastage_cost: float,
        holding_cost: float,
        max_batch_size: int,
        epsilon: float,
        gamma: float = 0.95,
        checkpoint_frequency: int = 1,
        resume_from_checkpoint: Union[bool, str] = False,
        periodic_convergence_check: bool = True,
    ):

        """Class to run value iteration for mirjalili_perishable_platelet scenario

        Args:
            max_demand: int,
            weekday_demand_negbin_n: parameter n of the negative binomial distribution, one for each weekday in order [M, T, W, T, F, S, S]
            weekday_demand_negbin_delta: paramter delta of the negative binomial distribution, one for each weekday in order [M, T, W, T, F, S, S]
            max_useful_life: maximum useful life of product, m >= 1
            shelf_life_at_arrival_distribution_c_0: paramter c_0 used to determine parameters of multinomial distribution of useful life on arrival in order [2, ..., m]
            shelf_life_at_arrival_distribution_c_1: paramter c_1 used to determine parameters of multinomial distribution of useful life on arrival in order [2, ..., m]
            max_order_quantity: maximum order quantity
            variable_order_cost: cost per unit ordered
            fixed_order_cost: cost incurred when order > 0
            shortage_cost: cost per unit of demand not met
            wastage_cost: cost per unit of product that expires before use
            holding_cost: cost per unit of product in stock at the end of the day
            max_batch_size: Maximum number of states to update in parallel using vmap, will depend on GPU memory
            epsilon: Convergence criterion for value iteration
            gamma: Discount factor
            checkpoint_frequency: Frequency with which to save checkpoints, 0 for no checkpoints
            resume_from_checkpoint: If False, start from scratch; if filename, resume from checkpoint
            periodic_convergence_check: If True, use periodic convergence check, otherwise test for convergence of values themselves

        """

        if self._is_shelf_life_at_arrival_distribution_valid(
            shelf_life_at_arrival_distribution_c_0,
            shelf_life_at_arrival_distribution_c_1,
            max_useful_life,
        ):
            self.shelf_life_at_arrival_distribution_c_0 = jnp.array(
                shelf_life_at_arrival_distribution_c_0
            )
            self.shelf_life_at_arrival_distribution_c_1 = jnp.array(
                shelf_life_at_arrival_distribution_c_1
            )

        # Calculate probability of success, from parameterisation provided in MM thesis
        self.weekday_demand_negbin_n = jnp.array(weekday_demand_negbin_n)
        self.weekday_demand_negbin_delta = jnp.array(weekday_demand_negbin_delta)
        self.weekday_demand_negbin_p = self.weekday_demand_negbin_n / (
            self.weekday_demand_negbin_delta + self.weekday_demand_negbin_n
        )
        self.max_demand = max_demand

        assert (
            max_useful_life >= 1
        ), "max_useful_life must be greater than or equal to 1"
        self.max_useful_life = max_useful_life
        self.max_order_quantity = max_order_quantity
        self.cost_components = jnp.array(
            [
                variable_order_cost,
                fixed_order_cost,
                shortage_cost,
                wastage_cost,
                holding_cost,
            ]
        )
        self.max_batch_size = max_batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = Path("checkpoints_op/")
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint

        self.periodic_convergence_check = periodic_convergence_check
        if periodic_convergence_check:
            assert (
                self.checkpoint_frequency == 1
            ), "Checkpoint frequency must be 1 to use periodic convergence check"

        self.weekdays = {
            0: "monday",
            1: "tuesday",
            2: "wednesday",
            3: "thursday",
            4: "friday",
            5: "saturday",
            6: "sunday",
        }

        self._setup()

    def generate_states(self) -> tuple[chex.Array, dict[str, int]]:
        """Returns a tuple consisting of an array of all possible states and a dictionary
        that maps descriptive names of the components of the state to indices that can be
        used to extract them from an individual state"""

        possible_orders = range(0, self.max_order_quantity + 1)
        product_arg = [possible_orders] * (self.max_useful_life - 1)
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
        return state_tuples, state_component_idx_dict

    def create_state_to_idx_mapping(self) -> chex.Array:
        """Returns an array that maps from a state (represented as a tuple) to its index
        in the state array"""
        state_to_idx = np.zeros(
            (
                len(self.weekdays.keys()),
                *[self.max_order_quantity + 1] * (self.max_useful_life - 1),
            )
        )
        for idx, state in enumerate(self.state_tuples):
            state_to_idx[state] = idx
        state_to_idx = jnp.array(state_to_idx, dtype=jnp.int32)
        return state_to_idx

    def generate_actions(self) -> tuple[chex.Array, list[str]]:
        """Returns a tuple consisting of an array of all possible actions and a
        list of descriptive names for each action dimension"""
        actions = jnp.arange(0, self.max_order_quantity + 1)
        action_labels = ["order_quantity"]
        return actions, action_labels

    def generate_possible_random_outcomes(self) -> tuple[chex.Array, dict[str, int]]:
        """Returns a tuple consisting of an array of all possible random outcomes and a dictionary
        that maps descriptive names of the components of a random outcome to indices that can be
        used to extract them from an individual random outcome."""
        # Possible demands
        demands = jnp.arange(self.max_demand + 1).reshape(1, -1)

        # Possible received order quantities split by age
        rec_combos = np.array(
            list(
                itertools.product(
                    *[
                        list(range(self.max_order_quantity + 1))
                        for i in range(self.max_useful_life)
                    ]
                )
            )
        )
        jnp_rec_combos = jnp.array(rec_combos)
        # Exclude any where total received greater than max_order_quantity
        jnp_rec_combos = jnp_rec_combos[
            jnp_rec_combos.sum(axis=1) <= self.max_order_quantity
        ]

        # Combine the two random elements - demand and remaining useful life on arrival
        demands_repeated = demands.repeat(len(jnp_rec_combos), axis=0).reshape(-1, 1)
        received_units_repeated = jnp.repeat(
            jnp_rec_combos, self.max_demand + 1, axis=0
        )
        possible_random_outcomes = jnp.hstack(
            [demands_repeated, received_units_repeated]
        )

        pro_component_idx_dict = {}
        pro_component_idx_dict["demand"] = 0
        pro_component_idx_dict["order_start"] = 1
        pro_component_idx_dict["order_stop"] = self.max_useful_life + 1
        return possible_random_outcomes, pro_component_idx_dict

    def deterministic_transition_function(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        random_outcome: chex.Array,
    ) -> tuple[chex.Array, float]:
        """Returns the next state and single-step reward for the provided state, action and random combination"""
        demand = random_outcome[self.pro_component_idx_dict["demand"]]
        max_stock_received = random_outcome[
            self.pro_component_idx_dict["order_start"] : self.pro_component_idx_dict[
                "order_stop"
            ]
        ]
        opening_stock_after_delivery = (
            jnp.hstack(
                [
                    0,
                    state[
                        self.state_component_idx_dict[
                            "stock_start"
                        ] : self.state_component_idx_dict["stock_stop"]
                    ],
                ]
            )
            + max_stock_received
        )

        # Limit any one element of opening stock to max_order_quantity
        # Assume any units that would take an element over this are
        # not accepted at delivery
        opening_stock_after_delivery = opening_stock_after_delivery.clip(
            0, self.max_order_quantity
        )

        stock_after_issue = self._issue_oufo(opening_stock_after_delivery, demand)

        # Compute variables required to calculate the cost
        variable_order = action
        fixed_order = action > 0
        shortage = jnp.max(
            jnp.array([demand - jnp.sum(opening_stock_after_delivery), 0])
        )
        expiries = stock_after_issue[-1]
        closing_stock = stock_after_issue[0 : self.max_useful_life - 1]
        holding = jnp.sum(closing_stock)

        # These components must be in the same order as self.cost_components
        transition_function_reward_output = jnp.array(
            [variable_order, fixed_order, shortage, expiries, holding]
        )

        # Update the weekday
        next_weekday = (state[self.state_component_idx_dict["weekday"]] + 1) % 7

        next_state = jnp.hstack([next_weekday, closing_stock]).astype(jnp.int32)

        single_step_reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output
        )

        return next_state, single_step_reward

    def get_probabilities(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
    ) -> chex.Array:
        """Returns an array of the probabilities of each possible random outcome for the provides state-action pair"""
        weekday = state[self.state_component_idx_dict["weekday"]]
        n = self.weekday_demand_negbin_n[weekday]
        p = self.weekday_demand_negbin_p[weekday]
        # tfd NegBin is distribution over successes until observe `total_count` failures,
        # versus MM thesis where distribtion over failures until certain number of successes
        # Therefore use 1-p for prob (prob of failure is 1 - prob of success)
        demand_dist = numpyro.distributions.NegativeBinomialProbs(
            total_count=n, probs=(1 - p)
        )
        demand_probs = jnp.exp(demand_dist.log_prob(jnp.arange(0, self.max_demand + 1)))
        # Truncate distribution as in Eq 6.23 of thesis
        # by adding probability mass for demands > max_demand to max_demand
        demand_probs = demand_probs.at[self.max_demand].add(1 - jnp.sum(demand_probs))

        demand_component_probs = demand_probs[
            possible_random_outcomes[:, self.pro_component_idx_dict["demand"]]
        ]

        multinomial_logits = self._get_multinomial_logits(action)
        dist = numpyro.distributions.Multinomial(
            logits=multinomial_logits, total_count=action
        )
        received_component_probs = jnp.where(
            possible_random_outcomes[
                :,
                self.pro_component_idx_dict[
                    "order_start"
                ] : self.pro_component_idx_dict["order_stop"],
            ].sum(axis=1)
            == action,
            jnp.exp(
                dist.log_prob(
                    possible_random_outcomes[
                        :,
                        self.pro_component_idx_dict[
                            "order_start"
                        ] : self.pro_component_idx_dict["order_stop"],
                    ]
                )
            ),
            0,
        )
        return demand_component_probs * received_component_probs

    def calculate_initial_values(self) -> chex.Array:
        """Returns an array of the initial values for each state"""
        return jnp.zeros(len(self.states))

    def check_converged(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Convergence check to determine whether to stop value iteration. We allow the
        choice of two convergence checks based on the value of
        self.periodic_convergence_check: the periodic convergence check will generally
        require fewer iterations, but the standard convergence check can be used if
        the values are needed"""
        # By default, we used periodic convergence check
        # because we're interested in the policy rather than the value function
        if self.periodic_convergence_check:
            return self._check_converged_periodic(iteration, min_iter, V, V_old)
        else:
            # But we can also check for convergence of value function itself
            # as in DeMoor case
            return self._check_converged_v(iteration, min_iter, V, V_old)

    def _check_converged_periodic(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Periodic convergence check to determine whether to stop value iteration. Stops when the (undiscounted) change in
        a value over a period is the same for every state. The periodicity is 7 - the days of the week"""
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

    def _check_converged_v(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Standard convergence check to determine whether to stop value iteration. Stops when there is
        approximately (based on epsilon) no change between estimates of the value function at successive iterations"""
        # Here we use a discount factor, so
        # We want biggest change to a value to be less than epsilon
        # This is a difference conv check to the others
        max_delta = jnp.max(jnp.abs(V - V_old))
        if max_delta < self.epsilon:
            if iteration >= min_iter:
                log.info(f"Converged on iteration {iteration}")
                return True
            else:
                log.info(
                    f"Max delta below epsilon on iteration {iteration}, but min iterations not reached"
                )
                return False
        else:
            log.info(f"Iteration {iteration}, max delta: {max_delta}")
            return False

    ### Supporting functions for self._init__() ###
    def _is_shelf_life_at_arrival_distribution_valid(
        self,
        shelf_life_at_arrival_distribution_c_0: list[float],
        shelf_life_at_arrival_distribution_c_1: list[float],
        max_useful_life: int,
    ) -> bool:
        """Check that the shelf life at arrival distribution parameters are valid"""
        assert (
            len(shelf_life_at_arrival_distribution_c_0) == max_useful_life - 1
        ), "Shelf life at arrival distribution params should include an item for c_0 with max_useful_life - 1 parameters"
        assert (
            len(shelf_life_at_arrival_distribution_c_1) == max_useful_life - 1
        ), "Shelf life at arrival distribution params should include an item for c_1 with max_useful_life - 1 parameters"
        return True

    ### Supporting function for self.get_probabilities() ###
    def _get_multinomial_logits(self, action: int) -> chex.Array:
        """Return multinomial logits for a given order quantity action"""
        c_0 = self.shelf_life_at_arrival_distribution_c_0
        c_1 = self.shelf_life_at_arrival_distribution_c_1
        # Assume logit for useful_life=1 is 0, concatenate with logits
        # for other ages using provided coefficients and order size action

        # Parameters are provided in ascending remaining shelf life
        # So reverse to match ordering of stock array which is in
        # descending order of remaining useful life so that oldest
        # units are on the RHS
        return jnp.hstack([0, c_0 + (c_1 * action)])[::-1]

    ### Supporting function for self.deterministic_transition_function() ###
    def _issue_oufo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using OUFO policy"""
        # Oldest stock on RHS of vector, so reverse
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: int, stock_element: int
    ) -> tuple[int, int]:
        """Fill demand with stock of one age, representing one element in the state"""
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    def _calculate_single_step_reward(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        transition_function_reward_output: chex.Array,
    ) -> float:
        """Calculate the single step reward based on the provided state, action and
        output from the transition function"""
        # Minus one to reflect the fact that they are costs
        cost = jnp.dot(transition_function_reward_output, self.cost_components)
        reward = -1 * cost
        return reward

    ### Supporting functions for self._check_converged_periodic() ###

    def _calculate_period_deltas_without_discount(
        self, V: chex.Array, current_iteration: int, period: int
    ) -> tuple[float, float]:
        """Return the min and max change in the value function over a period if there is no discount factor"""
        # If there's no discount factor, just subtract Values one period ago
        # from current value estimate
        fname = self.cp_path / f"values_{current_iteration - period}.csv"
        V_one_period_ago_df = pd.read_csv(fname, index_col=0)
        V_one_period_ago = jnp.array(V_one_period_ago_df.values).reshape(-1)
        max_period_delta = jnp.max(V - V_one_period_ago)
        min_period_delta = jnp.min(V - V_one_period_ago)
        return min_period_delta, max_period_delta

    def _calculate_period_deltas_with_discount(
        self, V: chex.Array, current_iteration: int, period: int, gamma: float
    ) -> tuple[float, float]:
        """Return the min and max undiscounted change in the value function over a period
        if there is a discount factor"""
        # If there is a discount factor, we need to sum the differences between
        # each step in the period and adjust for the discount factor
        values_dict = self._read_multiple_previous_values(current_iteration, period)
        values_dict[current_iteration] = V
        period_deltas = jnp.zeros_like(V)
        for p in range(period):
            period_deltas += (
                values_dict[current_iteration - p]
                - values_dict[current_iteration - p - 1]
            ) / (gamma ** (current_iteration - p - 1))
        min_period_delta = jnp.min(period_deltas)
        max_period_delta = jnp.max(period_deltas)
        return min_period_delta, max_period_delta

    def _read_multiple_previous_values(
        self, current_iteration: int, period: int
    ) -> dict[int, chex.Array]:
        """Load the value functions from multiple previous iterations fo calculate period deltas"""
        values_dict = {}
        for p in range(1, period + 1):
            j = current_iteration - p
            fname = self.cp_path / f"values_{j}.csv"
            values_dict[j] = jnp.array(pd.read_csv(fname)["V"].values)
        return values_dict

    def _tree_flatten(self):
        children = (
            self.weekday_demand_negbin_n,
            self.weekday_demand_negbin_delta,
            self.weekday_demand_negbin_p,
            self.shelf_life_at_arrival_distribution_c_0,
            self.shelf_life_at_arrival_distribution_c_1,
            self.cost_components,
            self.state_to_idx_mapping,
            self.states,
            self.padded_batched_states,
            self.actions,
            self.possible_random_outcomes,
            self.V_old,
            self.iteration,
        )  # arrays / dynamic values
        aux_data = {
            "max_demand": self.min_demand,
            "max_useful_life": self.max_useful_life,
            "max_order_quantity": self.max_order_quantity,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "n_devices": self.n_devices,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "checkpoint_frequency": self.checkpoint_frequency,
            "cp_path": self.cp_path,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "state_tuples": self.state_tuples,
            "action_labels": self.action_labels,
            "state_component_idx_dict": self.state_component_idx_dict,
            "pro_component_idx_dict": self.pro_component_idx_dict,
            "n_pad": self.n_pad,
            "output_info": self.output_info,
            "weekdays": self.weekdays,
        }


tree_util.register_pytree_node(
    MirjaliliPerishablePlateletVIR,
    MirjaliliPerishablePlateletVIR._tree_flatten,
    MirjaliliPerishablePlateletVIR._tree_unflatten,
)


class MDet(MirjaliliPerishablePlateletVIR):
    def generate_possible_random_outcomes(self):
        possible_random_outcomes = jnp.arange(0, self.max_demand + 1).reshape(-1, 1)
        pro_component_idx_dict = {}
        pro_component_idx_dict["demand"] = 0

        return possible_random_outcomes, pro_component_idx_dict

    def get_probabilities(self, state, action, possible_random_outcomes):
        weekday = state[self.state_component_idx_dict["weekday"]]
        n = self.weekday_demand_negbin_n[weekday]
        p = self.weekday_demand_negbin_p[weekday]
        # tfd NegBin is distribution over successes until observe `total_count` failures,
        # versus MM thesis where distribtion over failures until certain number of successes
        # Therefore use 1-p for prob (prob of failure is 1 - prob of success)
        demand_dist = numpyro.distributions.NegativeBinomialProbs(
            total_count=n, probs=(1 - p)
        )
        demand_probs = jnp.exp(demand_dist.log_prob(jnp.arange(0, self.max_demand + 1)))
        # Truncate distribution as in Eq 6.23 of thesis
        # by adding probability mass for demands > max_demand to max_demand
        demand_probs = demand_probs.at[self.max_demand].add(1 - jnp.sum(demand_probs))

        return demand_probs

    def deterministic_transition_function(self, state, action, random_outcome):
        demand = random_outcome[self.pro_component_idx_dict["demand"]]
        opening_stock_after_delivery = jnp.hstack(
            [
                action,
                state[
                    self.state_component_idx_dict[
                        "stock_start"
                    ] : self.state_component_idx_dict["stock_stop"]
                ],
            ]
        )
        stock_after_issue = self._issue_oufo(opening_stock_after_delivery, demand)

        # Compute variables required to calculate the cost
        variable_order = action
        fixed_order = action > 0
        shortage = jnp.max(
            jnp.array([demand - jnp.sum(opening_stock_after_delivery), 0])
        )
        expiries = stock_after_issue[-1]
        closing_stock = stock_after_issue[0 : self.max_useful_life - 1]
        holding = jnp.sum(closing_stock)

        # These components must be in the same order as self.cost_components
        transition_function_reward_output = jnp.array(
            [variable_order, fixed_order, shortage, expiries, holding]
        )

        # Update the weekday
        next_weekday = (state[self.state_component_idx_dict["weekday"]] + 1) % 7

        next_state = jnp.hstack([next_weekday, closing_stock]).astype(jnp.int32)

        single_step_reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output
        )

        return next_state, single_step_reward
