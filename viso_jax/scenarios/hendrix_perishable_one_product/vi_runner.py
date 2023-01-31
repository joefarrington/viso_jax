import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import itertools
import logging
from viso_jax.value_iteration.base_vi_runner import ValueIterationRunner
from pathlib import Path
from jax import tree_util
from typing import Union
import chex

# Enable logging
log = logging.getLogger("ValueIterationRunner")

# NOTE: in Hendrix et al (2019), oldest stock on LHS of inventory vector
# Here, for consistency with De Moor and Mirjalili cases, oldest stock
# is now on the RHS of the inventory vector


class HendrixPerishableOneProductVIR(ValueIterationRunner):
    def __init__(
        self,
        max_useful_life: int,
        demand_poisson_mean: float,
        variable_order_cost: float,
        sales_price: float,
        max_order_quantity: int,
        max_batch_size: int,
        epsilon: float,
        gamma: float = 1,
        checkpoint_frequency: int = 0,
        resume_from_checkpoint: Union[bool, str] = False,
    ):

        """Class to run value iteration for de_moor_perishable scenario

        Args:
            max_useful_life: maximum useful life of product, m >= 1
            demand_poission_mean: mean of Poisson distribution that models demand
            variable_order_cost: cost per unit ordered
            sales_price: revenue per units issued to meet demand
            max_order_quantity: maximum order quantity
            max_batch_size: Maximum number of states to update in parallel using vmap, will depend on GPU memory
            epsilon: Convergence criterion for value iteration
            gamma: Discount factor
            checkpoint_frequency: Frequency with which to save checkpoints, 0 for no checkpoints
            resume_from_checkpoint: If False, start from scratch; if filename, resume from checkpoint

        """
        assert (
            max_useful_life >= 1
        ), "max_useful_life must be greater than or equal to 1"
        self.max_useful_life = max_useful_life
        self.demand_poisson_mean = demand_poisson_mean
        self.variable_order_cost = variable_order_cost
        self.sales_price = sales_price
        self.max_order_quantity = max_order_quantity

        self.max_batch_size = max_batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = Path("checkpoints/")
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint

        self._setup()
        log.info(f"Max order quantity: {self.max_order_quantity}")

    def _setup_before_states_actions_random_outcomes_created(self):
        """Calculate the maximum stock and maximum demand to be considered"""
        self.max_stock = self.max_order_quantity * self.max_useful_life
        self.max_demand = self.max_useful_life * (self.max_order_quantity + 2)

    def _setup_after_states_actions_random_outcomes_created(self):
        """Set up functions to calculate the expected sales revenue, used
        for the initial estimate of the value function"""

        self._calculate_expected_sales_revenue_vmap_states = jax.vmap(
            self._calculate_expected_sales_revenue
        )
        self._calculate_expected_sales_revenue_state_batch_jit = jax.jit(
            self._calculate_expected_sales_revenue_state_batch
        )
        self._calculate_expected_sales_revenue_scan_state_batches_pmap = jax.pmap(
            self._calculate_expected_sales_revenue_scan_state_batches, in_axes=(None, 0)
        )

    def generate_states(self) -> tuple[chex.Array, dict[str, int]]:
        """Returns a tuple consisting of an array of all possible states and a dictionary
        that maps descriptive names of the components of the state to indices that can be
        used to extract them from an individual state"""
        states = self._generate_states_single_product(self.max_order_quantity)

        # Use this dict to access specific parts of the state
        state_component_idx_dict = self._generate_one_product_state_component_idx_dict()
        return states, state_component_idx_dict

    def create_state_to_idx_mapping(self) -> chex.Array:
        """Returns an array that maps from a state (represented as a tuple) to its index
        in the state array"""
        state_to_idx = np.zeros(
            tuple([self.max_order_quantity + 1] * self.max_useful_life)
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
        # The transition depends on the number of units issued
        issued = jnp.arange(0, self.max_stock + 1)
        # Use this dict to access specific elements of the random outcomes
        pro_component_idx_dict = {}
        pro_component_idx_dict["issued"] = 0

        return issued, pro_component_idx_dict

    def deterministic_transition_function(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        random_outcome: chex.Array,
    ) -> tuple[chex.Array, float]:
        """Returns the next state and single-step reward for the provided state, action and random combination"""
        stock_after_issue = self._issue_fifo(
            state,
            random_outcome,
        )

        next_state = jnp.hstack(
            [action, stock_after_issue[0 : self.max_useful_life - 1]]
        )

        # Pass through the random outcome (units issued)
        single_step_reward = self._calculate_single_step_reward(
            state, action, random_outcome
        )

        return (next_state, single_step_reward)

    def get_probabilities(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
    ) -> chex.Array:
        """Returns an array of the probabilities of each possible random outcome for the provides state-action pair"""

        stock = jnp.sum(state)

        # Transition probabilities if demand for a less than or equal to stock
        demand_upto_max_stock = jnp.arange(0, self.max_stock + 1)
        prob_demand = jax.scipy.stats.poisson.pmf(
            jnp.arange(self.max_stock + 1), self.demand_poisson_mean
        )
        issued_probs = prob_demand * (jnp.arange(self.max_stock + 1) <= stock)

        # Plus component for demand greater than current stock, issue all stock
        prob_demand_gt_stock = 1 - jax.scipy.stats.poisson.cdf(
            stock, self.demand_poisson_mean
        )
        issued_probs = issued_probs.at[stock].add(prob_demand_gt_stock)

        return issued_probs

    def calculate_initial_values(self) -> chex.Array:
        """Returns an array of the initial values for each state, based on the
        expected one step ahead sales revenue"""

        padded_batched_expected_sales_revenue = (
            self._calculate_expected_sales_revenue_scan_state_batches_pmap(
                None, self.padded_batched_states
            )
        )

        expected_sales_revenue = self._unpad(
            padded_batched_expected_sales_revenue.reshape(-1), self.n_pad
        )

        return expected_sales_revenue

    def check_converged(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Convergence check to determine whether to stop value iteration. This convergence check
        is testing for the convergence of the policy, and will stop value iteration
        when the values for every state are changing by approximately the same amount."""
        delta = V - V_old
        max_delta = jnp.max(delta)
        min_delta = jnp.min(delta)
        delta_diff = max_delta - min_delta
        if delta_diff < self.epsilon:
            if iteration >= min_iter:
                log.info(f"Converged on iteration {iteration}")
                log.info(f"Max delta: {max_delta}")
                log.info(f"Min delta: {min_delta}")
                return True
            else:
                log.info(
                    f"Difference below epsilon on iteration {iteration}, but min iterations not reached"
                )
                return False
        else:
            log.info(f"Iteration {iteration}, delta diff: {delta_diff}")
            return False

    ### Supporting functions for self.generate_states() ###
    def _generate_states_single_product(self, max_order_quantity: int) -> list[tuple]:
        """Returns possible states, as a list of tuples"""
        possible_orders = range(0, max_order_quantity + 1)
        product_arg = [possible_orders] * self.max_useful_life
        return list(itertools.product(*product_arg))

    def _generate_one_product_state_component_idx_dict(self) -> dict[str, int]:
        """Returns a dictionary that maps descriptive names of the components of a state
        to indices of the elements in the state array"""
        state_component_idx_dict = {}
        state_component_idx_dict["stock_start"] = 0
        state_component_idx_dict["stock_len"] = self.max_useful_life
        state_component_idx_dict["stock_stop"] = (
            state_component_idx_dict["stock_start"]
            + state_component_idx_dict["stock_len"]
        )
        return state_component_idx_dict

    ### Supporting functions for self.deterministic_transition_function() ###
    def _issue_fifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using FIFO policy"""
        # Oldest stock on RHS of vector, so reverse
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: chex.Array, stock_element: int
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
        cost = action * self.variable_order_cost
        revenue = transition_function_reward_output * self.sales_price
        return revenue - cost

    ##### Support functions for self._calculate_single_step_reward() #####
    def _calculate_sales_revenue_for_possible_random_outcomes(self) -> chex.Array:
        """Calculate the sales revenue for each possible random outcome of demand"""
        return self.possible_random_outcomes * self.sales_price

    def _calculate_expected_sales_revenue(self, state: chex.Array) -> float:
        """Calculate the expected sales revenue for a given state"""
        issued_probabilities = self.get_probabilities(state, None, None)
        expected_sales_revenue = issued_probabilities.dot(
            self._calculate_sales_revenue_for_possible_random_outcomes()
        )
        return expected_sales_revenue

    def _calculate_expected_sales_revenue_state_batch(
        self, carry: None, batch_of_states: chex.Array
    ) -> tuple[None, chex.Array]:
        """Calculate the expected sales revenue for a batch of states"""
        revenue = self._calculate_expected_sales_revenue_vmap_states(batch_of_states)
        return carry, revenue

    def _calculate_expected_sales_revenue_scan_state_batches(
        self, carry: None, padded_batched_states: chex.Array
    ) -> chex.Array:
        """Calculate the expected sales revenue for multiple batches of states, using jax.lax.scan to loop over the batches of states"""
        carry, revenue_padded = jax.lax.scan(
            self._calculate_expected_sales_revenue_state_batch_jit,
            carry,
            padded_batched_states,
        )
        return revenue_padded

    ### Utility functions to set up pytree for class ###
    # See https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

    def _tree_flatten(self):
        children = (
            self.state_to_idx_mapping,
            self.states,
            self.padded_batched_states,
            self.actions,
            self.possible_random_outcomes,
            self.V_old,
            self.iteration,
        )  # arrays / dynamic values
        aux_data = {
            "max_useful_life": self.max_useful_life,
            "demand_poisson_mean": self.demand_poisson_mean,
            "variable_order_cost": self.variable_order_cost,
            "sales_price": self.sales_price,
            "max_order_quantity": self.max_order_quantity,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "n_devices": self.n_devices,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "checkpoint_frequency": self.checkpoint_frequency,
            "cp_path": self.cp_path,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "max_stock": self.max_stock,
            "max_demand": self.max_demand,
            "state_tuples": self.state_tuples,
            "action_labels": self.action_labels,
            "state_component_idx_dict": self.state_component_idx_dict,
            "pro_component_idx_dict": self.pro_component_idx_dict,
            "n_pad": self.n_pad,
            "output_info": self.output_info,
        }  # static values
        return (children, aux_data)


tree_util.register_pytree_node(
    HendrixPerishableOneProductVIR,
    HendrixPerishableOneProductVIR._tree_flatten,
    HendrixPerishableOneProductVIR._tree_unflatten,
)
