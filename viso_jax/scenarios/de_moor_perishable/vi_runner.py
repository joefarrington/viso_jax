import numpy as np
import jax
import jax.numpy as jnp
import itertools
import logging
import numpyro
from viso_jax.value_iteration.base_vi_runner import ValueIterationRunner
from pathlib import Path
from jax import tree_util
from typing import Union, Dict, Tuple, List, Optional
import chex
from datetime import datetime

# Enable logging
log = logging.getLogger("ValueIterationRunner")


class DeMoorPerishableVIR(ValueIterationRunner):
    def __init__(
        self,
        max_demand: int,
        demand_gamma_mean: float,
        demand_gamma_cov: float,
        max_useful_life: int,
        lead_time: int,
        max_order_quantity: int,
        variable_order_cost: float,
        shortage_cost: float,
        wastage_cost: float,
        holding_cost: float,
        issue_policy: str = "fifo",
        max_batch_size: int = 1000,
        epsilon: float = 1e-4,
        gamma: float = 1,
        output_directory: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 0.99,
        resume_from_checkpoint: Union[bool, str] = False,
    ):
        """Class to run value iteration for de_moor_perishable scenario

        Args:
            max_demand: maximum daily demand
            demand_gamma_mean: mean of gamma distribution that models demand
            demand_gamma_cov: coefficient of variation of gamma distribution that models demand
            max_useful_life: maximum useful life of product, m >= 1
            lead_time: lead time of product, L >= 1
            max_order_quantity: maximum order quantity
            variable_order_cost: cost per unit ordered
            shortage_cost: cost per unit of demand not met
            wastage_cost: cost per unit of product that expires before use
            holding_cost: cost per unit of product in stock at the end of the day
            issue_policy: should be either 'fifo' or 'lifo'
            max_batch_size: Maximum number of states to update in parallel using vmap, will depend on GPU memory
            epsilon: Convergence criterion for value iteration
            gamma: Discount factor
            output_directory: Directory to save output to, if None, will create a new directory
            checkpoint_frequency: Frequency with which to save checkpoints, 0 for no checkpoints
            resume_from_checkpoint: If False, start from scratch; if filename, resume from checkpoint

        """

        self.max_demand = max_demand

        # Paper provides mean, CoV for gamma dist, but numpyro distribution expects alpha (concentration) and beta (rate)
        (
            self.demand_gamma_alpha,
            self.demand_gamma_beta,
        ) = self._convert_gamma_parameters(demand_gamma_mean, demand_gamma_cov)
        self.demand_probabilities = self._calculate_demand_probabilities(
            self.demand_gamma_alpha, self.demand_gamma_beta
        )

        assert (
            max_useful_life >= 1
        ), "max_useful_life must be greater than or equal to 1"
        self.max_useful_life = max_useful_life

        assert lead_time >= 1, "lead_time must be greater than or equal to 1"
        self.lead_time = lead_time

        self.max_order_quantity = max_order_quantity
        self.cost_components = jnp.array(
            [
                variable_order_cost,
                shortage_cost,
                wastage_cost,
                holding_cost,
            ]
        )

        assert issue_policy in ["fifo", "lifo"], "Issue policy must be 'fifo' or 'lifo'"
        if issue_policy == "fifo":
            self._issue_stock = self._issue_fifo
        else:
            self._issue_stock = self._issue_lifo

        self.max_batch_size = max_batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        if output_directory is None:
            now = datetime.now()
            date = now.strftime("%Y-%m-%d)")
            time = now.strftime("%H-%M-%S")
            self.output_directory = Path(f"vi_output/{date}/{time}").absolute()
        else:
            self.output_directory = Path(output_directory).absolute()

        self.checkpoint_frequency = checkpoint_frequency

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = self.output_directory / "checkpoints"
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint

        self._setup()

    def generate_states(self) -> Tuple[List[Tuple], Dict[str, int]]:
        """Returns a tuple consisting of a list of all possible states as tuples and a
        dictionary that maps descriptive names of the components of the state to indices
        that can be used to extract them from an individual state"""

        possible_orders = range(0, self.max_order_quantity + 1)
        product_arg = [possible_orders] * (self.max_useful_life + self.lead_time - 1)
        state_tuples = list(itertools.product(*product_arg))

        state_component_idx_dict = {}

        state_component_idx_dict["in_transit_start"] = 0
        state_component_idx_dict["in_transit_len"] = self.lead_time - 1
        state_component_idx_dict["in_transit_stop"] = (
            state_component_idx_dict["in_transit_start"]
            + state_component_idx_dict["in_transit_len"]
        )

        state_component_idx_dict["stock_start"] = state_component_idx_dict[
            "in_transit_stop"
        ]
        state_component_idx_dict["stock_len"] = self.max_useful_life
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
                *[self.max_order_quantity + 1]
                * (self.max_useful_life + self.lead_time - 1),
            )
        )
        for idx, state in enumerate(self.state_tuples):
            state_to_idx[state] = idx
        state_to_idx = jnp.array(state_to_idx, dtype=jnp.int32)
        return state_to_idx

    def generate_actions(self) -> Tuple[chex.Array, List[str]]:
        """Returns a tuple consisting of an array of all possible actions and a
        list of descriptive names for each action dimension"""
        actions = jnp.arange(0, self.max_order_quantity + 1)
        action_labels = ["order_quantity"]
        return actions, action_labels

    def generate_possible_random_outcomes(self) -> Tuple[chex.Array, Dict[str, int]]:
        """Returns a tuple consisting of an array of all possible random outcomes and a dictionary
        that maps descriptive names of the components of a random outcome to indices that can be
        used to extract them from an individual random outcome."""
        possible_random_outcomes = jnp.arange(0, self.max_demand + 1).reshape(-1, 1)
        pro_component_idx_dict = {}
        pro_component_idx_dict["demand"] = 0

        return possible_random_outcomes, pro_component_idx_dict

    def deterministic_transition_function(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        random_outcome: chex.Array,
    ) -> Tuple[chex.Array, float]:
        """Returns the next state and single-step reward for the provided state, action and random combination"""
        demand = random_outcome[self.pro_component_idx_dict["demand"]]

        opening_in_transit = state[
            self.state_component_idx_dict[
                "in_transit_start"
            ] : self.state_component_idx_dict["in_transit_stop"]
        ]

        opening_stock = state[
            self.state_component_idx_dict[
                "stock_start"
            ] : self.state_component_idx_dict["stock_stop"]
        ]

        in_transit = jnp.hstack([action, opening_in_transit])

        stock_after_issue = self._issue_stock(opening_stock, demand)

        # Compute variables required to calculate the cost
        variable_order = action
        shortage = jnp.max(jnp.array([demand - jnp.sum(opening_stock), 0]))
        expiries = stock_after_issue[-1]
        holding = jnp.sum(stock_after_issue[0 : self.max_useful_life - 1])
        # These components must be in the same order as self.cost_components
        transition_function_reward_output = jnp.hstack(
            [variable_order, shortage, expiries, holding]
        )

        # Calculate single step reward
        single_step_reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output
        )

        # Age stock and Receive order placed at step t-(L-1)
        closing_stock = jnp.hstack(
            [in_transit[-1], stock_after_issue[0 : self.max_useful_life - 1]]
        )
        closing_in_transit = in_transit[0 : self.lead_time - 1]

        next_state = jnp.hstack([closing_in_transit, closing_stock]).astype(jnp.int32)

        return next_state, single_step_reward

    def get_probabilities(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
    ) -> chex.Array:
        """Returns an array of the probabilities of each possible random outcome for the provides state-action pair"""
        # Same probabilities for every (state, action) pair
        # so calculate once during setup
        return self.demand_probabilities

    def calculate_initial_values(self) -> chex.Array:
        """Returns an array of the initial values for each state"""
        return jnp.zeros(len(self.states))

    def check_converged(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Convergence check to determine whether to stop value iteration. This convergence check
        is testing for the convergence of the value function itself, and will stop value iteration
        when the values for every state are approximately equal to the values from the previous iteration"""

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

    ### Supporting function for self.deterministic_transition_function() ###
    def _issue_fifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using FIFO policy"""
        # Oldest stock on RHS of vector, so reverse
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_lifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using LIFO policy"""
        # Freshest stock on LHS of vector
        _, remaining_stock = jax.lax.scan(self._issue_one_step, demand, opening_stock)
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: int, stock_element: int
    ) -> Tuple[int, int]:
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
        cost = jnp.dot(transition_function_reward_output, self.cost_components)
        # Multiply by -1 to reflect the fact that they are costs
        reward = -1 * cost
        return reward

    ### Supporting functions for self.get_probabilities() ###
    def _convert_gamma_parameters(self, mean: float, cov: float) -> Tuple[float, float]:
        """Convert mean and coefficient of variation to gamma distribution parameters required
        by numpyro.distributions.Gamma"""
        alpha = 1 / (cov**2)
        beta = 1 / (mean * cov**2)
        return alpha, beta

    def _calculate_demand_probabilities(
        self, gamma_alpha: float, gamma_beta: float
    ) -> chex.Array:
        """Calculate the probability of each demand level (0, max_demand), given the gamma distribution parameters"""
        cdf = numpyro.distributions.Gamma(gamma_alpha, gamma_beta).cdf(
            jnp.hstack([0, jnp.arange(0.5, self.max_demand + 1.5)])
        )
        # Want integer demand, so calculate P(d<x+0.5) - P(d<x-0.5), except for 0 demand where use 0 and 0.5
        # This gives us the same results as in Fig 3 of the paper
        demand_probabilities = jnp.diff(cdf)
        # To make number of random outcomes finite, we truncate the distribution
        # Add any probability mass that is truncated back to the last demand level
        demand_probabilities = demand_probabilities.at[-1].add(
            1 - demand_probabilities.sum()
        )
        return demand_probabilities

    ### Utility functions to set up pytree for class ###
    # See https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

    def _tree_flatten(self):
        children = (
            self.cost_components,
            self.demand_probabilities,
            self.state_to_idx_mapping,
            self.states,
            self.padded_batched_states,
            self.actions,
            self.possible_random_outcomes,
            self.V_old,
            self.iteration,
        )  # arrays / dynamic values
        aux_data = {
            "max_demand": self.max_demand,
            "demand_gamma_alpha": self.demand_gamma_alpha,
            "demand_gamma_beta": self.demand_gamma_beta,
            "max_useful_life": self.max_useful_life,
            "lead_time": self.lead_time,
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
        }

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    DeMoorPerishableVIR,
    DeMoorPerishableVIR._tree_flatten,
    DeMoorPerishableVIR._tree_unflatten,
)
