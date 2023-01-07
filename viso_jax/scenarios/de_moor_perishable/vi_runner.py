import numpy as np
import jax
import jax.numpy as jnp
import itertools
import logging
import numpyro
from viso_jax.value_iteration.base_vi_runner import ValueIterationRunner
from pathlib import Path
from jax import tree_util

# Enable logging
log = logging.getLogger("ValueIterationRunner")


class DeMoorPerishableVIR(ValueIterationRunner):
    def __init__(
        self,
        max_demand,
        demand_gamma_mean,
        demand_gamma_cov,
        max_useful_life,
        lead_time,
        max_order_quantity,
        variable_order_cost,
        shortage_cost,
        wastage_cost,
        holding_cost,
        issue_policy="fifo",
        batch_size=1000,
        epsilon=1e-4,
        gamma=1,
        checkpoint_frequency=1,  # Zero for no checkpoints, otherwise every x iterations
        resume_from_checkpoint=False,  # Set to checkpoint file path to restore
        use_pmap=True,
    ):

        self.max_demand = max_demand

        # Paper provides mean, CoV for gamma dist, but numpyro distribution expects alpha (concentration) and beta (rate)
        (
            self.demand_gamma_alpha,
            self.demand_gamma_beta,
        ) = self._convert_gamma_parameters(demand_gamma_mean, demand_gamma_cov)
        self.demand_probabilities = self._calculate_demand_probabilities(
            self.demand_gamma_alpha, self.demand_gamma_beta
        )

        self.max_useful_life = max_useful_life

        assert lead_time >= 1, "Lead time must be greater than or equal to 1"

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

        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = Path("checkpoints_op/")
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint

        self.use_pmap = use_pmap

        self.setup()
        log.info(f"Output file directory: {Path.cwd()}")
        log.info(f"N states = {len(self.states)}")
        log.info(f"N actions = {len(self.actions)}")
        log.info(f"N random outcomes = {len(self.possible_random_outcomes)}")

    def generate_states(self):

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

    def create_state_to_idx_mapping(self):
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

    def generate_actions(self):
        actions = jnp.arange(0, self.max_order_quantity + 1)
        action_labels = ["order_quantity"]
        return actions, action_labels

    def generate_possible_random_outcomes(self):
        possible_random_outcomes = jnp.arange(0, self.max_demand + 1).reshape(-1, 1)
        pro_component_idx_dict = {}
        pro_component_idx_dict["demand"] = 0

        return possible_random_outcomes, pro_component_idx_dict

    def deterministic_transition_function(self, state, action, random_outcome):
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

    def get_probabilities(self, state, action=None, possible_random_combinations=None):
        # Same probabilities for every (state, action) pair
        # so calculate once during setup
        return self.demand_probabilities

    def calculate_initial_values(self):
        return jnp.zeros(len(self.states))

    def check_converged(self, iteration, min_iter, V, V_old):
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

    ##### Supporting function for deterministic transition function ####
    def _issue_fifo(self, opening_stock, demand):
        # Oldest stock on RHS of vector, so reverse
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_lifo(self, opening_stock, demand):
        # Freshest stock on LHS of vector
        _, remaining_stock = jax.lax.scan(self._issue_one_step, demand, opening_stock)
        return remaining_stock

    def _issue_one_step(self, remaining_demand, stock_element):
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    def _calculate_single_step_reward(
        self, state, action, transition_function_reward_output
    ):
        # Minus one to reflect the fact that they are costs
        cost = jnp.dot(transition_function_reward_output, self.cost_components)
        reward = -1 * cost
        return reward

    ##### Supporting functions for calculating demand probabilities
    def _convert_gamma_parameters(self, mean, cov):
        alpha = 1 / (cov**2)
        beta = 1 / (mean * cov**2)
        return alpha, beta

    def _calculate_demand_probabilities(self, gamma_alpha, gamma_beta):
        cdf = numpyro.distributions.Gamma(gamma_alpha, gamma_beta).cdf(
            jnp.hstack([0, jnp.arange(0.5, self.max_demand + 1.5)])
        )
        # Want integer demand, so calculate P(d<x+0.5) - P(d<x-0.5), except for 0 demand where use 0 and 0.5
        # This gives us the same results as in Fig 3 of the paper
        return jnp.diff(cdf)

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
            "issue_policy": self.issue_policy,
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
            "n_pad": self.n_pad,
            "use_pmap": self.use_pmap,
        }

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    DeMoorPerishableVIR,
    DeMoorPerishableVIR._tree_flatten,
    DeMoorPerishableVIR._tree_unflatten,
)
