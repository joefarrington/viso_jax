import numpy as np
import jax
import jax.numpy as jnp
import itertools
import scipy
import logging
from viso_jax.value_iteration.base_vi_runner import ValueIterationRunner
from pathlib import Path
from jax import tree_util
from scipy import stats
from typing import Union
import chex

# Enable logging
log = logging.getLogger("ValueIterationRunner")

# NOTE: in Hendrix et al (2019), oldest stock on LHS of inventory vector
# Here, for consistency with De Moor and Mirjalili cases, oldest stock
# is now on the RHS of the inventory vector


class HendrixPerishableSubstitutionTwoProductVIR(ValueIterationRunner):
    def __init__(
        self,
        max_useful_life: int,
        demand_poisson_mean_a: float,
        demand_poisson_mean_b: float,
        substitution_probability: float,
        variable_order_cost_a: float,
        variable_order_cost_b: float,
        sales_price_a: float,
        sales_price_b: float,
        max_order_quantity_a: int,
        max_order_quantity_b: int,
        max_batch_size: int,
        epsilon: float,
        gamma: float = 1,
        checkpoint_frequency: int = 0,
        resume_from_checkpoint: Union[bool, str] = False,
    ):
        """Class to run value iteration for hendrix_perishable_substitution_two_product scenario

        Args:
            max_useful_life: maximum useful life of product, m >= 1
            demand_poission_mean_a: mean of Poisson distribution that models demand for product A
            demand_poission_mean_b: mean of Poisson distribution that models demand for product B
            substituion_probability: probability that excess demand for product B can satisfied by product A
            variable_order_cost_a: cost per unit of product A ordered
            variable_order_cost_b: cost per unit of product B ordered
            sales_price_a: revenue per unit of product A issued to meet demand
            sales_price_b: revenue per unit of product B issued to meet demand
            max_order_quantity_a: maximum order quantity for product A
            max_order_quantity_b: maximum order quantity for product B
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
        self.demand_poisson_mean_a = demand_poisson_mean_a
        self.demand_poisson_mean_b = demand_poisson_mean_b
        self.substitution_probability = substitution_probability
        self.variable_order_cost_a = variable_order_cost_a
        self.variable_order_cost_b = variable_order_cost_b
        self.sales_price_a = sales_price_a
        self.sales_price_b = sales_price_b
        self.variable_order_costs = jnp.array(
            [self.variable_order_cost_a, self.variable_order_cost_b]
        )
        self.sales_prices = jnp.array([self.sales_price_a, self.sales_price_b])
        self.max_order_quantity_a = max_order_quantity_a
        self.max_order_quantity_b = max_order_quantity_b

        self.max_batch_size = max_batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = Path("checkpoints/")
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint

        self._setup()
        log.info(f"Max order quantity for product a: {self.max_order_quantity_a}")
        log.info(f"Max order quantity for product b: {self.max_order_quantity_b}")

    def _setup_before_states_actions_random_outcomes_created(self):
        """Calculate the maximum stock and maximum demand to be considered"""
        self.max_stock_a = self.max_order_quantity_a * self.max_useful_life
        self.max_stock_b = self.max_order_quantity_b * self.max_useful_life
        self.max_demand = self.max_useful_life * (
            max(self.max_order_quantity_a, self.max_order_quantity_b) + 2
        )

    def _setup_after_states_actions_random_outcomes_created(self):
        """Set up functions to calculate the expected sales revenue, used
        for the initial estimate of the value function and precompute
        probability distributions for probability of excess demand for product B
        willing to accept product a given demand for product B exceeds stock of
        product B (pu) and total demand for product A (pz)"""

        self._calculate_expected_sales_revenue_vmap_states = jax.vmap(
            self._calculate_expected_sales_revenue
        )
        self._calculate_expected_sales_revenue_state_batch_jit = jax.jit(
            self._calculate_expected_sales_revenue_state_batch
        )
        self._calculate_expected_sales_revenue_scan_state_batches_pmap = jax.pmap(
            self._calculate_expected_sales_revenue_scan_state_batches, in_axes=(None, 0)
        )

        # Pre-compute conditional probability distributions
        self.pu = self._calculate_pu()
        self.pz = self._calculate_pz()

    def generate_states(self) -> tuple[list[tuple], dict[str, int]]:
        """Returns a tuple consisting of a list of all possible states as tuples and a
        dictionary that maps descriptive names of the components of the state to indices
        that can be used to extract them from an individual state"""
        possible_states_a = self._generate_states_single_product(
            self.max_order_quantity_a
        )
        possible_states_b = self._generate_states_single_product(
            self.max_order_quantity_b
        )
        combined_states = list(itertools.product(possible_states_a, possible_states_b))

        # Use this dict to access specific parts of the state
        state_component_idx_dict = self._generate_two_product_state_component_idx_dict()

        return [s[0] + s[1] for s in combined_states], state_component_idx_dict

    def create_state_to_idx_mapping(self) -> chex.Array:
        """Returns an array that maps from a state (represented as a tuple) to its index
        in the state array"""
        state_to_idx = np.zeros(
            tuple(
                [self.max_order_quantity_a + 1] * self.max_useful_life
                + [self.max_order_quantity_b + 1] * self.max_useful_life
            )
        )
        for idx, state in enumerate(self.state_tuples):
            state_to_idx[state] = idx
        state_to_idx = jnp.array(state_to_idx, dtype=jnp.int32)
        return state_to_idx

    def generate_actions(self) -> tuple[chex.Array, list[str]]:
        """Returns a tuple consisting of an array of all possible actions and a
        list of descriptive names for each action dimension"""
        actions = jnp.array(
            list(
                itertools.product(
                    range(0, self.max_order_quantity_a + 1),
                    range(0, self.max_order_quantity_b + 1),
                )
            )
        )
        action_labels = ["order_quantity_a", "order_quantity_b"]
        return actions, action_labels

    def generate_possible_random_outcomes(self) -> tuple[chex.Array, dict[str, int]]:
        """Returns a tuple consisting of an array of all possible random outcomes and a dictionary
        that maps descriptive names of the components of a random outcome to indices that can be
        used to extract them from an individual random outcome."""
        # The transition depends on the number of units issued
        # So here, we look at the possible combinations of the number
        # of units issued of products a and b
        a = range(0, self.max_stock_a + 1)
        b = range(0, self.max_stock_b + 1)

        # Use this dict to access specific elements of the random outcomes
        pro_component_idx_dict = {}
        pro_component_idx_dict["issued_a"] = 0
        pro_component_idx_dict["issued_b"] = 1

        return jnp.array(list(itertools.product(a, b))), pro_component_idx_dict

    def deterministic_transition_function(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        random_outcome: chex.Array,
    ) -> tuple[chex.Array, float]:
        """Returns the next state and single-step reward for the provided state, action and random combination"""
        opening_stock_a = state[
            self.state_component_idx_dict[
                "stock_a_start"
            ] : self.state_component_idx_dict["stock_a_stop"]
        ]
        opening_stock_b = state[
            self.state_component_idx_dict[
                "stock_b_start"
            ] : self.state_component_idx_dict["stock_b_stop"]
        ]

        issued_a = random_outcome[self.pro_component_idx_dict["issued_a"]]
        issued_b = random_outcome[self.pro_component_idx_dict["issued_b"]]

        stock_after_issue_a = self._issue_fifo(opening_stock_a, issued_a)
        stock_after_issue_b = self._issue_fifo(opening_stock_b, issued_b)

        # Age stock one day and receive the order from the morning
        closing_stock_a = jnp.hstack(
            [action[0], stock_after_issue_a[0 : self.max_useful_life - 1]]
        )
        closing_stock_b = jnp.hstack(
            [action[1], stock_after_issue_b[0 : self.max_useful_life - 1]]
        )

        next_state = jnp.concatenate([closing_stock_a, closing_stock_b], axis=-1)

        # Pass through the random outcome (units issued)
        single_step_reward = self._calculate_single_step_reward(
            state, action, random_outcome
        )
        return (
            next_state,
            single_step_reward,
        )

    def get_probabilities(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
    ) -> chex.Array:
        """Returns an array of the probabilities of each possible random outcome for the provides state-action pair"""
        # Get total stock of A and B in the current state
        stock_a = jnp.sum(
            jax.lax.dynamic_slice(
                state,
                (self.state_component_idx_dict["stock_a_start"],),
                (self.state_component_idx_dict["stock_a_len"],),
            )
        )
        stock_b = jnp.sum(
            jax.lax.dynamic_slice(
                state,
                (self.state_component_idx_dict["stock_b_start"],),
                (self.state_component_idx_dict["stock_b_len"],),
            )
        )
        # Issued a less than stock of a, issued b less than stock of b
        probs_1 = self._get_probs_ia_lt_stock_a_ib_lt_stock_b(stock_a, stock_b)
        # Issued a equal to stock of a, issued b less than stock of b
        probs_2 = self._get_probs_ia_eq_stock_a_ib_lt_stock_b(stock_a, stock_b)
        # Issued a less than stock of a, issued b equal to stock of b
        probs_3 = self._get_probs_ia_lt_stock_a_ib_eq_stock_b(stock_a, stock_b)
        # Issued a equal to stock of a, issued b equal to stock of b
        probs_4 = self._get_probs_ia_eq_stock_a_ib_eq_stock_b(stock_a, stock_b)

        return (probs_1 + probs_2 + probs_3 + probs_4).reshape(-1)

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

    ### Support functions for self.generate_states() ###
    def _generate_states_single_product(self, max_order_quantity: int) -> list[tuple]:
        """Returns possible states, as a list of tuples"""
        possible_orders = range(0, max_order_quantity + 1)
        product_arg = [possible_orders] * self.max_useful_life
        return list(itertools.product(*product_arg))

    def _generate_two_product_state_component_idx_dict(self) -> dict[str, int]:
        """Returns a dictionary that maps descriptive names of the components of a state
        to indices of the elements in the state array"""
        state_component_idx_dict = {}
        state_component_idx_dict["stock_a_start"] = 0
        state_component_idx_dict["stock_a_len"] = self.max_useful_life
        state_component_idx_dict["stock_a_stop"] = (
            state_component_idx_dict["stock_a_start"]
            + state_component_idx_dict["stock_a_len"]
        )
        state_component_idx_dict["stock_b_start"] = state_component_idx_dict[
            "stock_a_stop"
        ]
        state_component_idx_dict["stock_b_len"] = self.max_useful_life
        state_component_idx_dict["stock_b_stop"] = (
            state_component_idx_dict["stock_b_start"]
            + state_component_idx_dict["stock_b_len"]
        )
        return state_component_idx_dict

    ### Support functions for self.deterministic_transition_function() ###
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
        cost = jnp.dot(action, self.variable_order_costs)
        revenue = jnp.dot(transition_function_reward_output, self.sales_prices)
        return revenue - cost

    ### Support functions for self.get_probabilities() ###

    # TODO: Could try to rewrite for speed, but only runs once
    def _calculate_pu(self) -> np.ndarray:
        """Returns an array of the conditional probabilities of substitution demand
        given that demand for b exceeds stock of b. pu[u,y] is Prob(u|y), conditional
        probability of u substitution demand given y units of b in stock"""

        pu = np.zeros((self.max_demand + 1, self.max_stock_b + 1))
        for y in range(0, self.max_stock_b + 1):
            x = np.arange(0, self.max_demand - y)
            pu[0, y] = scipy.stats.poisson.pmf(x + y, self.demand_poisson_mean_b).dot(
                scipy.stats.binom.pmf(0, x, self.substitution_probability)
            )

            for u in range(1, self.max_demand - y):
                x = np.arange(u, self.max_demand - y)
                pu[u, y] = scipy.stats.poisson.pmf(
                    x + y, self.demand_poisson_mean_b
                ).dot(scipy.stats.binom.pmf(u, x, self.substitution_probability))

        return pu

    #
    # TODO: Could try to rewrite for speed, but only runs once
    def _calculate_pz(self) -> np.ndarray:
        """Returns an array of the conditional probabilities of total demand
        for a given that demand for b is at least equal to total demand for b.
        pz[z,y] is Prob(z|y), conditional probability of z demand from product a given
        demand for product b is at least equal to y, number of units in stock"""
        pz = np.zeros((self.max_demand + 1, self.max_stock_b + 1))
        pa = scipy.stats.poisson.pmf(
            np.arange(self.max_demand + 1), self.demand_poisson_mean_a
        )
        # No demand for a itself, and no subst demand
        pz[0, :] = pa[0] * self.pu[0, :]
        for y in range(0, self.max_stock_b + 1):
            for z in range(1, self.max_demand + 1):
                pz[z, y] = pa[np.arange(0, z + 1)].dot(
                    self.pu[z - np.arange(0, z + 1), y]
                )
        return pz

    def _get_probs_ia_lt_stock_a_ib_lt_stock_b(
        self, stock_a: chex.Array, stock_b: chex.Array
    ) -> chex.Array:
        """Returns probabilities for issued quantities of a and b given that issued a < stock a, issued_b < stock_b"""
        # P(i_a, i_b) = P(d_a=ia) * P(d_b=ib)
        # Easy cases, all demand met and no substitution
        prob_da = jax.scipy.stats.poisson.pmf(
            jnp.arange(self.max_stock_a + 1), self.demand_poisson_mean_a
        )
        prob_da_masked = prob_da * (jnp.arange(self.max_stock_a + 1) < stock_a)
        prob_db = jax.scipy.stats.poisson.pmf(
            jnp.arange(self.max_stock_b + 1), self.demand_poisson_mean_b
        )
        prob_db_masked = prob_db * (jnp.arange(self.max_stock_b + 1) < stock_b)
        issued_probs = jnp.outer(prob_da_masked, prob_db_masked)

        return issued_probs

    def _get_probs_ia_eq_stock_a_ib_lt_stock_b(
        self, stock_a: chex.Array, stock_b: chex.Array
    ) -> chex.Array:
        """Returns probabilities for issued quantities of a and b given that issued a = stock a, issued_b < stock_b"""
        # Therefore P(i_a, i_b) = P(d_a>=ia) * P(d_b=ib)
        # No substitution
        issued_probs = jnp.zeros((self.max_stock_a + 1, self.max_stock_b + 1))

        # Demand for a higher than stock_a, but demand for b less than than stock_b
        prob_da_gteq_stock_a = 1 - jax.scipy.stats.poisson.cdf(
            stock_a - 1, self.demand_poisson_mean_a
        )
        prob_db = jax.scipy.stats.poisson.pmf(
            jnp.arange(self.max_stock_b + 1), self.demand_poisson_mean_b
        )
        prob_db_masked = prob_db * (jnp.arange(self.max_stock_b + 1) < stock_b)
        probs = prob_da_gteq_stock_a * prob_db_masked
        issued_probs = issued_probs.at[stock_a, :].add(probs)

        return issued_probs

    def _get_probs_ia_lt_stock_a_ib_eq_stock_b(
        self, stock_a: chex.Array, stock_b: chex.Array
    ) -> chex.Array:
        """Returns probabilities for issued quantities of a and b given that issued a < stock a, issued_b = stock_b"""
        # Therefore total demand for a is < stock_a, demand for b >= stock_b
        issued_probs = jnp.zeros((self.max_stock_a + 1, self.max_stock_b + 1))

        # Demand for b higher than stock_b, so substitution possible

        probs_issued_a = jax.lax.dynamic_slice(
            self.pz, (0, stock_b), (self.max_demand + 1, 1)
        ).reshape(-1)

        probs_issued_a_masked = probs_issued_a * (
            jnp.arange(len(probs_issued_a)) < stock_a
        )

        # Trim array to max_stock_a
        probs_issued_a_masked = jax.lax.dynamic_slice(
            probs_issued_a_masked, (0,), (self.max_stock_a + 1,)
        )

        issued_probs = issued_probs.at[:, stock_b].add(probs_issued_a_masked)

        return issued_probs

    def _get_probs_ia_eq_stock_a_ib_eq_stock_b(
        self, stock_a: chex.Array, stock_b: chex.Array
    ) -> chex.Array:
        """Returns probabilities for issued quantities of a and b given that issued a = stock a, issued_b = stock_b"""
        # Therefore total demand for a is >= stock_a, demand for b >= stock_b
        issued_probs = jnp.zeros((self.max_stock_a + 1, self.max_stock_b + 1))

        # Demand for b higher than stock_b, so subsitution possible
        probs_issued_a = jax.lax.dynamic_slice(
            self.pz, (0, stock_b), (self.max_demand + 1, 1)
        ).reshape(-1)
        prob_combined_demand_gteq_stock_a = probs_issued_a.dot(
            jnp.arange(len(probs_issued_a)) >= stock_a
        )

        issued_probs = issued_probs.at[stock_a, stock_b].add(
            prob_combined_demand_gteq_stock_a
        )

        return issued_probs

    ### Support functions for self._calculate_single_step_reward() ###
    def _calculate_sales_revenue_for_possible_random_outcomes(self) -> chex.Array:
        """Calculate the sales revenue for each possible random outcome of demand"""
        return (self.possible_random_outcomes.dot(self.sales_prices)).reshape(-1)

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
            self.pu,
            self.pz,
            self.V_old,
            self.iteration,
            self.variable_order_costs,
            self.sales_prices,
        )  # arrays / dynamic values
        aux_data = {
            "max_useful_life": self.max_useful_life,
            "demand_poisson_mean_a": self.demand_poisson_mean_a,
            "demand_poisson_mean_b": self.demand_poisson_mean_b,
            "substitution_probability": self.substitution_probability,
            "variable_order_cost_a": self.variable_order_cost_a,
            "variable_order_cost_b": self.variable_order_cost_b,
            "sales_price_a": self.sales_price_a,
            "sales_price_b": self.sales_price_b,
            "max_order_quantity_a": self.max_order_quantity_a,
            "max_order_quantity_b": self.max_order_quantity_b,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "n_devices": self.n_devices,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "checkpoint_frequency": self.checkpoint_frequency,
            "cp_path": self.cp_path,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "max_stock_a": self.max_stock_a,
            "max_stock_b": self.max_stock_b,
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
    HendrixPerishableSubstitutionTwoProductVIR,
    HendrixPerishableSubstitutionTwoProductVIR._tree_flatten,
    HendrixPerishableSubstitutionTwoProductVIR._tree_unflatten,
)
