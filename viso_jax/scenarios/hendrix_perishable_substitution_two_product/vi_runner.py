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

# Enable logging
log = logging.getLogger("ValueIterationRunner")

# Note that in Hendrix, oldest stock on LHS of inventory vector
# Here, for consistency with DeMoor and Mirjalili, oldest stock
# is now on the RHS of the vector


class HendrixPerishableSubstitutionTwoProductVIR(ValueIterationRunner):
    def __init__(
        self,
        max_useful_life,
        demand_poisson_mean_a,
        demand_poisson_mean_b,
        substitution_probability,
        variable_order_cost_a,
        variable_order_cost_b,
        sales_price_a,
        sales_price_b,
        max_order_quantity_a,
        max_order_quantity_b,
        batch_size,
        epsilon,
        gamma=1,
        checkpoint_frequency=0,  # Zero for no checkpoints, otherwise every x iterations
        resume_from_checkpoint=False,  # Set to checkpoint file path to restore
        use_pmap=True,
    ):
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

        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = Path("checkpoints/")
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint

        self.use_pmap = use_pmap

        self.setup()
        log.info(f"Output file directory: {Path.cwd()}")
        log.info(f"Max order quantity for product a: {self.max_order_quantity_a}")
        log.info(f"Max order quantity for product b: {self.max_order_quantity_b}")
        log.info(f"N states = {len(self.states)}")
        log.info(f"N actions = {len(self.actions)}")
        log.info(f"N random outcomes = {len(self.possible_random_outcomes)}")

    def _setup_before_states_actions_random_outcomes_created(self):
        self.max_stock_a = self.max_order_quantity_a * self.max_useful_life
        self.max_stock_b = self.max_order_quantity_b * self.max_useful_life
        self.max_demand = self.max_useful_life * (
            max(self.max_order_quantity_a, self.max_order_quantity_b) + 2
        )

    def _setup_after_states_actions_random_outcomes_created(self):

        self._calculate_expected_sales_revenue_vmap_states = jax.vmap(
            self._calculate_expected_sales_revenue
        )
        self._calculate_expected_sales_revenue_state_batch_jit = jax.jit(
            self._calculate_expected_sales_revenue_state_batch
        )
        self._calculate_expected_sales_revenue_scan_state_batches_pmap = jax.pmap(
            self._calculate_expected_sales_revenue_scan_state_batches, in_axes=(None, 0)
        )

        # jit and vmap function for calculating initial order cost for each state
        self._calculate_initial_ordering_cost_vmap_states = jax.vmap(
            self._calculate_initial_ordering_cost
        )
        self._calculate_initial_ordering_cost_state_batch_jit = jax.jit(
            self._calculate_initial_ordering_cost_state_batch
        )
        self._calculate_initial_ordering_cost_scan_state_batches_pmap = jax.pmap(
            self._calculate_initial_ordering_cost_scan_state_batches, in_axes=(None, 0)
        )

        # Pre-compute conditional probability distributions
        log.info("Calculating P(u)")
        self.pu = self._calculate_pu()
        log.info("Calculated P(u)")
        log.info("Calculating P(z)")
        self.pz = self._calculate_pz()
        log.info("Calculated P(z)")

    def generate_states(self):
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

    def create_state_to_idx_mapping(self):
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

    def generate_actions(self):
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

    def generate_possible_random_outcomes(self):
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

    def deterministic_transition_function(self, state, action, random_outcome):
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

    def get_probabilities(self, state, action, possible_random_outcomes):

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

        probs_1 = self._get_probs_ia_lt_stock_a_ib_lt_stock_b(stock_a, stock_b)
        probs_2 = self._get_probs_ia_eq_stock_a_ib_lt_stock_b(stock_a, stock_b)
        probs_3 = self._get_probs_ia_lt_stock_a_ib_gteq_stock_b(stock_a, stock_b)
        probs_4 = self._get_probs_ia_gteq_stock_a_ib_gteq_stock_b(stock_a, stock_b)

        return (probs_1 + probs_2 + probs_3 + probs_4).reshape(-1)

    def calculate_initial_values(self):
        if self.use_pmap:
            padded_batched_initial_ordering_costs = (
                self._calculate_initial_ordering_cost_scan_state_batches_pmap(
                    None, self.padded_batched_states
                )
            )
            padded_batched_expected_sales_revenue = (
                self._calculate_expected_sales_revenue_scan_state_batches_pmap(
                    None, self.padded_batched_states
                )
            )

        else:
            padded_batched_initial_ordering_costs = (
                self._calculate_initial_ordering_cost_scan_state_batches(
                    None, self.padded_batched_states
                )
            )
            padded_batched_expected_sales_revenue = (
                self._calculate_expected_sales_revenue_scan_state_batches(
                    None, self.padded_batched_states
                )
            )
        initial_ordering_costs = self._unpad(
            padded_batched_initial_ordering_costs.reshape(-1), self.n_pad
        )
        expected_sales_revenue = self._unpad(
            padded_batched_expected_sales_revenue.reshape(-1), self.n_pad
        )

        return expected_sales_revenue - initial_ordering_costs

    def check_converged(self, iteration, min_iter, V, V_old):
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

    ##### Support functions for self.generate_states() #####
    def _generate_states_single_product(self, max_order_quantity):
        possible_orders = range(0, max_order_quantity + 1)
        product_arg = [possible_orders] * self.max_useful_life
        return list(itertools.product(*product_arg))

    def _generate_two_product_state_component_idx_dict(self):
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

    ##### Support functions for self.deterministic_transition_function() #####
    def _issue_fifo(self, opening_stock, demand):
        # Oldest stock on RHS of vector, so reverse
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(self, remaining_demand, stock_element):
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    def _calculate_single_step_reward(
        self, state, action, transition_function_reward_output
    ):
        cost = jnp.dot(action, self.variable_order_costs)
        revenue = jnp.dot(transition_function_reward_output, self.sales_prices)
        return revenue - cost

    ##### Support functions for self.get_probabilities() #####

    # pu[u,y] is, from paper, Prob(u|y), conditional prob of u substitution demand given
    # y units of b in stock
    # TODO: Could try to rewrite for speed, but only runs once
    def _calculate_pu(self):
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

    # pz[z,y] is, from paper, Prob(z|y), probability of z demand from product a given
    # demand for product b is at least equal to y, number of units in stock
    # TODO: Could try to rewrite for speed, but only runs once
    def _calculate_pz(self):
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

    def _get_probs_ia_lt_stock_a_ib_lt_stock_b(self, stock_a, stock_b):
        # issued a < stock a, issued_b < stock_b
        # Therefore P(i_a, i_b) = P(d_a=ia) * P(d_b=ib)
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

    def _get_probs_ia_eq_stock_a_ib_lt_stock_b(self, stock_a, stock_b):
        # issued a >= stock a, issued_b < stock_b
        # Therefore P(i_a, i_b) = P(d_a>=ia) * P(d_b=ib)
        # No substitution
        issued_probs = jnp.zeros((self.max_stock_a + 1, self.max_stock_b + 1))

        # Demand for a higher than stock_a, but demand for b less than than stock_b
        # Top right quadrant
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

    def _get_probs_ia_lt_stock_a_ib_gteq_stock_b(self, stock_a, stock_b):
        issued_probs = jnp.zeros((self.max_stock_a + 1, self.max_stock_b + 1))

        # Demand for b higher than stock_b, so subsitution possible

        probs_issued_a = jax.lax.dynamic_slice(
            self.pz, (0, stock_b), (self.max_demand + 1, 1)
        ).reshape(-1)
        # prob_combined_demand_gteq_stock_a = probs_issued_a.dot(
        #    jnp.arange(len(probs_issued_a)) >= stock_a
        # )
        probs_issued_a_masked = probs_issued_a * (
            jnp.arange(len(probs_issued_a)) < stock_a
        )
        # If combined demand greater than or equal to stock of a, all stock of a issued
        # probs_issued_a_masked = probs_issued_a_masked.at[stock_a].add(
        #   prob_combined_demand_gteq_stock_a
        # )

        # Trim array to max_stock_a
        probs_issued_a_masked = jax.lax.dynamic_slice(
            probs_issued_a_masked, (0,), (self.max_stock_a + 1,)
        )

        issued_probs = issued_probs.at[:, stock_b].add(probs_issued_a_masked)

        return issued_probs

    def _get_probs_ia_gteq_stock_a_ib_gteq_stock_b(self, stock_a, stock_b):
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

    ##### Support functions for self._calculate_single_step_reward() #####
    def _calculate_sales_revenue_for_possible_random_outcomes(self):
        return (self.possible_random_outcomes.dot(self.sales_prices)).reshape(-1)

    def _calculate_expected_sales_revenue(self, state):
        issued_probabilities = self.get_probabilities(state, None, None)
        expected_sales_revenue = issued_probabilities.dot(
            self._calculate_sales_revenue_for_possible_random_outcomes()
        )
        return expected_sales_revenue

    def _calculate_expected_sales_revenue_state_batch(self, carry, batch_of_states):
        revenue = self._calculate_expected_sales_revenue_vmap_states(batch_of_states)
        return carry, revenue

    def _calculate_expected_sales_revenue_scan_state_batches(
        self, carry, padded_batched_states
    ):
        carry, revenue_padded = jax.lax.scan(
            self._calculate_expected_sales_revenue_state_batch_jit,
            carry,
            padded_batched_states,
        )
        return revenue_padded

    ##### Support functions for self.calculate_initial_values() #####
    def _calculate_initial_order_quantity(
        self, stock_by_age, max_order_quantity, demand_poisson_mean
    ):
        total_stock = jnp.sum(stock_by_age)
        comp1 = jnp.where(
            stock_by_age[-1] - demand_poisson_mean > 0,
            stock_by_age[-1] - demand_poisson_mean,
            0,
        )
        order_quantity = jnp.where(
            max_order_quantity - total_stock + comp1 > 0,
            max_order_quantity - total_stock + comp1,
            0,
        )
        return order_quantity

    def _calculate_initial_ordering_cost(self, state):
        state_a = jax.lax.dynamic_slice(
            state,
            (self.state_component_idx_dict["stock_a_start"],),
            (self.state_component_idx_dict["stock_a_len"],),
        )
        state_b = jax.lax.dynamic_slice(
            state,
            (self.state_component_idx_dict["stock_b_start"],),
            (self.state_component_idx_dict["stock_b_len"],),
        )
        cost_a = self.variable_order_cost_a * self._calculate_initial_order_quantity(
            state_a, self.max_order_quantity_a, self.demand_poisson_mean_a
        )
        cost_b = self.variable_order_cost_b * self._calculate_initial_order_quantity(
            state_b, self.max_order_quantity_b, self.demand_poisson_mean_b
        )
        return cost_a + cost_b

    def _calculate_initial_ordering_cost_state_batch(self, carry, batch_of_states):
        cost = self._calculate_initial_ordering_cost_vmap_states(batch_of_states)
        return carry, cost

    def _calculate_initial_ordering_cost_scan_state_batches(
        self, carry, padded_batched_states
    ):
        carry, cost_padded = jax.lax.scan(
            self._calculate_initial_ordering_cost_state_batch_jit,
            carry,
            padded_batched_states,
        )
        return cost_padded

    ##### Utility functions to set up pytree for class #####
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
            "use_pmap": self.use_pmap,
        }  # static values
        return (children, aux_data)


tree_util.register_pytree_node(
    HendrixPerishableSubstitutionTwoProductVIR,
    HendrixPerishableSubstitutionTwoProductVIR._tree_flatten,
    HendrixPerishableSubstitutionTwoProductVIR._tree_unflatten,
)
