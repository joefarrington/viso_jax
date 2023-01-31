import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from typing import Tuple
import chex
from flax import struct
import numpyro


@struct.dataclass
class EnvState:
    stock_a: chex.Array
    stock_b: chex.Array
    step: int


# default env params are for m=2, experiment 1
@struct.dataclass
class EnvParams:
    demand_poisson_mean_a: float
    demand_poisson_mean_b: float
    substitution_probability: float
    variable_order_cost_a: float
    variable_order_cost_b: float
    sales_price_a: float
    sales_price_b: float
    max_order_quantity_a: int
    max_order_quantity_b: int
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        cls,
        demand_poisson_mean_a: float = 5.0,
        demand_poisson_mean_b: float = 5.0,
        substitution_probability: float = 0.5,
        variable_order_cost_a: float = 0.5,
        variable_order_cost_b: float = 0.5,
        sales_price_a: float = 1.0,
        sales_price_b: float = 1.0,
        max_order_quantity_a: int = 25,
        max_order_quantity_b: int = 25,
        max_steps_in_episode: int = 365,
        gamma: float = 1.0,
    ):
        return cls(
            demand_poisson_mean_a,
            demand_poisson_mean_b,
            substitution_probability,
            variable_order_cost_a,
            variable_order_cost_b,
            sales_price_a,
            sales_price_b,
            max_order_quantity_a,
            max_order_quantity_b,
            max_steps_in_episode,
            gamma,
        )


# Avoid warnings by using standard int based on whether
# double precision is enabled or not
jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


class HendrixPerishableSubstitutionTwoProductGymnax(environment.Environment):
    def __init__(self, max_useful_life: int = 2):
        super().__init__()
        self.max_useful_life = max_useful_life

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        cumulative_gamma = self.cumulative_gamma(state, params)

        # 3 random operations, create a subkey for each
        key, demand_a_key, demand_b_key, substitution_key = jax.random.split(key, 4)

        # Sample demand for each product
        demand_a = jax.random.poisson(demand_a_key, params.demand_poisson_mean_a)
        demand_b = jax.random.poisson(demand_b_key, params.demand_poisson_mean_b)

        # Sample substitutable demand and calculate issued units of A and B
        total_stock_a = jnp.sum(state.stock_a)
        total_stock_b = jnp.sum(state.stock_b)
        issued_b = jnp.min(jnp.array([demand_b, total_stock_b]))
        excess_demand_b = demand_b - issued_b
        substitution_dist = numpyro.distributions.Binomial(
            excess_demand_b, params.substitution_probability
        )
        substitutable_demand_b = substitution_dist.sample(substitution_key)
        total_demand_a = demand_a + substitutable_demand_b
        issued_a = jnp.min(jnp.array([total_demand_a, total_stock_a]))

        stock_after_issue_a = self._issue_fifo(state.stock_a, issued_a)
        stock_after_issue_b = self._issue_fifo(state.stock_b, issued_b)

        # Age stock one day and receive the order from the morning
        expiries_a = stock_after_issue_a[-1]
        expiries_b = stock_after_issue_b[-1]

        closing_stock_a = jnp.hstack(
            [action[0], stock_after_issue_a[0 : self.max_useful_life - 1]]
        )
        closing_stock_b = jnp.hstack(
            [action[1], stock_after_issue_b[0 : self.max_useful_life - 1]]
        )

        # Calculate extra variables for info
        shortage_a = jnp.max(jnp.array([demand_a - total_stock_a, 0]))
        shortage_b = jnp.max(jnp.array([demand_b - total_stock_b, 0]))
        shortage_b_inc_sub = shortage_b - jnp.max(jnp.array([issued_a - demand_a, 0]))
        holding_a = jnp.sum(stock_after_issue_a[0 : self.max_useful_life - 1])
        holding_b = jnp.sum(stock_after_issue_b[0 : self.max_useful_life - 1])

        state = EnvState(
            closing_stock_a.astype(jnp_int),
            closing_stock_b.astype(jnp_int),
            state.step + 1,
        )
        reward = self._calculate_single_step_reward(
            state, action, issued_a, issued_b, params
        )
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {
                "discount": self.discount(state, params),
                "cumulative_gamma": cumulative_gamma,
                "order_a": action[0],
                "order_b": action[1],
                "demand_a": demand_a,
                "demand_b": demand_b,
                "excess_demand_b": excess_demand_b,
                "substitutable_demand_b": substitutable_demand_b,
                "total_demand_a": total_demand_a,
                "issued_a": issued_a,
                "issued_b": issued_b,
                "shortage_a": shortage_a,
                "shortage_b": shortage_b,
                "shortage_b_inc_sub": shortage_b_inc_sub,
                "holding_a": holding_a,
                "holding_b": holding_b,
                "expiries_a": expiries_a,
                "expiries_b": expiries_b,
            },
        )

    # Start with zero inventory
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        state = EnvState(
            stock_a=jnp.zeros(self.max_useful_life, dtype=jnp_int),
            stock_b=jnp.zeros(self.max_useful_life, dtype=jnp_int),
            step=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([*state.stock_a, *state.stock_b])

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        done_steps = state.step >= params.max_steps_in_episode
        return done_steps

    def cumulative_gamma(self, state: EnvState, params: EnvParams) -> float:
        """Return cumulative discount factor"""
        return params.gamma**state.step

    def _calculate_single_step_reward(
        self,
        state: EnvState,
        action: chex.Array,
        issued_a: int,
        issued_b: int,
        params: EnvParams,
    ) -> int:
        cost = jnp.dot(
            action,
            jnp.array([params.variable_order_cost_a, params.variable_order_cost_b]),
        )
        revenue = jnp.dot(
            jnp.array([issued_a, issued_b]),
            jnp.array([params.sales_price_a, params.sales_price_b]),
        )

        return revenue - cost

    def _issue_fifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using FIFO policy"""
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: int, stock_element: int
    ) -> Tuple[int, int]:
        """Fill demand with stock of one age, representing one element in the state"""
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    @property
    def name(self) -> str:
        """Environment name."""
        return "HendrixPerishableSubstitutionTwoProduct"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        low = jnp.array([0, 0], jnp_int)
        high = jnp.array(
            [params.max_order_quantity_a, params.max_order_quantity_b], dtype=jnp_int
        )
        return spaces.Box(low, high, (2,), jnp_int)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        if params is None:
            params = self.default_params
        low = jnp.array([0] * 2 * self.max_useful_life)
        high = jnp.array(
            [params.max_order_quantity_a] * self.max_useful_life
            + [params.max_order_quantity_b] * self.max_useful_life
        )
        return spaces.Box(low, high, (2 * self.max_useful_life,), dtype=jnp_int)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "stock_a": spaces.Box(
                    0, params.max_order_quantity_a, (self.max_useful_life,), jnp_int
                ),
                "stock_b": spaces.Box(
                    0, params.max_order_quantity_b, (self.max_useful_life,), jnp_int
                ),
                "step": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @classmethod
    def calculate_kpis(cls, rollout_results: dict) -> dict[str, float]:
        """Calculate KPIs for each rollout, using the output of a rollout from RolloutWrapper"""
        service_level_a = (
            rollout_results["info"]["demand_a"] - rollout_results["info"]["shortage_a"]
        ).sum(axis=-1) / rollout_results["info"]["demand_a"].sum(axis=-1)
        service_level_b = (
            rollout_results["info"]["demand_b"] - rollout_results["info"]["shortage_b"]
        ).sum(axis=-1) / rollout_results["info"]["demand_b"].sum(axis=-1)
        service_level_b_inc_sub = (
            rollout_results["info"]["demand_b"]
            - rollout_results["info"]["shortage_b_inc_sub"]
        ).sum(axis=-1) / rollout_results["info"]["demand_b"].sum(axis=-1)

        wastage_a = rollout_results["info"]["expiries_a"].sum(
            axis=-1
        ) / rollout_results["info"]["order_a"].sum(axis=-1)
        wastage_b = rollout_results["info"]["expiries_b"].sum(
            axis=-1
        ) / rollout_results["info"]["order_b"].sum(axis=-1)

        holding_units_a = rollout_results["info"]["holding_a"].mean(axis=-1)
        holding_units_b = rollout_results["info"]["holding_b"].mean(axis=-1)

        return {
            "service_level_%_a": service_level_a * 100,
            "service_level_%_b": service_level_b * 100,
            "service_level_%_b_inc_sub": service_level_b_inc_sub * 100,
            "wastage_%_a": wastage_a * 100,
            "wastage_%_b": wastage_b * 100,
            "holding_units_a": holding_units_a,
            "holding_units_b": holding_units_b,
        }
