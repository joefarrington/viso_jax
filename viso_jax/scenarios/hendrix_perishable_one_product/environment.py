import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct


@struct.dataclass
class EnvState:
    stock: chex.Array
    step: int


@struct.dataclass
class EnvParams:
    demand_poisson_mean: float
    variable_order_cost: float
    sales_price: float
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        # default env params are for m=2, experiment 1
        cls,
        demand_poisson_mean: float = 5.0,
        variable_order_cost: float = 0.5,
        sales_price: float = 1.0,
        max_steps_in_episode: int = 3650,
        gamma: float = 1.0,
    ):
        return cls(
            demand_poisson_mean,
            variable_order_cost,
            sales_price,
            max_steps_in_episode,
            gamma,
        )


# Avoid warnings by using standard int based on whether
# double precision is enabled or not
jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


class HendrixPerishableOneProductGymnax(environment.Environment):
    def __init__(self, max_useful_life: int = 2, max_order_quantity: int = 20):
        super().__init__()
        self.max_useful_life = max_useful_life
        self.max_oder_quantity = max_order_quantity

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        cumulative_gamma = self.cumulative_gamma(state, params)

        # Sample demand
        demand = jax.random.poisson(key, params.demand_poisson_mean)

        total_stock = jnp.sum(state.stock)
        issued = jnp.min(jnp.array([demand, total_stock]))
        stock_after_issue = self._issue_fifo(state.stock, issued)

        # Age stock one day and receive the order from the morning
        expiries = stock_after_issue[-1]

        closing_stock = jnp.hstack(
            [action, stock_after_issue[0 : self.max_useful_life - 1]]
        )

        # Calculate extra variables for info
        shortage = jnp.max(jnp.array([demand - total_stock, 0]))
        holding = jnp.sum(stock_after_issue[0 : self.max_useful_life - 1])

        state = EnvState(closing_stock.astype(jnp_int), state.step + 1)
        reward = self._calculate_single_step_reward(state, action, issued, params)
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {
                "discount": self.discount(state, params),
                "cumulative_gamma": cumulative_gamma,
                "demand": demand,
                "shortage": shortage,
                "holding": holding,
                "expiries": expiries,
            },
        )

    # Start with no inventory
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        state = EnvState(
            stock=jnp.zeros(self.max_useful_life, dtype=jnp_int),
            step=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([*state.stock])

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
        action: int,
        issued: int,
        params: EnvParams,
    ) -> int:
        cost = action * params.variable_order_cost
        revenue = issued * params.sales_price
        return revenue - cost

    def _issue_fifo(self, opening_stock, demand):
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(self, remaining_demand, stock_element):
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    @property
    def name(self) -> str:
        """Environment name."""
        return "HendrixPerishableOneProduct"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.max_order_quantity + 1

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(self.max_order_quantity + 1)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        if params is None:
            params = self.default_params
        low = jnp.array([0] * (self.max_useful_life))
        high = jnp.array([params.max_order_quantity] * self.max_useful_life)
        return spaces.Box(low, high, (len(low),), dtype=jnp_int)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "stock": spaces.Box(
                    0, params.max_order_quantity, (self.max_useful_life,), jnp_int
                ),
                "step": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @classmethod
    def calculate_kpis(clas, rollout_results: dict):
        """Calculate KPIs, using the output of a rollout from RolloutWrapper"""
        service_level = (
            rollout_results["info"]["demand"] - rollout_results["info"]["shortage"]
        ).sum(axis=-1) / rollout_results["info"]["demand"].sum(axis=-1)

        wastage = rollout_results["info"]["expiries"].sum(axis=-1) / rollout_results[
            "action"
        ].sum(axis=(-1))

        holding = rollout_results["info"]["holding"].mean(axis=-1)

        return {
            "service_level": service_level,
            "wastage": wastage,
            "holding": holding,
        }
