import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Union, Dict, List
import chex
from flax import struct
from tensorflow_probability.substrates import jax as tfp

# NOTE: As with Gymnax envs in this repo, we follow convention that stock ages left to right (freshest stock left-most in state/obs)
# This is different to the original gym env where, in line with original paper, oldest stock was right-most.


@struct.dataclass
class EnvState:
    stock: chex.Array
    weekday: int
    step: int


@struct.dataclass
class EnvParams:
    max_demand: int
    weekday_demand_poisson_mean: chex.Array
    cost_components: chex.Array
    initial_weekday: int
    initial_stock: chex.Array
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        # Default env params are for m=3, exp1 (exogenous)
        cls,
        max_demand: int = 60,
        weekday_demand_poisson_mean: List[float] = [
            37.5,
            37.3,
            39.2,
            37.8,
            40.5,
            27.2,
            28.4,
        ],
        variable_order_cost: float = 650,
        fixed_order_cost: float = 225,
        shortage_cost: float = 3250,
        wastage_cost: float = 650,
        holding_cost: float = 130,
        initial_weekday: int = 6,  # At the first observation, it is Sunday evening
        initial_stock: List[int] = [0, 0],
        max_steps_in_episode: int = 365,
        gamma: float = 1.0,
    ):
        cost_components = jnp.array(
            [
                variable_order_cost,
                fixed_order_cost,
                shortage_cost,
                wastage_cost,
                holding_cost,
            ]
        )
        assert initial_weekday in range(
            -1, 7
        ), "Initial weekday must be in range 0-6 (Mon-Sun), or -1 to sample on each reset"
        return EnvParams(
            max_demand,
            jnp.array(weekday_demand_poisson_mean),
            cost_components,
            initial_weekday,
            jnp.array(initial_stock),
            max_steps_in_episode,
            gamma,
        )


# Avoid warnings by using standard int based on whether
# double precision is enabled or not
jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


class RajendranPerishablePlateletGymnax(environment.Environment):
    # We need to pass in max_useful_life because it affects array shapes
    # We need to pass in max_order_quantity because self.num_actions depends on it
    def __init__(self, max_useful_life: int = 3, max_order_quantity: int = 60):
        super().__init__()
        self.max_useful_life = max_useful_life
        self.max_order_quantity = max_order_quantity

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        cumulative_gamma = self.cumulative_gamma(state, params)

        # Update weekday
        weekday = (state.weekday + 1) % 7

        # Receive previous order
        stock_received = jnp.clip(action, 0, self.max_order_quantity)
        opening_stock_after_delivery = jnp.hstack([stock_received, state.stock])

        # Generate demand, and truncate at max_demand
        demand = jax.random.poisson(
            key, params.weekday_demand_poisson_mean[weekday]
        ).clip(0, params.max_demand)

        # Meet demand
        stock_after_issue = self._issue_oufo(opening_stock_after_delivery, demand)

        # Compute variables required to calculate the cost
        variable_order = jnp.array(action)
        fixed_order = jnp.array(action > 1)
        shortage = jnp.max(
            jnp.array([demand - jnp.sum(opening_stock_after_delivery), 0])
        )
        expiries = stock_after_issue[-1]
        closing_stock = stock_after_issue[0 : self.max_useful_life - 1]
        holding = jnp.sum(closing_stock)
        # Same order as params.cost_components

        transition_function_reward_output = jnp.hstack(
            [variable_order, fixed_order, shortage, expiries, holding]
        )

        # Calculate reward
        reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output, params
        )

        # Update the state
        state = EnvState(closing_stock, weekday, state.step + 1)
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

    # NOTE: Starting with zero inventory here
    # This is what we did before,
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        state = EnvState(
            stock=params.initial_stock,
            weekday=params.initial_weekday,
            step=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.weekday, *state.stock])

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.step >= params.max_steps_in_episode
        return done_steps

    def cumulative_gamma(self, state: EnvState, params: EnvParams) -> float:
        """Return cumulative discount factor"""
        return params.gamma**state.step

    def _calculate_reward(
        self,
        action: int,
        closing_stock: chex.Array,
        backorders: int,
        expiries: int,
        params: EnvParams,
    ) -> int:
        costs = jnp.array(
            [
                params.fixed_order_cost,
                params.variable_order_cost,
                params.holding_cost,
                params.emergency_procurement_cost,
                params.wastage_cost,
            ]
        )
        values = jnp.array(
            [
                jnp.where(action > 0, 1, 0),
                action,
                jnp.sum(closing_stock),
                backorders,
                expiries,
            ]
        )
        return jnp.dot(costs, values)

    def _calculate_single_step_reward(
        self,
        state: EnvState,
        action: int,
        transition_function_reward_output: chex.Array,
        params: EnvParams,
    ) -> int:
        """Calculate reward for a single step transition"""
        cost = jnp.dot(transition_function_reward_output, params.cost_components)
        reward = -1 * cost
        return reward

    def _issue_oufo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using OUFO policy"""
        # Oldest stock on RHS of vector, so reverse
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
        return "RajendranPerishablePlatelet"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.max_order_quantity + 1

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Discrete(self.max_order_quantity + 1)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        # [weekday, freshest_stock, ..., oldest_stock]
        # Only need max_useful_life-1 stock entries, because we process expiries before observing and ordering
        if params is None:
            params = self.default_params
        obs_len = self.max_useful_life
        low = jnp.array([0] * obs_len)
        high = jnp.array([6] + [self.max_order_quantity] * (obs_len - 1))
        return spaces.Box(low, high, (obs_len,), dtype=jnp_int)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "weekday": spaces.Discrete(7),
                "stock": spaces.Box(
                    0,
                    self.max_order_quantity,
                    (self.max_useful_life - 1,),
                    dtype=jnp_int,
                ),
                "step": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @classmethod
    def calculate_kpis(cls, rollout_results: Dict) -> Dict[str, float]:
        """Calculate KPIs for each rollout, using the output of a rollout from RolloutWrapper"""
        service_level = (
            rollout_results["info"]["demand"] - rollout_results["info"]["shortage"]
        ).sum(axis=-1) / rollout_results["info"]["demand"].sum(axis=-1)

        wastage = rollout_results["info"]["expiries"].sum(axis=-1) / rollout_results[
            "action"
        ].sum(axis=(-1))

        holding_units = rollout_results["info"]["holding"].mean(axis=-1)
        demand = rollout_results["info"]["demand"].mean(axis=-1)
        order_q = rollout_results["action"].mean(axis=-1)
        order_made = (rollout_results["action"] > 0).mean(axis=-1)

        return {
            "service_level_%": service_level * 100,
            "wastage_%": wastage * 100,
            "holding_units": holding_units,
            "demand": demand,
            "order_quantity": order_q,
            "wastage_units": rollout_results["info"]["expiries"].mean(axis=-1),
            "shortage_units": rollout_results["info"]["shortage"].mean(axis=-1),
            "days_with_shortage_%": (rollout_results["info"]["shortage"] > 0).mean(
                axis=-1
            ),
            "order_made_%": order_made,
        }
