import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from tensorflow_probability.substrates import jax as tfp


@struct.dataclass
class EnvState:
    weekday: int
    stock: chex.Array
    step: int


@struct.dataclass
class EnvParams:
    max_demand: int
    weekday_demand_negbin_n: chex.Array
    weekday_demand_negbin_delta: chex.Array
    weekday_demand_negbin_p: chex.Array
    shelf_life_at_arrival_distribution_c_0: chex.Array
    shelf_life_at_arrival_distribution_c_1: chex.Array
    cost_components: chex.Array
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        # Default env params are for m=3, endo experiment 1
        cls,
        max_demand: int = 20,
        weekday_demand_negbin_n: chex.Array = [3.5, 11.0, 7.2, 11.1, 5.9, 5.5, 2.2],
        weekday_demand_negbin_delta: chex.Array = [5.7, 6.9, 6.5, 6.2, 5.8, 3.3, 3.4],
        shelf_life_at_arrival_distribution_c_0: chex.Array = [1.0, 0.5],
        shelf_life_at_arrival_distribution_c_1: chex.Array = [0.4, 0.8],
        variable_order_cost: float = 0.0,
        fixed_order_cost: float = 10.0,
        shortage_cost: float = 20.0,
        wastage_cost: float = 5.0,
        holding_cost: float = 1.0,
        max_steps_in_episode: int = 3650,
        gamma: float = 0.95,
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
        weekday_demand_param_p = jnp.array(weekday_demand_negbin_n) / (
            jnp.array(weekday_demand_negbin_delta) + jnp.array(weekday_demand_negbin_n)
        )
        return EnvParams(
            max_demand,
            jnp.array(weekday_demand_negbin_n),
            jnp.array(weekday_demand_negbin_delta),
            weekday_demand_param_p,
            jnp.array(shelf_life_at_arrival_distribution_c_0),
            jnp.array(shelf_life_at_arrival_distribution_c_1),
            cost_components,
            max_steps_in_episode,
            gamma,
        )


# Avoid warnings by using standard int based on whether
# double precision is enabled or not
jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


class MirjaliliPerishablePlateletGymnax(environment.Environment):
    def __init__(self, max_useful_life: int = 3, max_order_quantity: int = 20):
        super().__init__()
        self.max_useful_life = max_useful_life
        self.max_order_quantity = max_order_quantity

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        cumulative_gamma = self.cumulative_gamma(state, params)

        key, arrival_key, demand_key = jax.random.split(key, 3)

        # Receive previous order, the age dist of which can depend on order quantity
        max_stock_received = self._sample_units_received(arrival_key, action, params)
        opening_stock_after_delivery = jnp.hstack([0, state.stock]) + max_stock_received
        # Clip so no element is greater that max_order_quantity
        opening_stock_after_delivery = opening_stock_after_delivery.clip(
            0, self.max_order_quantity
        )
        # Calculate this to report in info - units accepted for delivery
        new_stock_accepted = opening_stock_after_delivery - jnp.hstack([0, state.stock])

        # Generate demand
        n = jax.lax.dynamic_slice(
            params.weekday_demand_negbin_n, (state.weekday,), (1,)
        )[0]
        p = jax.lax.dynamic_slice(
            params.weekday_demand_negbin_p, (state.weekday,), (1,)
        )[0]
        # tfd NegBin is distribution over successes until observe `total_count` failures,
        # versus MM thesis where distribtion over failures until certain number of successes
        # Therefore use 1-p for prob (prob of failure is 1 - prob of success)
        demand_dist = tfp.distributions.NegativeBinomial(total_count=n, probs=(1 - p))
        # Demand distribution is truncated at max_demand, so clip
        demand = (
            demand_dist.sample(seed=demand_key)
            .clip(0, params.max_demand)
            .astype(jnp.int32)
        )

        # Meet demand
        stock_after_issue = self._issue_oufo(opening_stock_after_delivery, demand)

        # Compute variables required to calculate the cost
        variable_order = jnp.array(action)
        fixed_order = jnp.array(action > 1)
        shortage = jnp.max(
            jnp.array([demand - jnp.sum(opening_stock_after_delivery), 0])
        )
        expiries = stock_after_issue[-1]
        holding = jnp.sum(stock_after_issue[: self.max_useful_life - 1])
        # Same order as params.cost_components
        transition_function_reward_output = jnp.hstack(
            [variable_order, fixed_order, shortage, expiries, holding]
        )

        # Calculate reward
        reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output, params
        )

        # Update the state - move to next weekday, age stock
        next_weekday = (state.weekday + 1) % 7
        closing_stock = stock_after_issue[0 : self.max_useful_life - 1]
        state = EnvState(next_weekday, closing_stock, state.step + 1)
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {
                "discount": self.discount(state, params),
                "cumulative_gamma": cumulative_gamma,
                "max_stock_received": max_stock_received,
                "new_stock_accepted": new_stock_accepted,
                "demand": demand,
                "shortage": shortage,
                "holding": holding,
                "expiries": expiries,
            },
        )

    # Start with zero inventory
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Always start on Monday morning with no stock
        state = EnvState(
            weekday=0,
            stock=jnp.zeros(self.max_useful_life - 1, dtype=jnp.int32),
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

    def _sample_units_received(self, key: chex.PRNGKey, action: int, params: EnvParams):
        # Assume logit for useful_life=1 is 0, concatenate with logits
        # for other ages using provided coefficients and order size action
        multinomial_logits = self._get_multinomial_logits(action, params)
        dist = tfp.distributions.Multinomial(
            logits=multinomial_logits, total_count=action.astype(jnp.float32)
        )
        sample = dist.sample(seed=key).astype(jnp.int32)
        return sample

    def _calculate_single_step_reward(
        self,
        state: EnvState,
        action: int,
        transition_function_reward_output: chex.Array,
        params: EnvParams,
    ) -> int:
        cost = jnp.dot(transition_function_reward_output, params.cost_components)
        reward = -1 * cost
        return reward

    def _issue_oufo(self, opening_stock, demand):
        # Oldest stock on RHS of vector, so reverse
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(self, remaining_demand, stock_element):
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    def _get_multinomial_logits(self, action, params):
        c_0 = params.shelf_life_at_arrival_distribution_c_0
        c_1 = params.shelf_life_at_arrival_distribution_c_1
        # Assume logit for useful_life=1 is 0, concatenate with logits
        # for other ages using provided coefficients and order size action

        # Parameters are provided in ascending remaining shelf life
        # So reverse to match ordering of stock array which is in
        # descending order of remaining useful life so that oldest
        # units are on the RHS
        return jnp.hstack([0, c_0 + (c_1 * action)])[::-1]

    @property
    def name(self) -> str:
        """Environment name."""
        return "MirjaliliPerishablePlatelet"

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
        # [weekday, oldest_stock, ..., freshest_stock]
        # Only need max_useful_life-1 stock entries, because we process expiries before observing and ordering
        if params is None:
            params = self.default_params
        obs_len = self.max_useful_life
        low = jnp.array([0] * obs_len)
        high = jnp.array([6] + [self.max_order_quantity] * (obs_len - 1))
        return spaces.Box(low, high, (obs_len,), dtype=jnp.int32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "weekday": spaces.Discrete(7),
                "stock": spaces.Box(
                    0, self.max_order_quantity, (self.max_useful_life - 1,), jnp.int32
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

        holding_units = rollout_results["info"]["holding"].mean(axis=-1)

        return {
            "service_level_%": service_level * 100,
            "wastage_%": wastage * 100,
            "holding_units": holding_units,
        }
