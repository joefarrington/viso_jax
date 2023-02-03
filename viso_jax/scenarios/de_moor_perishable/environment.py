import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict
import chex
from flax import struct
import numpyro


@struct.dataclass
class EnvState:
    in_transit: chex.Array
    stock: chex.Array
    step: int


@struct.dataclass
class EnvParams:
    max_demand: int
    demand_gamma_alpha: float
    demand_gamma_beta: float
    cost_components: chex.Array
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        # Default env params are for m=2, experiment 1
        cls,
        max_demand: int = 100,
        demand_gamma_mean: float = 4.0,
        demand_gamma_cov: float = 0.5,
        variable_order_cost: float = 3.0,
        shortage_cost: float = 5.0,
        wastage_cost: float = 7.0,
        holding_cost: float = 1.0,
        max_steps_in_episode: int = 365,
        gamma: float = 0.99,
    ):
        demand_gamma_alpha = 1 / (demand_gamma_cov**2)
        demand_gamma_beta = 1 / (demand_gamma_mean * demand_gamma_cov**2)
        cost_components = jnp.array(
            [
                variable_order_cost,
                shortage_cost,
                wastage_cost,
                holding_cost,
            ]
        )
        return cls(
            max_demand,
            demand_gamma_alpha,
            demand_gamma_beta,
            cost_components,
            max_steps_in_episode,
            gamma,
        )


# Avoid warnings by using standard int based on whether
# double precision is enabled or not
jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


class DeMoorPerishableGymnax(environment.Environment):
    def __init__(
        self,
        max_useful_life: int = 2,
        lead_time: int = 1,
        max_order_quantity: int = 10,
        issue_policy: str = "lifo",
    ):
        super().__init__()
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantity = max_order_quantity
        self.issue_policy = issue_policy

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, Dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        cumulative_gamma = self.cumulative_gamma(state, params)

        # Add ordered unit to in_transit
        in_transit = jnp.hstack([action, state.in_transit])

        # Generate demand
        demand_dist = numpyro.distributions.Gamma(
            concentration=params.demand_gamma_alpha, rate=params.demand_gamma_beta
        )
        demand = (
            jnp.round(demand_dist.sample(key=key))
            .clip(0, params.max_demand)  # Truncate at max demand
            .astype(jnp_int)
        )

        # Meet demand
        stock_after_issue = jax.lax.cond(
            self.issue_policy == "fifo",
            self._issue_fifo,
            self._issue_lifo,
            state.stock,
            demand,
        )

        # Compute variables required to calculate the
        variable_order = jnp.array(action)
        shortage = jnp.max(jnp.array([demand - jnp.sum(state.stock), 0]))
        expiries = stock_after_issue[-1]
        holding = jnp.sum(stock_after_issue[: self.max_useful_life - 1])

        # Same order as params.cost_components
        transition_function_reward_output = jnp.array(
            [variable_order, shortage, expiries, holding]
        )

        # Calculate reward
        reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output, params
        )

        # Update the state - age, stock, receive order placed at step t-(L-1)
        # This order is assumed to arrive just prior to the start of the next
        # period, and so is included in the updated state but no holding costs
        # are charged on it
        closing_stock = jnp.hstack(
            [in_transit[-1], stock_after_issue[: self.max_useful_life - 1]]
        )
        closing_in_transit = in_transit[0 : self.lead_time - 1]
        state = EnvState(closing_in_transit, closing_stock, state.step + 1)
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

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Always start with no stock and nothing in transit
        state = EnvState(
            in_transit=jnp.zeros(self.lead_time - 1, dtype=jnp_int),
            stock=jnp.zeros(self.max_useful_life, dtype=jnp_int),
            step=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([*state.in_transit, *state.stock])

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.step >= params.max_steps_in_episode
        return done_steps

    def cumulative_gamma(self, state: EnvState, params: EnvParams) -> float:
        """Return cumulative discount factor"""
        return params.gamma**state.step

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

    def _issue_fifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using FIFO policy"""
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_lifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using LIFO policy"""
        _, remaining_stock = jax.lax.scan(self._issue_one_step, demand, opening_stock)
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
        return "DeMoorPerishable"

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
        # [O, X], in_transit and in_stock, stock ages left to right
        if params is None:
            params = self.default_params
        obs_len = self.max_useful_life + self.lead_time - 1
        low = jnp.array([0] * obs_len)
        high = jnp.array([self.max_order_quantity] * obs_len)
        return spaces.Box(
            low,
            high,
            (obs_len,),
            dtype=jnp_int,
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "in_transit": spaces.Box(
                    0, self.max_order_quantity, (self.lead_time - 1,), jnp_int
                ),
                "stock": spaces.Box(
                    0, self.max_order_quantity, (self.max_useful_life,), jnp_int
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
        return {
            "service_level_%": service_level * 100,
            "wastage_%": wastage * 100,
            "holding_units": holding_units,
        }
