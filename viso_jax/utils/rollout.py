# This is based on gymnax.experimental.rollout.RolloutWrapper by Robert T. Lange
# https://github.com/RobertTLange/gymnax/blob/main/gymnax/experimental/rollout.py
# Modified from commit 7146e35

import jax
import jax.numpy as jnp
import viso_jax
import gymnax
from functools import partial
from typing import Optional, Callable
import chex


class RolloutWrapper(object):
    def __init__(
        self,
        model_forward: Callable = None,
        env_id: str = "DeMoorPerishable",
        num_env_steps: Optional[int] = None,
        env_kwargs: dict = {},
        env_params: dict = {},
        num_burnin_steps: int = 0,
        return_info: bool = False,
    ):
        """Wrapper to define batch evaluation for policy parameters."""
        self.env_id = env_id
        # Define the RL environment & network forward function
        self.env, default_env_params = viso_jax.make(self.env_id, **env_kwargs)

        if num_env_steps is None:
            self.num_env_steps = default_env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps

        # Run a total of num_burnin_steps + num_env_steps
        # The burn-in steps are run first, and not included
        # in the reported outputs
        self.num_burnin_steps = num_burnin_steps

        # None of our environments have a fixed number of steps
        # so set to match desired number of steps
        env_params["max_steps_in_episode"] = self.num_env_steps + self.num_burnin_steps
        self.env_params = default_env_params.create_env_params(**env_params)
        self.model_forward = model_forward

        # If True, include info from each step in output
        self.return_info = return_info

    @partial(jax.jit, static_argnums=(0,))
    def population_rollout(
        self, rng_eval: chex.PRNGKey, policy_params: chex.Array
    ) -> dict[str, chex.Array]:
        """Reshape parameter vector and evaluate the generation."""
        # Evaluate population of nets on gymnax task - vmap over rng & params
        pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, 0))
        return pop_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0,))
    def batch_rollout(
        self, rng_eval: chex.PRNGKey, policy_params: chex.Array
    ) -> dict[str, chex.Array]:
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None))
        return batch_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0,))
    def single_rollout(
        self, rng_input: chex.PRNGKey, policy_params: chex.Array
    ) -> dict[str, chex.Array]:
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            (
                obs,
                state,
                policy_params,
                rng,
                discounted_cum_reward,
                valid_mask,
            ) = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            if self.model_forward is not None:
                action = self.model_forward(policy_params, obs, rng_net)
            else:
                action = self.env.action_space(self.env_params).sample(rng_net)
            next_obs, next_state, reward, done, info = self.env.step(
                rng_step, state, action, self.env_params
            )

            new_discounted_cum_reward = discounted_cum_reward + jnp.where(
                state.step >= self.num_burnin_steps,
                reward
                * valid_mask
                * (
                    self.env.cumulative_gamma(state, self.env_params)
                    / self.env_params.gamma**self.num_burnin_steps
                ),
                0,
            )

            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                policy_params,
                rng,
                new_discounted_cum_reward,
                new_valid_mask,
            ]

            if self.return_info:
                y = [
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                    info,
                ]
            else:
                y = [obs, action, reward, next_obs, done]

            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_params,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps + self.num_burnin_steps,
        )

        output = {}
        start_idx = self.num_burnin_steps
        stop_idx = self.num_burnin_steps + self.num_env_steps
        if self.return_info:
            (
                obs,
                action,
                reward,
                next_obs,
                done,
                info,
            ) = scan_out
            output["info"] = {k: v[start_idx:stop_idx] for k, v in info.items()}
            output["info"]["cumulative_gamma"] = output["info"]["cumulative_gamma"] / (
                self.env_params.gamma**self.num_burnin_steps
            )  # Discounting start from end of burnin period
        else:
            obs, action, reward, next_obs, done = scan_out

        # Extract the discounted sum of rewards accumulated by agent in episode rollout
        cum_return = carry_out[-2]

        output["obs"] = obs[start_idx:stop_idx]
        output["action"] = action[start_idx:stop_idx]
        output["reward"] = reward[start_idx:stop_idx]
        output["next_obs"] = next_obs[start_idx:stop_idx]
        output["done"] = done[start_idx:stop_idx]
        output["cum_return"] = cum_return

        return output

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape
