import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import re
import logging
import math
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional
import chex
from datetime import datetime
from viso_jax.value_iteration.base_vi_runner import ValueIterationRunner

# Enable logging
log = logging.getLogger("ValueIterationRunner")

# NOTE: This won't work properly if properties are changed after setup is run
# because the construction of the state and actions spaces etc depend on those values.
# This is currently reflected in a logged warning at the completion of setup, but in
# a future version could be handled using getter/setter methods for properties etc to
# prevent a user from making changes.


class AsyncValueIterationRunner(ValueIterationRunner):

    def __init__(
        self,
        max_batch_size: int,
        epsilon: float,
        gamma: float,
        output_directory: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 0,
        resume_from_checkpoint: Union[bool, str] = False,
        shuffle_states: bool = True,
        key: int = 0,
    ):
        """Base class for running value iteration

        Args:
            max_batch_size: Maximum number of states to update in parallel using vmap, will depend on GPU memory
            epsilon: Convergence criterion for value iteration
            gamma: Discount factor
            output_directory: Directory to save output to, if None, will create a new directory
            checkpoint_frequency: Frequency with which to save checkpoints, 0 for no checkpoints
            resume_from_checkpoint: If False, start from scratch; if filename, resume from checkpoint
            shuffle_states: Whether to shuffle states before each iteration
            key: Random seed for jax.random.PRNGKey

        """
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
        self.shuffle_states = shuffle_states
        self.key = jax.random.PRNGKey(key)
        self._setup()

    def run_value_iteration(
        self,
        max_iter: int = 100,
        min_iter: int = 1,
        extract_policy: bool = True,
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """Run value iteration for a given number of iterations, or until convergence. Optionally save checkpoints of the

        Args:
            value function after each iteration, and the final value function and policy at the end of the run.
            max_iter: maximum number of iterations to run
            min_iter: minimum number of iterations to run, even if convergence is reached before this
            extract_policy: whether to save the final policy as a csv file

        Returns:
            A dictionary containing information to log, the final value function and, optionally, the policy

        """

        # If already run more than max_iter, raise an error
        if self.iteration > max_iter:
            raise ValueError(
                f"At least {max_iter} iterations have already been completed"
            )

        # If min_iter greater than max_iter, raise an error
        if min_iter > max_iter:
            raise ValueError(f"min_iter must be less than or equal to max_iter")

        log.info(f"Starting value iteration at iteration {self.iteration}")

        for i in range(self.iteration, max_iter + 1):

            if self.shuffle_states:

                _key, self.key = jax.random.split(self.key)
                shuffled_state_idxs = jax.random.permutation(
                    _key, jnp.arange(len(self.states))
                )
                shuffled_states = self.states[shuffled_state_idxs]
                self.padded_batched_states, self.n_pad, self.padding_mask = (
                    self._pad_and_batch_states_for_pmap(
                        shuffled_states, self.batch_size, self.n_devices
                    )
                )

            padded_batched_V = self.calculate_updated_value_scan_state_batches_pmap(
                (self.actions, self.possible_random_outcomes, self.V_old),
                (self.padded_batched_states, self.padding_mask),
            )

            V = self._unpad(padded_batched_V.reshape(-1), self.n_pad)

            if self.shuffle_states:
                # Reorder V to match original state order
                # NOTE: Might need to get the indices using a loop over batches for large problems
                V = V.at[shuffled_state_idxs].set(V)
                pass

            # Check for convergence
            if self.check_converged(i, min_iter, V, self.V_old):
                break
            else:

                if self.checkpoint_frequency > 0 and (
                    i % self.checkpoint_frequency == 0
                ):
                    values_df = pd.DataFrame(V, index=self.state_tuples, columns=["V"])
                    values_df.to_csv(self.cp_path / f"values_{i}.csv")

                self.V_old = V

            self.iteration += 1

        to_return = {}

        # Put final values into pd.DataFrame to return
        values_df = pd.DataFrame(np.array(V), index=self.state_tuples, columns=["V"])
        to_return[f"V"] = values_df

        # If extract_policy is True, extract policy with one-step ahead search
        # to return in output
        if extract_policy:
            log.info("Extracting final policy")
            best_order_actions_df = self.get_policy(V)
            log.info("Final policy extracted")
            to_return["policy"] = best_order_actions_df

        to_return["output_info"] = self.output_info

        return to_return

    def _setup(self) -> None:
        """Run setup to create arrays of states, actions and random outcomes;
        pmap, vmap and jit methods where required, and load checkpoint if provided"""

        # Manual updates to any parameters will not be reflected in output unless
        # set-up is rerun

        log.info("Starting setup")

        log.info(f"Devices: {jax.devices()}")

        # Vmap and/or JIT methods
        self._deterministic_transition_function_vmap_random_outcomes = jax.vmap(
            self.deterministic_transition_function, in_axes=[None, None, 0]
        )
        self._get_value_next_state_vmap_next_states = jax.jit(
            jax.vmap(self._get_value_next_state, in_axes=[0, None])
        )
        self._calculate_updated_state_action_value_vmap_actions = jax.vmap(
            self._calculate_updated_state_action_value, in_axes=[None, 0, None, None]
        )

        self._calculate_updated_value_vmap_states = jax.vmap(
            self._calculate_updated_value, in_axes=[0, None, None, None]
        )
        self._calculate_updated_value_state_batch_jit = jax.jit(
            self._calculate_updated_value_state_batch
        )
        self.calculate_updated_value_scan_state_batches_pmap = jax.pmap(
            self._calculate_updated_value_scan_state_batches,
            in_axes=((None, None, None), 0),
        )

        self._extract_policy_vmap_states = jax.vmap(
            self._extract_policy_one_state, in_axes=[0, None, None, None]
        )
        self._extract_policy_state_batch_jit = jax.jit(self._extract_policy_state_batch)
        self._extract_policy_scan_state_batches_pmap = jax.pmap(
            self._extract_policy_scan_state_batches, in_axes=((None, None, None), 0)
        )

        # New for async
        self._get_state_idx_vmap_states = jax.vmap(self._get_state_idx)

        # Hook for custom setup in subclasses
        self._setup_before_states_actions_random_outcomes_created()

        # Get the states as tuples initially so they can be used to get state_to_idx_mapping
        # before being converted to a jax.numpy array
        self.state_tuples, self.state_component_idx_dict = self.generate_states()
        self.state_to_idx_mapping = self.create_state_to_idx_mapping()
        self.states = jnp.array(np.array(self.state_tuples))

        self.n_devices = len(jax.devices())
        self.batch_size = min(
            self.max_batch_size, math.ceil(len(self.states) / self.n_devices)
        )

        if not self.shuffle_states:
            # Reshape states into shape (N_devices x N_batches x max_batch_size x state_size)
            self.padded_batched_states, self.n_pad, self.padding_mask = (
                self._pad_and_batch_states_for_pmap(
                    self.states, self.batch_size, self.n_devices
                )
            )

        # Get the possible actions
        self.actions, self.action_labels = self.generate_actions()

        # Generate the possible random outcomes
        (
            self.possible_random_outcomes,
            self.pro_component_idx_dict,
        ) = self.generate_possible_random_outcomes()

        # Hook for custom setup in subclasses
        self._setup_after_states_actions_random_outcomes_created()

        if not self.resume_from_checkpoint:
            # Initialise the value function
            self.V_old = self.calculate_initial_values()
            self.iteration = 1  # start at iteration 1
        else:
            # Allow basic loading of checkpoint for resumption
            log.info(
                f"Loading initial values from checkpoint file: {self.resume_from_checkpoint}"
            )
            loaded_cp_iteration = int(
                re.search("values_(.*).csv", self.resume_from_checkpoint).group(1)
            )
            log.info(
                f"Checkpoint was iteration {loaded_cp_iteration}, so start at iteration {loaded_cp_iteration+1}"
            )
            values_df_loaded = pd.read_csv(
                Path(self.resume_from_checkpoint), index_col=0
            )
            self.V_old = jnp.array(values_df_loaded.iloc[:, 0])
            log.info("Checkpoint loaded")

            self.iteration = (
                loaded_cp_iteration + 1
            )  # first iteration will be after last one in checkpoint

        # Use this to store elements to be reported in res tables
        # for easy collation
        self.output_info = {}
        self.output_info["set_sizes"] = {}
        self.output_info["set_sizes"]["N_states"] = len(self.states)
        self.output_info["set_sizes"]["N_actions"] = len(self.actions)
        self.output_info["set_sizes"]["N_random_outcomes"] = len(
            self.possible_random_outcomes
        )

        # Log some basic information about the problem
        log.info("Setup complete")
        log.warning(
            "Changes to properties of the class after setup will not necessarily be reflected in the output and may lead to errors. To run an experiment with different settings, create a new value iteration runner"
        )
        log.info(f"Output file directory: {self.output_directory}")
        log.info(f"N states = {self.output_info['set_sizes']['N_states']}")
        log.info(f"N actions = {self.output_info['set_sizes']['N_actions']}")
        log.info(
            f"N random outcomes = {self.output_info['set_sizes']['N_random_outcomes']}"
        )

    def _pad_and_batch_states_for_pmap(
        self, states: chex.Array, batch_size: int, n_devices: int
    ) -> Tuple[chex.Array, int]:
        """Pad states and reshape to (N_devices x N_batches x max_batch_size x state_size) to support
        pmap over devices, and using jax.lax.scan to loop over batches of states."""
        n_pad = (n_devices * batch_size) - (len(states) % (n_devices * batch_size))
        padded_states = jnp.vstack(
            [states, jnp.zeros((n_pad, states.shape[-1]), dtype=jnp.int32)]
        )
        padded_batched_states = padded_states.reshape(
            n_devices,
            -1,
            batch_size,
            states.shape[-1],
        )
        padding_mask = jnp.array([False] * len(states) + [True] * n_pad).reshape(
            n_devices, -1, batch_size
        )
        return padded_batched_states, n_pad, padding_mask

    def _get_state_idx(self, state: chex.Array) -> float:
        """Lookup the value of the next state in the value function from the previous iteration."""
        return self.state_to_idx_mapping[tuple(state)]

    def _calculate_updated_value_state_batch(
        self, carry, padded_batched_state_input: Tuple[chex.Array, chex.Array]
    ) -> Tuple[Tuple[Union[int, chex.Array], chex.Array, chex.Array], chex.Array]:
        """Calculate the updated value for a batch of states"""
        batch_of_states, batch_padding_mask = padded_batched_state_input
        V_batch = self._calculate_updated_value_vmap_states(batch_of_states, *carry)

        # New for async - update V in carry to reflect these new updated values for the batch
        c1, c2, V = carry
        update_idxs = self._get_state_idx_vmap_states(batch_of_states)
        V = V.at[update_idxs].set(
            jnp.where(batch_padding_mask, V[update_idxs], V_batch)
        )  # Only update V for non-padding states
        carry = (c1, c2, V)
        return carry, V_batch

    def _calculate_updated_value_scan_state_batches(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        padded_batched_state_input: chex.Array,
    ) -> chex.Array:
        """Calculate the updated value for multiple batches of states, using jax.lax.scan to loop over batches of states."""
        carry, V_padded = jax.lax.scan(
            self._calculate_updated_value_state_batch_jit,
            carry,
            padded_batched_state_input,
        )
        return V_padded

    def get_policy(self, V) -> pd.DataFrame:
        """Return the best policy, based on the input values V,
        as a dataframe"""
        # Find that a smaller batch size required for this part
        policy_batch_size = self.batch_size // 2

        # This is slightly round-about way of constructing the table
        # but in practice seemed to help avoid GPU OOM error

        (self.padded_batched_states, self.n_pad, self.padding_mask) = (
            self._pad_and_batch_states_for_pmap(
                self.states, policy_batch_size, self.n_devices
            )
        )
        best_action_idxs_padded = self._extract_policy_scan_state_batches_pmap(
            (self.actions, self.possible_random_outcomes, V),
            self.padded_batched_states,
        )
        best_action_idxs = self._unpad(best_action_idxs_padded.reshape(-1), self.n_pad)
        best_order_actions = jnp.take(self.actions, best_action_idxs, axis=0)
        best_order_actions_df = pd.DataFrame(
            np.array(best_order_actions),
            index=self.state_tuples,
            columns=self.action_labels,
        )
        return best_order_actions_df

    def _extract_policy_state_batch(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        batch_of_states: chex.Array,
    ) -> chex.Array:
        """Extract the best action for a batch of states"""
        best_action_idxs = self._extract_policy_vmap_states(batch_of_states, *carry)
        return carry, best_action_idxs

    def _extract_policy_scan_state_batches(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        padded_batched_states: chex.Array,
    ) -> chex.Array:
        """Extract the best action for multiple batches of states, using jax.lax.scan to loop over batches of states."""
        carry, best_action_idxs_padded = jax.lax.scan(
            self._extract_policy_state_batch_jit,
            carry,
            padded_batched_states,
        )
        return best_action_idxs_padded

    ##### Utility functions to set up pytree for class #####
    # See https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

    def _tree_flatten(self):
        # This method should return two items as described in the documentation linked above
        # 1) A tuple containing any arrays/dynamic values that are properties of the class
        # 2) A dictionary containing any static values that are properties of the class
        children = (
            self.state_to_idx_mapping,
            self.states,
            self.padded_batched_state_idxs,
            self.actions,
            self.possible_random_outcomes,
            self.V_old,
            self.iteration,
        )  # arrays / dynamic values
        aux_data = {
            "max_batch_size": self.max_batch_size,
            "batch_size": self.batch_size,
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
            "padding_mask": self.padding_mask,
            "output_info": self.output_info,
        }  # static values
        return (children, aux_data)
