import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import re
import logging
import math
from pathlib import Path
from typing import Union, Tuple
import chex

# Enable logging
log = logging.getLogger("ValueIterationRunner")

# TODO: This won't work properly if people change properties after initialization
# because the construction of the state and actions spaces etc depend on those values
# Could restrict to read-only access using properties, or use custom setter to force
# re-running setup if one is changed.


class ValueIterationRunner:
    def __init__(
        self,
        max_batch_size: int,
        epsilon: float,
        gamma: float,
        checkpoint_frequency: int,
        resume_from_checkpoint: Union[bool, str],
    ):
        """Base class for running value iteration

        Args:
            max_batch_size: Maximum number of states to update in parallel using vmap, will depend on GPU memory
            epsilon: Convergence criterion for value iteration
            gamma: Discount factor
            checkpoint_frequency: Frequency with which to save checkpoints, 0 for no checkpoints
            resume_from_checkpoint: If False, start from scratch; if filename, resume from checkpoint

        """
        self.max_batch_size = max_batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.checkpoint_frequency = checkpoint_frequency

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = Path("checkpoints_op/")
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint
        self._setup()

    ### Essential methods to implement in subclass ###

    # In addition to the eight methods below, self._tree_flatten
    #  should also be updated to include any subclass properties

    def generate_states(self) -> Tuple[chex.Array, Union[None, dict[str, int]]]:
        """Returns a tuple consisting of an array of all possible states and a dictionary
        that maps descriptive names of the components of the state to indices that can be
        used to extract them from an individual state.
        The array of states should be of shape (N_states, state_size)
        The dictionary is not used by methods in the base class and is therefore optional."""
        raise NotImplementedError

    def create_state_to_idx_mapping(self) -> chex.Array:
        """Returns an array that maps from a state (represented as a tuple) to its index
        in the state array
        The array should have the same number of dimensions at the state array and have
        a total number of elements equal to N_states"""
        raise NotImplementedError

    def generate_actions(self) -> Tuple[chex.Array, list[str]]:
        """Returns a tuple consisting of an array of all possible actions and a
        list of descriptive names for each action dimension (e.g. if the action consists
        of order quantities for one product, the list should contain a single string,
        and if it consists of order quantities for two products, the list should contain
        two strings)
        The array of actions should be of shape (N_actions, action_size)
        The list of descriptive names should be of length action_size"""

        raise NotImplementedError

    def generate_possible_random_outcomes(
        self,
    ) -> Tuple[chex.Array, Union[None, dict[str, int]]]:
        """Returns a tuple consisting of an array of all possible random outcomes and a dictionary
        that maps descriptive names of the components of a random outcome to indices that can be
        used to extract them from an individual random outcome.
        The random outcome contains all the information that, along with a state and action, makes
        the transition to the next state and calculation of the reward deterministic.
        The array of random outcomes should be of shape (N_random_outcome, random_outcome_size)
        The dictionary is not used by methods in the base class and is therefore optional.
        """
        raise NotImplementedError

    def deterministic_transition_function(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        random_combination: chex.Array,
    ) -> Tuple[chex.Array, float]:
        """Returns the next state and single-step reward for the provided state, action and random combination"""
        raise NotImplementedError

    def get_probabilities(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
    ) -> chex.Array:
        """Returns an array of the probabilities of each possible random outcome for the provides state-action pair
        Output array should be of shape (N_possible_random_outcomes,)"""
        raise NotImplementedError

    def calculate_initial_values(self) -> chex.Array:
        """Returns an array of the initial values for each state.
        Output array should be of shape (N_states,)"""
        raise NotImplementedError

    def check_converged(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Convergence check to determine whether to stop value iteration"""
        raise NotImplementedError

    ### End of essential methods to implement in subclass ###

    ### Optional methods to implement in subclass ###

    def _setup_before_states_actions_random_outcomes_created(self) -> None:
        """Function that is run during setup before the arrays of states, actions
        and random outcomes are created. Use this to perform any calculations or set
        any properties that are required for those arrays to be created."""
        pass

    def _setup_after_states_actions_random_outcomes_created(self) -> None:
        """Function that is run during setup after the arrays of states, actions
        and random outcomes are created. Use this to perform any calculations or set
        any properties that depend on those arrays having been created."""
        pass

    ### End of optional methods to implement in subclass ###

    def run_value_iteration(
        self,
        max_iter: int = 100,
        min_iter: int = 1,
        save_final_values: bool = True,
        save_policy: bool = True,
    ) -> dict[str, Union[pd.DataFrame, dict]]:
        """Run value iteration for a given number of iterations, or until convergence. Optionally save checkpoints of the

        Args:
            value function after each iteration, and the final value function and policy at the end of the run.
            max_iter: maximum number of iterations to run
            min_iter: minimum number of iterations to run, even if convergence is reached before this
            save_final_values: whether to save the final value function as a csv file
            save_policy: whether to save the final policy as a csv file

        Returns:
            A dictionary containing information to log and, optionally, the final value function and policy

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
            padded_batched_V = self.calculate_updated_value_scan_state_batches_pmap(
                (self.actions, self.possible_random_outcomes, self.V_old),
                self.padded_batched_states,
            )

            V = self._unpad(padded_batched_V.reshape(-1), self.n_pad)

            # Check for convergence
            if self.check_converged(i, min_iter, V, self.V_old):
                self.V_old = V
                break
            else:

                if self.checkpoint_frequency > 0 and (
                    i % self.checkpoint_frequency == 0
                ):
                    values_df = pd.DataFrame(V, index=self.state_tuples, columns=["V"])
                    values_df.to_csv(Path(self.cp_path / f"values_{i}.csv"))

                self.V_old = V

            self.iteration += 1

        to_return = {}

        # Potentially save down final values and policy
        if save_final_values:
            log.info("Saving final values")
            values_df = pd.DataFrame(V, index=self.state_tuples, columns=["V"])
            values_df.to_csv(f"values_{i}.csv")
            log.info("Final values saved")
            to_return["V"] = values_df

        if save_policy:
            log.info("Extracting and saving policy")

            best_order_actions_df = self.get_policy()

            best_order_actions_df.to_csv("best_order_actions.csv")
            log.info("Policy saved")
            to_return["policy"] = best_order_actions_df

        to_return["output_info"] = self.output_info

        return to_return

    def get_policy(self) -> pd.DataFrame:
        """Return the best policy based on the currently stored values, self.V_old,
        as a dataframe"""
        # Find that a smaller batch size required for this part
        policy_batch_size = self.batch_size // 2

        # This is slightly round-about way of constructing the table
        # but in practice seemed to help avoid GPU OOM error

        (self.padded_batched_states, self.n_pad,) = self._pad_and_batch_states_for_pmap(
            self.states, policy_batch_size, self.n_devices
        )
        best_action_idxs_padded = self._extract_policy_scan_state_batches_pmap(
            (self.actions, self.possible_random_outcomes, self.V_old),
            self.padded_batched_states,
        )
        best_action_idxs = self._unpad(best_action_idxs_padded.reshape(-1), self.n_pad)
        best_order_actions = jnp.take(self.actions, best_action_idxs, axis=0)
        best_order_actions_df = pd.DataFrame(
            best_order_actions,
            index=self.state_tuples,
            columns=self.action_labels,
        )
        return best_order_actions_df

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

        # Reshape states into shape (N_devices x N_batches x max_batch_size x state_size)
        self.padded_batched_states, self.n_pad = self._pad_and_batch_states_for_pmap(
            self.states, self.batch_size, self.n_devices
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
        log.info(f"Output file directory: {Path.cwd()}")
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
            [states, jnp.zeros((n_pad, states.shape[1]), dtype=jnp.int32)]
        )
        padded_batched_states = padded_states.reshape(
            n_devices, -1, batch_size, states.shape[1]
        )
        return padded_batched_states, n_pad

    def _unpad(self, padded_array: chex.Array, n_pad: int) -> chex.Array:
        """Remove padding from array"""
        return padded_array[:-n_pad]

    def _get_value_next_state(self, next_state: chex.Array, V_old: chex.Array) -> float:
        """Lookup the value of the next state in the value function from the previous iteration."""
        return V_old[self.state_to_idx_mapping[tuple(next_state)]]

    def _calculate_updated_state_action_value(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
        V_old: chex.Array,
    ) -> float:
        """Update the state-action value for a given state, action pair"""
        (
            next_states,
            single_step_rewards,
        ) = self._deterministic_transition_function_vmap_random_outcomes(
            state,
            action,
            possible_random_outcomes,
        )
        next_state_values = self._get_value_next_state_vmap_next_states(
            next_states, V_old
        )
        probs = self.get_probabilities(state, action, possible_random_outcomes)
        new_state_action_value = (
            single_step_rewards + self.gamma * next_state_values
        ).dot(probs)
        return new_state_action_value

    def _calculate_updated_value(
        self,
        state: chex.Array,
        actions: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
        V_old: chex.Array,
    ) -> float:
        """Update the value for a given state, by taking the max of the updated state-action
        values over all actions"""
        return jnp.max(
            self._calculate_updated_state_action_value_vmap_actions(
                state, actions, possible_random_outcomes, V_old
            )
        )

    def _calculate_updated_value_state_batch(
        self, carry, batch_of_states: chex.Array
    ) -> Tuple[Tuple[Union[int, chex.Array], chex.Array, chex.Array], chex.Array]:
        """Calculate the updated value for a batch of states"""
        V = self._calculate_updated_value_vmap_states(batch_of_states, *carry)
        return carry, V

    def _calculate_updated_value_scan_state_batches(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        padded_batched_states: chex.Array,
    ) -> chex.Array:
        """Calculate the updated value for multiple batches of states, using jax.lax.scan to loop over batches of states."""
        carry, V_padded = jax.lax.scan(
            self._calculate_updated_value_state_batch_jit,
            carry,
            padded_batched_states,
        )
        return V_padded

    def _extract_policy_one_state(
        self,
        state: chex.Array,
        actions: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
        V: chex.Array,
    ) -> int:
        """Extract the best action for a single state, by taking the argmax of the updated state-action values over all actions"""
        best_action_idx = jnp.argmax(
            self._calculate_updated_state_action_value_vmap_actions(
                state, actions, possible_random_outcomes, V
            )
        )
        return best_action_idx

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
            self.padded_batched_states,
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
            "output_info": self.output_info,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
