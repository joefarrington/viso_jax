import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import re
import logging
import math
from pathlib import Path

# Enable logging
log = logging.getLogger("ValueIterationRunner")

# TODO: This won't work properly if people change properties after initialization
# because the construction of the state and actions spaces etc depend on those values
# Could restrict to read-only access using properties, or use custom setter to force
# re-running setup if one is changed.

# TODO: Turn the comment descriptions of each function into proper docstrings
# and consider adding type hints


class ValueIterationRunner:
    def __init__(
        self,
        max_batch_size,
        epsilon,
        gamma,
        checkpoint_frequency,
        resume_from_checkpoint,
    ):
        self.max_batch_size = max_batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.checkpoint_frequency = checkpoint_frequency

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = Path("checkpoints_op/")
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint
        self.setup()
        pass

    def setup(self):
        # Setup will run vmap/jit where required
        # Manual updates tp any parameters will not be reflected in output unless
        # set-up is rerun

        log.info("Starting setup")

        log.info(f"Devices: {jax.devices()}")

        # Vmap and/or JIT methods
        self.deterministic_transition_function_vmap_random_outcomes = jax.vmap(
            self.deterministic_transition_function, in_axes=[None, None, 0]
        )
        self.get_value_next_state_vmap_next_states = jax.jit(
            jax.vmap(self.get_value_next_state, in_axes=[0, None])
        )
        self.calculate_updated_state_action_value_vmap_actions = jax.vmap(
            self.calculate_updated_state_action_value, in_axes=[None, 0, None, None]
        )

        self.calculate_updated_value_vmap_states = jax.vmap(
            self.calculate_updated_value, in_axes=[0, None, None, None]
        )
        self.calculate_updated_value_state_batch_jit = jax.jit(
            self.calculate_updated_value_state_batch
        )
        self.calculate_updated_value_scan_state_batches_pmap = jax.pmap(
            self.calculate_updated_value_scan_state_batches,
            in_axes=((None, None, None), 0),
        )

        self.extract_policy_vmap_states = jax.vmap(
            self.extract_policy, in_axes=[0, None, None, None]
        )
        self.extract_policy_state_batch_jit = jax.jit(self.extract_policy_state_batch)
        self.extract_policy_scan_state_batches_pmap = jax.pmap(
            self.extract_policy_scan_state_batches, in_axes=((None, None, None), 0)
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

        # Reshape states into devices x n_batches x batch_size x state_dim
        self.padded_batched_states, self.n_pad = self._pad_and_batch_states_for_pmap(
            self.states, self.batch_size, self.n_devices
        )

        # Get the possible actions
        self.actions, self.action_labels = self.generate_actions()

        # Generate the possible random outcomes
        # The method get_probabilities() return the probability of each occurring from a provided state
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

        # Use this to store elements to be reported in res tables as csv
        # for easy collation
        self.to_report = {}
        self.to_report["N_states"] = len(self.states)
        self.to_report["N_actions"] = len(self.actions)
        self.to_report["N_random_outcomes"] = len(self.possible_random_outcomes)

        log.info("Setup complete")
        log.info(f"Output file directory: {Path.cwd()}")
        log.info(f"N states = {self.to_report['N_states']}")
        log.info(f"N actions = {self.to_report['N_actions']}")
        log.info(f"N random outcomes = {self.to_report['N_random_outcomes']}")

    def _setup_before_states_actions_random_outcomes_created(self):
        # This method will be run during setup before states, actions
        # and random_outcomes arrays are generated. It should be used
        # to set any properites that are required for those arrays to
        # be generated
        pass

    def _setup_after_states_actions_random_outcomes_created(self):
        # This method will be run during setup after states, actions
        # and random_outcomes arrays are generated. It should be used
        # to set any properties that depend on those arrays already being
        # available
        pass

    def generate_states(self):
        # This should return two items
        # 1) The states as a list of tuples
        # 2) A dictionary that maps descriptive names of components of the state
        #    to indices. Dictionary is not used by methods in base class
        #    but can make indexing operations in subclass methods easier to read.
        #    If dict is not needed for subclass methods, can return None instead.
        pass

    def create_state_to_idx_mapping(self):
        # This should return a jnp.array with the same number of dimensions
        # as the state. It should return the index of the state in the states
        # array when the state (represented as a tuple) is used to index it
        pass

    def _pad_and_batch_states_for_pmap(self, states, batch_size, n_devices):
        n_pad = (n_devices * batch_size) - (len(states) % (n_devices * batch_size))
        padded_states = jnp.vstack(
            [states, jnp.zeros((n_pad, states.shape[1]), dtype=jnp.int32)]
        )
        padded_batched_states = padded_states.reshape(
            n_devices, -1, batch_size, states.shape[1]
        )
        return padded_batched_states, n_pad

    def _pad_and_batch_states_no_pmap(self, states, batch_size):
        n_pad = (batch_size) - (len(states) % (batch_size))
        padded_states = jnp.vstack(
            [states, jnp.zeros((n_pad, states.shape[1]), dtype=jnp.int32)]
        )
        padded_batched_states = padded_states.reshape(-1, batch_size, states.shape[1])
        return padded_batched_states, n_pad

    def _unpad(self, padded_array, n_pad):
        return padded_array[:-n_pad]

    def generate_actions(self):
        # This should return two items
        # 1) The actions as a jnp.array (dimensions n_actions x action_size)
        # 2) Labels for each dimension of the action

        pass

    def generate_possible_random_outcomes(self):
        # This should return two items
        # 1) The possible random outcomes as a jnp.array
        #    (dimensions n_possible_random_combination x random_combination_size)
        #    Each random combination represents the information that needs to be
        #    provided, along with the state and action, to make the transition
        #    deterministic.
        # 2) A dictionary that maps descriptive names of components of the random
        #    outcomes to indices. Dictionary is not used by methods in base class
        #    but can make indexing operations in subclass methods easier to read.
        #    If dict is not needed for subclass methods, can return None instead.
        pass

    def deterministic_transition_function(self, state, action, random_combination):
        # This should return two items
        # 1) The next state as a jnp.array
        # 2) The single-step reward, a single float

        pass

    def get_probabilities(self, state, action, possible_random_outcomes):
        # This should return a jnp.array of the probabilities of each random outcome
        # for a provided (state, action) pair
        pass

    def get_value_next_state(self, next_state, V_old):
        return V_old[self.state_to_idx_mapping[tuple(next_state)]]

    def calculate_updated_state_action_value(
        self, state, action, possible_random_outcomes, V_old
    ):
        (
            next_states,
            single_step_rewards,
        ) = self.deterministic_transition_function_vmap_random_outcomes(
            state,
            action,
            possible_random_outcomes,
        )
        next_state_values = self.get_value_next_state_vmap_next_states(
            next_states, V_old
        )
        probs = self.get_probabilities(state, action, possible_random_outcomes)
        new_state_action_value = (
            single_step_rewards + self.gamma * next_state_values
        ).dot(probs)
        return new_state_action_value

    def calculate_updated_value(self, state, actions, possible_random_outcomes, V_old):
        return jnp.max(
            self.calculate_updated_state_action_value_vmap_actions(
                state, actions, possible_random_outcomes, V_old
            )
        )

    def calculate_updated_value_state_batch(self, carry, batch_of_states):
        V = self.calculate_updated_value_vmap_states(batch_of_states, *carry)
        return carry, V

    def calculate_updated_value_scan_state_batches(self, carry, padded_batched_states):
        carry, V_padded = jax.lax.scan(
            self.calculate_updated_value_state_batch_jit,
            carry,
            padded_batched_states,
        )
        return V_padded

    def calculate_initial_values(self):
        # This should return a jnp.array of the same length
        # as the states array - one initial estimate of the
        # value function for each state
        pass

    def check_converged(self, iteration, min_iter, V, V_old):
        # This should return True is value iteration has converged
        # and we have completed at least min_iter iterations, and
        # False if not
        pass

    def run_value_iteration(
        self,
        max_iter=100,
        min_iter=1,
        save_final_values=True,
        save_policy=True,
    ):

        # If already run more than max_iter, raise an error
        if self.iteration > max_iter:
            raise ValueError(
                f"At least {max_iter} iterations have already been completed"
            )

        log.info(f"Starting value iteration at iteration {self.iteration}")

        for i in range(self.iteration, max_iter + 1):
            padded_batched_V = self.calculate_updated_value_scan_state_batches_pmap(
                (self.actions, self.possible_random_outcomes, self.V_old),
                self.padded_batched_states,
            )

            V = self._unpad(padded_batched_V.reshape(-1), self.n_pad)

            # Check for convergence
            if self.check_converged(i, min_iter, V, self.V_old):
                break
            else:

                if self.checkpoint_frequency > 0 and (
                    i % self.checkpoint_frequency == 0
                ):
                    values_df = pd.DataFrame(V, index=self.state_tuples, columns=["V"])
                    values_df.to_csv(Path(self.cp_path / f"values_{i}.csv"))

                self.V_old = V

            self.iteration += 1

        # Potentially save down final values and policy
        if save_final_values:
            log.info("Saving final values")
            values_df = pd.DataFrame(V, index=self.state_tuples, columns=["V"])
            values_df.to_csv(f"values_{i}.csv")
            log.info("Final values saved")

        # This is slightly round-about way of constructing the table
        # but in practice seemed to help avoid GPU OOM error
        if save_policy:
            log.info("Extracting and saving policy")

            # Find that a smaller batch size required for this part
            policy_batch_size = self.batch_size // 2

            (
                self.padded_batched_states,
                self.n_pad,
            ) = self._pad_and_batch_states_for_pmap(
                self.states, policy_batch_size, self.n_devices
            )
            best_action_idxs_padded = self.extract_policy_scan_state_batches_pmap(
                (self.actions, self.possible_random_outcomes, V),
                self.padded_batched_states,
            )
            best_action_idxs = self._unpad(
                best_action_idxs_padded.reshape(-1), self.n_pad
            )
            best_order_actions = jnp.take(self.actions, best_action_idxs, axis=0)
            best_order_actions_df = pd.DataFrame(
                best_order_actions,
                index=self.state_tuples,
                columns=self.action_labels,
            )

            best_order_actions_df.to_csv("best_order_actions.csv")
            log.info("Policy saved")

            return {
                "V": values_df,
                "policy": best_order_actions_df,
                "to_report": self.to_report,
            }

    def extract_policy(self, state, actions, possible_random_outcomes, V):
        best_action_idx = jnp.argmax(
            self.calculate_updated_state_action_value_vmap_actions(
                state, actions, possible_random_outcomes, V
            )
        )
        return best_action_idx

    def extract_policy_state_batch(self, carry, batch_of_states):
        best_action_idxs = self.extract_policy_vmap_states(batch_of_states, *carry)
        return carry, best_action_idxs

    def extract_policy_scan_state_batches(self, carry, padded_batched_states):
        carry, best_action_idxs_padded = jax.lax.scan(
            self.extract_policy_state_batch_jit,
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
            "to_report": self.to_report,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
