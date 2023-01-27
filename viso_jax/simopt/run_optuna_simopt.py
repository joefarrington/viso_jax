import hydra
from omegaconf import OmegaConf
import logging
from datetime import datetime
import pandas as pd
import optuna
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
from viso_jax.evaluation.evaluate_policy import create_evaluation_output_summary
from viso_jax.utils.yaml import to_yaml, from_yaml

# Enable logging
log = logging.getLogger(__name__)


def param_search_bounds_from_config(cfg, policy):
    """Create a search bound dict from the config file"""
    # Specify search bounds for each parameter
    if cfg.param_search.search_bounds.all_params is None:
        try:
            search_bounds = {
                p: {
                    "low": cfg.param_search.search_bounds[p]["low"],
                    "high": cfg.param_search.search_bounds[p]["high"],
                }
                for p in policy.param_names.flat
            }
        except:
            raise ValueError(
                "Ranges for each parameter must be specified if not using same range for all parameters"
            )
    # Otherwise, use the same range for all parameters
    else:
        search_bounds = {
            p: {
                "low": cfg.param_search.search_bounds.all_params.low,
                "high": cfg.param_search.search_bounds.all_params.high,
            }
            for p in policy.param_names.flat
        }
    return search_bounds


def grid_search_space_from_config(search_bounds, policy):
    """Create a grid search space from the search bounds"""
    search_space = {
        p: list(
            range(
                search_bounds[p]["low"],
                search_bounds[p]["high"] + 1,
            )
        )
        for p in policy.param_names.flat
    }
    return search_space


# Grid sampler is not straightforwardly compatible with the ask/tell
# interface so we need to treat it a bit differently to avoid
# to avoid duplication and handle RuntimeError
# https://github.com/optuna/optuna/issues/4121
def simopt_grid_sampler(cfg, policy, rollout_wrapper, rng_eval):
    search_bounds = param_search_bounds_from_config(cfg, policy)
    search_space = grid_search_space_from_config(search_bounds, policy)
    sampler = hydra.utils.instantiate(
        cfg.param_search.sampler, search_space=search_space, seed=cfg.param_search.seed
    )
    study = optuna.create_study(sampler=sampler, direction="maximize")

    i = 1
    while (
        len(sampler._get_unvisited_grid_ids(study)) > 0
        and i <= cfg.param_search.max_iterations
    ):
        trials = []
        policy_params = []
        log.info(f"Round {i}: Suggesting parameters")
        num_parallel_trials = min(
            len(sampler._get_unvisited_grid_ids(study)),
            cfg.param_search.max_parallel_trials,
        )
        for j in range(num_parallel_trials):
            trial = study.ask()
            trials.append(trial)
            policy_params.append(
                np.array(
                    [
                        trial.suggest_int(
                            f"{p}",
                            search_bounds[p]["low"],
                            search_bounds[p]["high"],
                        )
                        for p in policy.param_names.flat
                    ]
                ).reshape(policy.params_shape)
            )
        policy_params = jnp.array(policy_params)
        log.info(f"Round {i}: Simulating rollouts")
        rollout_results = rollout_wrapper.population_rollout(rng_eval, policy_params)
        log.info(f"Round {i}: Processing results")
        objectives = rollout_results["cum_return"].mean(axis=(-2, -1))

        for idx in range(num_parallel_trials):
            trials[idx].set_user_attr(
                "daily_undiscounted_reward_mean",
                rollout_results["reward"][idx].mean(axis=(-2, -1)),
            )
            trials[idx].set_user_attr(
                "daily_undiscounted_reward_std",
                rollout_results["reward"][idx].mean(axis=-1).std(),
            )
            trials[idx].set_user_attr(
                "cumulative_discounted_return_std",
                rollout_results["cum_return"][idx].std(),
            )
            try:
                study.tell(trials[idx], objectives[idx])
            except RuntimeError:
                break
        # Override rollout_results; helps to avoid GPU OOM error on larger problems
        rollout_results = 0
        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
        )
        i += 1
    return study


def simopt_other_sampler(cfg, policy, rollout_wrapper, rng_eval):
    search_bounds = param_search_bounds_from_config(cfg, policy)
    sampler = hydra.utils.instantiate(
        cfg.param_search.sampler, seed=cfg.param_search.seed
    )
    study = optuna.create_study(sampler=sampler, direction="maximize")

    # Counter for early stopping
    es_counter = 0

    for i in range(1, cfg.param_search.max_iterations + 1):
        trials = []
        policy_params = []
        log.info(f"Round {i}: Suggesting parameters")
        for j in range(cfg.param_search.max_parallel_trials):
            trial = study.ask()
            trials.append(trial)
            policy_params.append(
                np.array(
                    [
                        trial.suggest_int(
                            f"{p}",
                            search_bounds[p]["low"],
                            search_bounds[p]["high"],
                        )
                        for p in policy.param_names.flat
                    ]
                ).reshape(policy.params_shape)
            )
        policy_params = jnp.array(policy_params)
        log.info(f"Round {i}: Simulating rollouts")
        rollout_results = rollout_wrapper.population_rollout(rng_eval, policy_params)
        log.info(f"Round {i}: Processing results")
        objectives = rollout_results["cum_return"].mean(axis=(-2, -1))

        for idx in range(cfg.param_search.max_parallel_trials):
            trials[idx].set_user_attr(
                "daily_undiscounted_reward_mean",
                rollout_results["reward"][idx].mean(axis=(-2, -1)),
            )
            trials[idx].set_user_attr(
                "daily_undiscounted_reward_std",
                rollout_results["reward"][idx].mean(axis=-1).std(),
            )
            trials[idx].set_user_attr(
                "cumulative_discounted_return_std",
                rollout_results["cum_return"][idx].std(),
            )
            study.tell(trials[idx], objectives[idx])

        # Override rollout_results; helps to avoid GPU OOM error on larger problems
        rollout_results = 0
        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
        )
        # Perform early stopping starting on the second round
        if i > 1:
            if study.best_params == best_params_last_round:
                es_counter += 1
            else:
                es_counter = 0
        if es_counter >= cfg.param_search.early_stopping_rounds:
            log.info(
                f"No change in best parameters for {cfg.param_search.early_stopping_rounds} rounds. Stopping search."
            )
            break
        best_params_last_round = study.best_params
    return study


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    start_time = datetime.now()

    output_info = {}
    policy = hydra.utils.instantiate(cfg.policy)
    rollout_wrapper = hydra.utils.instantiate(
        cfg.rollout_wrapper, model_forward=policy.forward
    )
    rng_eval = jax.random.split(
        jax.random.PRNGKey(cfg.param_search.seed), cfg.param_search.num_rollouts
    )

    if cfg.param_search.sampler._target_ == "optuna.samplers.GridSampler":
        study = simopt_grid_sampler(cfg, policy, rollout_wrapper, rng_eval)
    else:
        study = simopt_other_sampler(cfg, policy, rollout_wrapper, rng_eval)

    best_trial_idx = study.best_trial.number
    trials_df = study.trials_dataframe()
    trials_df.to_csv("trials.csv")

    best_trial_df = trials_df.loc[[best_trial_idx]]
    best_trial_df.to_csv("best_trial.csv")

    simopt_complete_time = datetime.now()
    simopt_run_time = simopt_complete_time - start_time
    log.info(
        f"Simulation optimization complete. Duration: {(simopt_run_time).total_seconds():.2f}s.  Best params: {study.best_params}, mean return: {study.best_value:.4f}"
    )
    output_info["running_times"] = {}
    output_info["running_times"]["simopt_run_time"] = simopt_run_time.total_seconds()

    # Extract best params and add to output_info
    # We assume here that all parameters are integers
    # which they should be for the kinds of heuristic
    # policies we're using

    log.info("Running evaluation rollouts for the best params")

    best_params = np.array([v for v in study.best_params.values()]).reshape(
        policy.params_shape
    )
    # If no row labels, we don't want a multi-level dict
    # so handle separately
    if policy.param_row_names is None:
        output_info["policy_params"] = {
            str(param_name): int(param_value)
            for param_name, param_value in zip(
                policy.param_names.flat, best_params.flat
            )
        }
    # If there are row labels, easiest to convert to a dataframe and then into nested dict
    else:
        output_info["policy_params"] = pd.DataFrame(
            best_params,
            index=policy.param_row_names,
            columns=policy.param_col_names,
        ).to_dict()

    # Run evaluation for best policy, including computing kpis
    rollout_wrapper = hydra.utils.instantiate(
        cfg.rollout_wrapper, model_forward=policy.forward, return_info=True
    )
    rng_eval = jax.random.split(
        jax.random.PRNGKey(cfg.evaluation.seed), cfg.evaluation.num_rollouts
    )
    policy_params = jnp.array(best_params)
    rollout_results = rollout_wrapper.batch_rollout(rng_eval, policy_params)
    evaluation_output = create_evaluation_output_summary(cfg, rollout_results)

    eval_complete_time = datetime.now()
    eval_run_time = eval_complete_time - simopt_complete_time
    log.info(f"Evaluation duration: {(eval_run_time).total_seconds():.2f}s")
    output_info["running_times"]["eval_run_time"] = eval_run_time.total_seconds()

    log.info(f"Results from running best heuristic policy in simulation:")
    for k, v in evaluation_output.items():
        log.info(f"{k}: {v:.4f}")

    output_info["evaluation_output"] = evaluation_output
    to_yaml(output_info, "output_info.yaml")
    log.info("Evaluation output saved")


if __name__ == "__main__":
    main()
