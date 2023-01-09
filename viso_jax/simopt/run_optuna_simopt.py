import hydra
from omegaconf import OmegaConf
import logging
import pandas as pd
import optuna
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
from viso_jax.evaluation.evaluate_policy import create_evaluation_output_summary

# Enable logging
log = logging.getLogger(__name__)

# Grid sampler is not straightfordly compatible with the ask/tell
# interface so we need to treat it a bit differently to avoid
# to avoid duplication and handle RuntimeError
# https://github.com/optuna/optuna/issues/4121
def simopt_grid_sampler(cfg, policy, rollout_wrapper, rng_eval):
    search_space = {
        k: list(range(cfg.param_search.param_low, cfg.param_search.param_high + 1))
        for k in policy.param_names.flat
    }
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
        log.info(f"Round {i}: Suggesting parameters.")
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
                            cfg.param_search.param_low,
                            cfg.param_search.param_high,
                        )
                        for p in policy.param_names.flat
                    ]
                ).reshape(policy.params_shape)
            )
        policy_params = jnp.array(policy_params)
        log.info(f"Round {i}: Simulating rollouts.")
        rollout_results = rollout_wrapper.population_rollout(rng_eval, policy_params)
        log.info(f"Round {i}: Processing results.")
        objectives = rollout_results["reward"].mean(axis=(-2, -1))

        for idx in range(num_parallel_trials):
            trials[idx].set_user_attr(
                "mean_cumulative_discounted_reward",
                rollout_results["cum_return"][idx].mean(),
            )
            try:
                study.tell(trials[idx], objectives[idx])
            except RuntimeError:
                break

        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean daily_reward: {study.best_value}"
        )
        i += 1
    return study


def simopt_other_sampler(cfg, policy, rollout_wrapper, rng_eval):
    sampler = hydra.utils.instantiate(
        cfg.param_search.sampler, seed=cfg.param_search.seed
    )
    study = optuna.create_study(sampler=sampler, direction="maximize")

    for i in range(1, cfg.param_search.max_iterations + 1):
        trials = []
        policy_params = []
        log.info(f"Round {i}: Suggesting parameters.")
        for j in range(cfg.param_search.max_parallel_trials):
            trial = study.ask()
            trials.append(trial)
            policy_params.append(
                np.array(
                    [
                        trial.suggest_int(
                            f"{p}",
                            cfg.param_search.param_low,
                            cfg.param_search.param_high,
                        )
                        for p in policy.param_names.flat
                    ]
                ).reshape(policy.params_shape)
            )
        policy_params = jnp.array(policy_params)
        log.info(f"Round {i}: Simulating rollouts.")
        rollout_results = rollout_wrapper.population_rollout(rng_eval, policy_params)
        log.info(f"Round {i}: Processing results.")
        objectives = rollout_results["reward"].mean(axis=(-2, -1))

        for idx in range(cfg.param_search.max_parallel_trials):
            trials[idx].set_user_attr(
                "mean_cumulative_discounted_reward",
                rollout_results["cum_return"][idx].mean(),
            )
            study.tell(trials[idx], objectives[idx])

        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean daily_reward: {study.best_value}"
        )

    return study


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
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

    best_params = np.array([v for v in study.best_params.values()]).reshape(
        policy.params_shape
    )
    pd.DataFrame(
        best_params, index=policy.param_row_names, columns=policy.param_col_names
    ).to_csv("best_params.csv")

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
    evaluation_output.to_csv("best_policy_evaluation_output.csv")

    log.info("Search completed and results saved")


if __name__ == "__main__":
    main()
