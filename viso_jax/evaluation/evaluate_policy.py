import hydra
from omegaconf import OmegaConf
import logging
import pandas as pd
import jax
from viso_jax.utils.kpis import get_kpi_function
from viso_jax.utils.yaml import to_yaml

# Enable logging
log = logging.getLogger(__name__)


def create_evaluation_output_summary(cfg, rollout_results):

    log.info(
        f"Evaluating policy with {cfg.evaluation.num_rollouts} rollouts, each {cfg.rollout_wrapper.num_env_steps} steps long after a burn-in period of {cfg.rollout_wrapper.num_burnin_steps} steps"
    )

    eval_output = {}
    eval_output["daily_undiscounted_reward_mean"] = float(
        rollout_results["reward"].mean()
    )  # Equivalent to calculation mean for each rollout, then mean of those
    eval_output["daily_undiscounted_reward_std"] = float(
        rollout_results["reward"].mean(axis=-1).std()
    )  # Calc mean for each rollout, then std of those
    eval_output["cumulative_discounted_return_mean"] = float(
        rollout_results["cum_return"].mean()
    )  # One per rollout
    eval_output["cumulative_discounted_return_std"] = float(
        rollout_results["cum_return"].std()
    )  # One per rollout

    kpi_function = get_kpi_function(cfg.rollout_wrapper.env_id)
    kpis_per_rollout = kpi_function(rollout_results=rollout_results)
    for k, v in kpis_per_rollout.items():
        eval_output[f"{k}_mean"] = float(v.mean())
        eval_output[f"{k}_std"] = float(v.std())
    return eval_output


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    policy = hydra.utils.instantiate(cfg.policy)
    policy_params = policy.policy_params
    rollout_wrapper = hydra.utils.instantiate(
        cfg.rollout_wrapper, model_forward=policy.forward
    )
    rng_eval = jax.random.split(
        jax.random.PRNGKey(cfg.evaluation.seed), cfg.evaluation.num_rollouts
    )
    rollout_results = rollout_wrapper.batch_rollout(rng_eval, policy_params)

    evaluation_output = create_evaluation_output_summary(cfg, rollout_results)
    to_yaml(evaluation_output, "evaluation_output.yaml")


if __name__ == "__main__":
    main()
