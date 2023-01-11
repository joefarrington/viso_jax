import hydra
from omegaconf import OmegaConf
import logging
import pandas as pd
import jax
from viso_jax.utils.kpis import get_kpi_function

# Enable logging
log = logging.getLogger(__name__)


def create_evaluation_output_summary(cfg, rollout_results):

    log.info(
        f"Evaluating policy with {cfg.evaluation.num_rollouts} rollouts, each {cfg.rollout_wrapper.num_env_steps} steps long after a burn-in period of {cfg.rollout_wrapper.num_burnin_steps} steps"
    )

    log_dict = {}
    log_dict["daily_undiscounted_reward_mean"] = rollout_results[
        "reward"
    ].mean()  # Equivalent to calculation mean for each rollout, then mean of those
    log_dict["daily_undiscounted_reward_std"] = (
        rollout_results["reward"].mean(axis=-1).std()
    )  # Calc mean for each rollout, then std of those
    log_dict["cumulative_discounted_return_mean"] = rollout_results[
        "cum_return"
    ].mean()  # One per rollout
    log_dict["cumulative_discounted_return_std"] = rollout_results[
        "cum_return"
    ].std()  # One per rollout

    kpi_function = get_kpi_function(cfg.rollout_wrapper.env_id)
    kpis_per_rollout = kpi_function(rollout_results=rollout_results)
    for k, v in kpis_per_rollout.items():
        log_dict[f"{k}_mean"] = v.mean()
        log_dict[f"{k}_std"] = v.std()
    return pd.DataFrame(log_dict, index=[0])


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
    evaluation_output.to_csv("evaluation_output.csv")


if __name__ == "__main__":
    main()
