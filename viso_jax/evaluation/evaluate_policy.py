import hydra
from omegaconf import OmegaConf
import logging
import pandas as pd
import jax
import jax.numpy as jnp
import gymnax
import viso_jax
from viso_jax.utils.kpis import get_kpi_function

# Enable logging
log = logging.getLogger(__name__)


def create_evaluation_output_summary(cfg, rollout_results):
    log_dict = OmegaConf.to_container(
        cfg.rollout_wrapper.env_kwargs
    ) | OmegaConf.to_container(cfg.rollout_wrapper.env_params)
    log_dict = {
        k: str(v) for k, v in log_dict.items()
    }  # Convert values to string so any arrays can be stored in one df row
    log_dict["seed"] = cfg.evaluation.seed
    log_dict["num_rollouts"] = cfg.evaluation.num_rollouts
    log_dict["env_id"] = cfg.rollout_wrapper.env_id
    log_dict["num_burnin_steps"] = cfg.rollout_wrapper.num_burnin_steps
    log_dict["num_env_steps"] = cfg.rollout_wrapper.num_env_steps
    log_dict["mean_cumulative_return"] = rollout_results["cum_return"].mean()
    log_dict["mean_daily_undiscounted_reward"] = rollout_results["reward"].mean()

    kpi_function = get_kpi_function(cfg.rollout_wrapper.env_id)
    kpis_per_rollout = kpi_function(rollout_results=rollout_results)
    kpi_dict = {k: v.mean() for k, v in kpis_per_rollout.items()}
    log_dict = log_dict | kpi_dict
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