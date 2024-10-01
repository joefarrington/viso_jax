from datetime import datetime
import logging
from jax.config import config as jax_config
import hydra
from viso_jax.evaluation.evaluate_policy import create_evaluation_output_summary
from viso_jax.utils.yaml import to_yaml
import jax
from omegaconf.dictconfig import DictConfig
from pathlib import Path

# Enable logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run value iteration and optionally evaluate the resulting policy in simulation."""
    start_time = datetime.now()

    if cfg.jax_settings.double_precision:
        jax_config.update("jax_enable_x64", True)

    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    VIR = hydra.utils.instantiate(cfg.vi_runner, output_directory=output_dir)
    vi_output = VIR.run_value_iteration(**cfg.run_settings)

    # Save final values
    log.info("Saving final values")
    vi_output["V"].to_csv(output_dir / "V.csv")
    log.info("Final values saved")

    # Save final policy if extracted
    if "policy" in vi_output.keys():
        log.info("Saving final policy")
        vi_output["policy"].to_csv(output_dir / "policy.csv")
        log.info("Final policy saved")

    vi_complete_time = datetime.now()
    vi_run_time = vi_complete_time - start_time
    log.info(f"Value iteration duration: {(vi_run_time).total_seconds():.2f}s")
    output_info = vi_output["output_info"]
    output_info["running_times"] = {}
    output_info["running_times"]["vi_run_time"] = vi_run_time.total_seconds()

    if cfg.evaluation.perform_eval:
        # Simulation doesn't need to be in double precision
        jax_config.update("jax_enable_x64", False)

        log.info("Running evaluation rollouts for policy found using value iteration")

        # Run evaluation for best policy, including computing kpis
        policy = hydra.utils.instantiate(
            cfg.policy, policy_params_df=vi_output["policy"]
        )
        policy_params = policy.policy_params

        rollout_wrapper = hydra.utils.instantiate(
            cfg.rollout_wrapper, model_forward=policy.forward, return_info=True
        )
        rng_eval = jax.random.split(
            jax.random.PRNGKey(cfg.evaluation.seed),
            cfg.evaluation.num_rollouts,
        )
        rollout_results = rollout_wrapper.batch_rollout(rng_eval, policy_params)
        evaluation_output = create_evaluation_output_summary(cfg, rollout_results)

        log.info("Results from running VI policy in simulation:")
        for k, v in evaluation_output.items():
            log.info(f"{k}: {v:.4f}")

        output_info["evaluation_output"] = evaluation_output

        eval_complete_time = datetime.now()
        eval_run_time = eval_complete_time - vi_complete_time
        log.info(f"Evaluation duration: {(eval_run_time).total_seconds():.2f}s")
        output_info["running_times"]["eval_run_time"] = eval_run_time.total_seconds()

    to_yaml(output_info, "output_info.yaml")
    log.info("Output saved")


if __name__ == "__main__":
    main()
