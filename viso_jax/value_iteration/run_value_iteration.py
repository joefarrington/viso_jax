from datetime import datetime
import logging
from jax.config import config as jax_config
import hydra
from viso_jax.evaluation.evaluate_policy import create_evaluation_output_summary
import jax

# Enable logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    start_time = datetime.now()

    if cfg.jax_settings.double_precision:
        jax_config.update("jax_enable_x64", True)

    VIR = hydra.utils.instantiate(
        cfg.vi_runner,
    )
    vi_output = VIR.run_value_iteration(**cfg.run_settings)

    vi_complete_time = datetime.now()
    vi_run_time = vi_complete_time - start_time
    log.info(f"Value iteration duration: {(vi_run_time).total_seconds():.2}s")

    if cfg.evaluation.perform_eval:
        # Simulation doesn't need to be in double precision
        jax_config.update("jax_enable_x64", False)

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
        evaluation_output_df = create_evaluation_output_summary(cfg, rollout_results)
        evaluation_output_df.to_csv("vi_policy_evaluation_output.csv")

        eval_complete_time = datetime.now()
        eval_run_time = eval_complete_time - vi_complete_time
        log.info(f"Evaluation duration: {(eval_run_time).total_seconds():.2f}s")
        log.info(f"Results from running VI policy in simulation:")
        for k, v in dict(evaluation_output_df).items():
            log.info(f"{k}: {v[0]:.4f}")


if __name__ == "__main__":
    main()
