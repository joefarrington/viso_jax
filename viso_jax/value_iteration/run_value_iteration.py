from datetime import datetime
import logging
from jax.config import config as jax_config
import hydra

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

    run_time = datetime.now() - start_time
    log.info(f"Total duration: {(run_time).total_seconds()}")


if __name__ == "__main__":
    main()
