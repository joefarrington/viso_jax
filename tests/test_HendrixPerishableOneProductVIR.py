import pytest
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import hydra
import os
from .utils import get_absolute_config_paths

# Compare difference in value at each iteration after convergence (pi)
# to value reported in Hendrix et al (2019)
class TestConvergence:
    @pytest.mark.parametrize(
        "exp_config_name,reported_pi",
        [
            pytest.param("m2/exp1", 2.22, id="m2/exp1"),
            pytest.param("m3/exp1", 2.40, id="m3/exp1"),
            pytest.param("m4/exp1", 2.47, id="m4/exp1"),
        ],
    )
    def test_pi_same_as_reported(self, tmpdir, exp_config_name, reported_pi):

        jax.config.update("jax_enable_x64", True)
        absolute_config_paths = get_absolute_config_paths()

        # Change working directory to avoid clutter
        os.chdir(tmpdir)

        # Load in config settings for the experiment
        with hydra.initialize(
            version_base=None,
            config_path="../viso_jax/value_iteration/conf",
        ):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    f"hydra.searchpath={absolute_config_paths}",
                    f"+experiment=hendrix_perishable_one_product/{exp_config_name}",
                    "vi_runner.checkpoint_frequency=0",
                ],
            )

        VIR = hydra.utils.instantiate(cfg.vi_runner)
        vi_output = VIR.run_value_iteration(**cfg.run_settings)

        # Compare reported pi with max and min difference in values
        delta = vi_output["V"].values.reshape(-1) - VIR.V_old
        max_delta = jnp.max(delta)
        min_delta = jnp.min(delta)

        assert round(max_delta, 2) == reported_pi
        assert round(min_delta, 2) == reported_pi
