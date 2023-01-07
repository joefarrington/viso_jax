import pytest
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import hydra
import os
from .utils import get_absolute_config_paths

# Compare difference in value at each iteration after convergence (pi)

# Note that only the value for m2/exp1 is reported in Hendrix (2019), and
# it is reported as 5.509. This appears to be an error -
# the maximum value of pi should be 5 in this case.

# The comparisons here are to values of pi obtained when running the Matlab code
# kindly shared by Prof Hendrix using the settings for the two m=2 experiments


class TestConvergence:
    @pytest.mark.parametrize(
        "exp_config_name,reported_pi",
        [
            pytest.param("m2/exp1", 4.503, id="m2/exp1"),
            pytest.param("m2/exp2", 4.523, id="m2/exp2"),
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
                    f"+experiment=hendrix_perishable_substitution_two_product/{exp_config_name}",
                    "vi_runner.checkpoint_frequency=0",
                ],
            )

        VIR = hydra.utils.instantiate(cfg.vi_runner)

        vi_output = VIR.run_value_iteration(**cfg.run_settings)

        # Compare reported pi with max and min difference in values
        delta = vi_output["V"].values.reshape(-1) - VIR.V_old
        max_delta = jnp.max(delta)
        min_delta = jnp.min(delta)

        assert round(max_delta, 3) == reported_pi
        assert round(min_delta, 3) == reported_pi
