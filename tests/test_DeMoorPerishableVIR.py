import pytest
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import hydra
import os
from .utils import get_absolute_config_paths

# Compare policy output by DeMoorPerishableVIR with policies
# printed in Fig 3 of De Moor et al (2022)
class TestPolicy:
    @pytest.mark.parametrize(
        "exp_config_name,reported_policy_filename",
        [
            pytest.param(
                "m2/exp1",
                "de_moor_perishable_m2_exp1_reported_policy.csv",
                id="m2/exp1",
            ),
            pytest.param(
                "m2/exp2",
                "de_moor_perishable_m2_exp2_reported_policy.csv",
                id="m2/exp2",
            ),
        ],
    )
    def test_policy_same_as_reported(
        self, tmpdir, shared_datadir, exp_config_name, reported_policy_filename
    ):
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
                    f"+experiment=de_moor_perishable/{exp_config_name}",
                    "vi_runner.checkpoint_frequency=0",
                ],
            )

        VIR = hydra.utils.instantiate(
            cfg.vi_runner,
        )
        vi_output = VIR.run_value_iteration(**cfg.run_settings)

        # Post-process policy to match reported form
        # Including clipping so that only includes stock-holding up to 8 units per agre
        vi_policy = vi_output["policy"].reset_index()
        vi_policy.columns = ["state", "order_quantity"]
        vi_policy["Units in stock age 2"] = [(x)[1] for x in vi_policy["state"]]
        vi_policy["Units in stock age 1"] = [(x)[0] for x in vi_policy["state"]]
        vi_policy = vi_policy.pivot(
            index="Units in stock age 1",
            columns="Units in stock age 2",
            values="order_quantity",
        )
        vi_policy = vi_policy.loc[list(range(9)), list(range(9))].sort_index(
            ascending=False
        )

        # Load in the reported policy
        reported_policy = pd.read_csv(
            f"{shared_datadir}/{reported_policy_filename}",
            index_col=0,
            header=0,
        )

        assert np.all(vi_policy.values == reported_policy.values)
