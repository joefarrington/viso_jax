import pytest
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import hydra
import os
from .utils import get_absolute_config_paths

# Compare policy output by MirjaliliPerishablePlateletVIR (and
# for one cases, MirjaliliPerishablePlateletDeterministicUsefulLifeVIR
# a class written specifically for this comparison) with policies
# printed in Fig 6.1 of Mirjalili (2022)
# MM kindly provided the data used to produce the plots in thesis
# and the exact values used for the demand parameters in Table 6.1
# which we use below instead of the rounded values generally used
# because the rounding does lead to small differences in the policy


class TestPolicy:
    @pytest.mark.parametrize(
        "exp_config_name,VIR_class,fixed_order_cost,reported_policy_filename",
        [
            # m=3, fixed_order_cost=10, shelf life on arrival has endogenous uncertainity
            pytest.param(
                "m3/exp2",
                "viso_jax.scenarios.mirjalili_perishable_platelet.vi_runner.MirjaliliPerishablePlateletVIR",
                10,
                "mirjalili_perishable_platelet_m3_fig6_1_left.csv",
                id="fig6.1/left",
            ),
            # m=3, fixed_order_cost=0, shelf life on arrival has endogenous uncertainity
            pytest.param(
                "m3/exp2",
                "viso_jax.scenarios.mirjalili_perishable_platelet.vi_runner.MirjaliliPerishablePlateletVIR",
                0,
                "mirjalili_perishable_platelet_m3_fig6_1_middle.csv",
                id="fig6.1/middle",
            ),
            # m=3, fixed_order_cost=0, shelf life on arrival is deterministic
            pytest.param(
                "m3/exp2",
                "viso_jax.scenarios.mirjalili_perishable_platelet.vi_runner.MirjaliliPerishablePlateletDeterministicUsefulLifeVIR",
                0,
                "mirjalili_perishable_platelet_m3_fig6_1_right.csv",
                id="fig6.1/right",
            ),
        ],
    )
    def test_policy_same_as_reported(
        self,
        tmpdir,
        shared_datadir,
        exp_config_name,
        VIR_class,
        fixed_order_cost,
        reported_policy_filename,
    ):
        jax.config.update("jax_enable_x64", True)
        absolute_config_paths = get_absolute_config_paths()

        # Change working directory to avoid clutter
        os.chdir(tmpdir)

        # Demand parameters without rounding provided by MM
        n = [3.497361, 10.985837, 7.183407, 11.064622, 5.930222, 5.473242, 2.193797]
        delta = [5.660569, 6.922555, 6.504332, 6.165049, 5.816060, 3.326408, 3.426814]

        # Load in config settings for the experiment
        with hydra.initialize(
            version_base=None,
            config_path="../viso_jax/value_iteration/conf",
        ):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    f"hydra.searchpath={absolute_config_paths}",
                    f"+experiment=mirjalili_perishable_platelet/{exp_config_name}",
                    "vi_runner.checkpoint_frequency=1",
                    f"scenario_settings.fixed_order_cost={fixed_order_cost}",
                    f"scenario_settings.weekday_demand_negbin_n={n}",
                    f"scenario_settings.weekday_demand_negbin_delta={delta}",
                    f"vi_runner._target_={VIR_class}",
                ],
            )

        VIR = hydra.utils.instantiate(
            cfg.vi_runner,
        )
        vi_output = VIR.run_value_iteration(**cfg.run_settings)

        # Post-process policy to match reported form
        # Including clipping so that only includes stock-holding up to 10 units per age
        vi_policy = vi_output["policy"]
        vi_policy["state"] = [x[0] for x in vi_policy.index]
        vi_policy["x2"] = [x[1] for x in vi_policy.index]
        vi_policy["x1"] = [x[2] for x in vi_policy.index]
        vi_policy_monday = vi_policy[vi_policy["state"] == 0]
        vi_policy_monday = vi_policy_monday.pivot_table(
            index="x2", columns="x1", values="order_quantity"
        ).iloc[:11, :11]

        # Load in the reported policy
        reported_policy = pd.read_csv(
            f"{shared_datadir}/{reported_policy_filename}",
            index_col=0,
            header=0,
        )

        assert np.all(vi_policy_monday.values == reported_policy.values)
