import pytest
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import config as jax_config
import hydra
from viso_jax.simopt.run_optuna_simopt import (
    simopt_grid_sampler,
    simopt_other_sampler,
)
from .utils import get_absolute_config_paths

# Compare waste-conscious base stock policies identified
# through simulation optimization using
# HendrixPerishableOneProductGymnax with paramters reported
# in Section 2.3 of Hendrix et al (2019)


class TestHeuristicPolicy:
    @pytest.mark.parametrize(
        "exp_config_name,reported_params",
        [
            pytest.param(
                "m2/exp1",
                {"S": 13},
                id="m2/exp1",
            ),
            pytest.param(
                "m3/exp1",
                {"S": 15},
                id="m3/exp1",
            ),
            pytest.param(
                "m4/exp1",
                {
                    "S": 17
                },  # Reported as 16 in paper; in simulations with multiple seeds we find 17 better by a small (<0.01%) margin
                id="m4/exp1",
            ),
        ],
    )
    def test_heuristic_policy_same_as_reported(self, exp_config_name, reported_params):

        absolute_config_paths = get_absolute_config_paths()

        # Load in config settings for the experiment
        with hydra.initialize(
            version_base=None,
            config_path="../viso_jax/simopt/conf",
        ):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    f"hydra.searchpath={absolute_config_paths}",
                    f"+experiment=hendrix_perishable_one_product/{exp_config_name}",
                    "policy=waste_conscious_S_policy",
                ],
            )
        policy = hydra.utils.instantiate(cfg.policy)
        rollout_wrapper = hydra.utils.instantiate(
            cfg.rollout_wrapper, model_forward=policy.forward
        )
        rng_eval = jax.random.split(
            jax.random.PRNGKey(cfg.param_search.seed), cfg.param_search.num_rollouts
        )

        if cfg.param_search.sampler._target_ == "optuna.samplers.GridSampler":
            study = simopt_grid_sampler(cfg, policy, rollout_wrapper, rng_eval)
        else:
            study = simopt_other_sampler(cfg, policy, rollout_wrapper, rng_eval)

        assert study.best_params == reported_params
