#!/usr/bin/env bash

scenario_name=hendrix_perishable_substitution_two_product
max_useful_lives=( 2 3 )
experiments=( 1 2 3 4 ) # No exp 3, 4 for m=2, but doesn't cause any problems to run this way
date_now=$(date +"%Y-%m-%d")
time_now=$(date +"%H-%M-%S")

echo "Running all experiments for Scenario B with settings from Hendrix et al (2019) at ${date_now} ${time_now}"

cd ../viso_jax/value_iteration

for m in ${max_useful_lives[@]}
do
    for exp in ${experiments[@]}
    do  
        echo "Running value iteration for m=${m}, exp=${exp}"
        python run_value_iteration.py +experiment=${scenario_name}/m${m}/exp${exp} \
        hydra.run.dir=./outputs/run_all_scenario_b_hendrix_settings/${date_now}/${time_now}/m${m}/exp${exp}
    done
done

cd ../simopt

for m in ${max_useful_lives[@]}
do
    for exp in ${experiments[@]}
    do  
        echo "Running simulation optimization for m=${m}, exp=${exp}"
        python run_optuna_simopt.py +experiment=${scenario_name}/m${m}/exp${exp} \
        hydra.run.dir=./outputs/run_all_scenario_b_hendrix_settings/${date_now}/${time_now}/m${m}/exp${exp}
    done
done
echo "All experiments completed."