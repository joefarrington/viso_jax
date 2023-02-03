#!/usr/bin/env bash

scenario=de_moor_perishable
max_useful_life=( 2 3 4 5)
experiments=( 1 2 3 4 5 6 7 8)
date_now=$(date +"%Y-%m-%d")
time_now=$(date +"%H-%M-%S")

cd ../viso_jax/value_iteration

for m in ${max_useful_life[@]}
do
    for exp in ${experiments[@]}
    do  
        echo "Running value iteration for m=${m}, exp=${exp}"
        python run_value_iteration.py +experiment=${scenario}/m${m}/exp${exp} \
        hydra.run.dir=./outputs/run_all_de_moor_perishable/${date_now}/${time_now}/m${m}/exp${exp}
    done
done

cd ../simopt

for m in ${max_useful_life[@]}
do
    for exp in ${experiments[@]}
    do  
        echo "Running simulation optimization for m=${m}, exp=${exp}"
        python run_optuna_simopt.py +experiment=${scenario}/m${m}/exp${exp} \
        hydra.run.dir=./outputs/run_all_de_moor_perishable/${date_now}/${time_now}/m${m}/exp${exp}
    done
done

echo "All experiments completed."
