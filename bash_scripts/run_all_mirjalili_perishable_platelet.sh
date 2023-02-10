#!/usr/bin/env bash

scenario=mirjalili_perishable_platelet
max_useful_lives=( 3 5 8)
experiments=( 1 2 )
date_now=$(date +"%Y-%m-%d")
time_now=$(date +"%H-%M-%S")

echo "Running all experiments for scenario ${scenario} at ${date_now} ${time_now}"

cd ../viso_jax/value_iteration

for m in ${max_useful_lives[@]}
do  
    if [ $m -gt 5 ]
    then
        echo "Skipping m=${m}" # We can't run VI when m=8, so skip it
        continue
    fi
    for exp in ${experiments[@]}
    do  
        echo "Running value iteration for m=${m}, exp=${exp}"
        python run_value_iteration.py +experiment=${scenario}/m${m}/exp${exp} \
        hydra.run.dir=./outputs/run_all_${scenario}/${date_now}/${time_now}/m${m}/exp${exp}
    done
done

cd ../simopt

for m in ${max_useful_lives[@]}
do
    for exp in ${experiments[@]}
    do  
        echo "Running simulation optimization for m=${m}, exp=${exp}"
        python run_optuna_simopt.py +experiment=${scenario}/m${m}/exp${exp} \
        hydra.run.dir=./outputs/run_all_${scenario}/${date_now}/${time_now}/m${m}/exp${exp}
    done
done

echo "All experiments completed."
