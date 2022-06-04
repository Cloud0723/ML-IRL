## MUJOCO DATA COLLECT
We trained a SAC agent and used it to collect expert data.
You can see the MUJOCO DATA COLLECT part in run.sh to try this.

## MUJOCO IRL BENCHMARK
Extract the reward network from the expert data

## MUJOCO TRANSFER
Reward Transfer: Firstly use IRL algorithms to infer the reward functions in Custom-Ant, and then transfer these recovered reward functions to Disabled-Ant for further evaluation.
Data Transfer: Train IRL agents in Disabled-Ant by using the Custom-Ant expert trajectories

## files
common : sac algorithm, data collection and transfer codes
configs: configs file
envs   : Custom-ant and Disabled-ant mujoco envs 
expert_data : saved expert trajectories
logs   : Experiment results
ml  : ML-IRL algorithm
utils  : some other functions
