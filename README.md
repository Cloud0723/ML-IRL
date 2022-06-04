## MUJOCO DATA COLLECT
We trained an SAC agent and use it collect expert data.
You can see the MUJOCO DATA COLLECT part in run.sh to try this.

## MUJOCO IRL BENCHMARK
Extract the reward network from the expert data

## MUJOCO TRANSFER
Using the reward network do transfer task
Reward Transfer: use the trained reward network to retrain an SAC agent
Data Transfer: Train IRL agents in Disabled-Ant by using the Custom-Ant expert trajectories

## files
common : sac algorithm, data collection and transfer codes
configs: configs file
envs   : Custom-ant and Disabled-ant mujoco envs 
expert_data : saved expert trajectories
logs   : Experiment results
ml  : ML-IRL algorithm
utils  : some other functions