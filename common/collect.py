import sys, os, time
from ruamel.yaml import YAML
from utils import system

import gym
import numpy as np 
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import envs
from common.sac import ReplayBuffer, SAC
from ml.models.reward import MLPReward
from utils.plots.train_plot import plot_sac_curve

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name, env_T = v['env']['env_name'], v['env']['T']
    state_indices = v['env']['state_indices']
    seed = v['seed']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    train_env = gym.make(env_name)
    test_env = gym.make(env_name) # original reward

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    print(f"Transfer: training Expert on {env_name}")

    replay_buffer = ReplayBuffer(
        gym_env.observation_space.shape[0], 
        gym_env.action_space.shape[0],
        device=device,
        size=v['sac']['buffer_size'])
    
    sac_agent = SAC(env_fn, replay_buffer,
        steps_per_epoch=env_T,
        update_after=env_T * v['sac']['random_explore_episodes'], 
        max_ep_len=env_T,
        seed=seed,
        start_steps=env_T * v['sac']['random_explore_episodes'],
        device=device,
        **v['sac']
        )
    #assert sac_agent.reinitialize == True
    
    sac_agent.ac.load_state_dict(torch.load('CustomAntgd_Humanoid-v3_2021.pt'))
    sac_agent.collect_data()