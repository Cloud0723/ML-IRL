'''
Trained and saved reward estimator here.
'''
import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML

from ml.models.reward import MLPReward
from common.sac import ReplayBuffer, SAC

import envs
from utils import system, collect, logger, eval
import datetime
import dateutil.tz
import json, copy

def ML_loss(div: str, agent_samples, expert_samples, reward_func, device):
    ''' NOTE: only for ML: E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    assert div in ['maxentirl']
    sA, _, _ = agent_samples
    _, T, d = sA.shape

    sA_vec = torch.FloatTensor(sA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(expert_samples).reshape(-1, d).to(device)

    t1 = reward_func.r(sA_vec).view(-1) # E_q[r(tau)]
    t2 = reward_func.r(sE_vec).view(-1) # E_p[r(tau)]

    surrogate_objective = t1.mean() - t2.mean() # gradient ascent
    return T * surrogate_objective # same scale

def ML_sa_loss(div: str, agent_samples, expert_samples, reward_func, device):
    ''' NOTE: only for ML_sa: E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    #assert div in ['maxentirl']
    sA, aA, _ = agent_samples
    print(sA.shape,aA.shape)
    sA=np.concatenate([sA,aA],2)
    _, T, d = sA.shape

    sA_vec = torch.FloatTensor(sA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(expert_samples).reshape(-1, d).to(device)

    t1 = reward_func.r(sA_vec).view(-1) # E_q[r(tau)]
    t2 = reward_func.r(sE_vec).view(-1) # E_p[r(tau)]

    surrogate_objective = t1.mean() - t2.mean() # gradient ascent
    return T * surrogate_objective # same scale



def try_evaluate(itr: int, policy_type: str, sac_info):
    assert policy_type in ["Running"]
    update_time = itr * v['reward']['gradient_step']
    env_steps = itr * v['sac']['epochs'] * v['env']['T']
    agent_emp_states = samples[0].copy()
    assert agent_emp_states.shape[0] == v['irl']['training_trajs']

    metrics = eval.KL_summary(expert_samples, agent_emp_states.reshape(-1, agent_emp_states.shape[2]), 
                         env_steps, policy_type)
    # eval real reward
    real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['T'], True)
    metrics['Real Det Return'] = real_return_det
    print(f"real det return avg: {real_return_det:.2f}")
    logger.record_tabular("Real Det Return", round(real_return_det, 2))

    real_return_sto = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['T'], False)
    metrics['Real Sto Return'] = real_return_sto
    print(f"real sto return avg: {real_return_sto:.2f}")
    logger.record_tabular("Real Sto Return", round(real_return_sto, 2))


    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)

    return real_return_det, real_return_sto

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['irl']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()
    
    # assumptions
    assert v['obj'] in ['maxentirl','maxentirl_sa']
    assert v['IS'] == False

    # logs
    exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/{v['obj']}" # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)            
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp ml/irl_samples.py {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # load expert samples from trained policy
    #expert_trajs = torch.load(f'expert_data/states/{env_name}.pt').numpy()[:, :, state_indices]
    env_name=env_name.split('-')[0]
    expert_trajs = np.load(f'expert_data/{env_name}/states.npy')
    expert_trajs = expert_trajs[:num_expert_trajs, :, :] # select first expert_episodes
    expert_samples = expert_trajs.copy().reshape(-1, len(state_indices))
    #expert_a = torch.load(f'expert_data/actions/{env_name}.pt').numpy()[:, :, :]
    expert_a = np.load(f'expert_data/{env_name}/actions.npy')
    expert_a = expert_a[:num_expert_trajs, :, :] # select first expert_episodes
    expert_a_samples = expert_a.copy().reshape(-1, action_size)
    expert_samples_sa=np.concatenate([expert_samples,expert_a_samples],1)
    print(expert_trajs.shape, expert_samples_sa.shape) # ignored starting state

    # Initilialize reward as a neural network
    
    reward_func = MLPReward(len(state_indices), **v['reward'], device=device).to(device)
    sa=False
    if v['obj']=='maxentirl_sa':
        sa=True
        reward_func = MLPReward(len(state_indices)+action_size, **v['reward'], device=device).to(device)
    reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=v['reward']['lr'], 
        weight_decay=v['reward']['weight_decay'], betas=(v['reward']['momentum'], 0.999))
    
    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    for itr in range(v['irl']['n_itrs']):
        if v['sac']['reinitialize'] or itr == 0:
            # Reset SAC agent with old policy, new environment, and new replay buffer
            print("Reinitializing sac")
            replay_buffer = ReplayBuffer(
                state_size, 
                action_size,
                device=device,
                size=v['sac']['buffer_size'])
                
            sac_agent = SAC(env_fn, replay_buffer,
                steps_per_epoch=v['env']['T'],
                update_after=v['env']['T'] * v['sac']['random_explore_episodes'], 
                max_ep_len=v['env']['T'],
                seed=seed,
                start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
                reward_state_indices=state_indices,
                device=device,
                sa=sa,
                **v['sac']
            )
        
        sac_agent.reward_function = reward_func.get_scalar_reward # only need to change reward in sac
        sac_info = sac_agent.learn_mujoco(print_out=True)

        start = time.time()
        samples = collect.collect_trajectories_policy_single(gym_env, sac_agent, 
                        n = v['irl']['training_trajs'], state_indices=state_indices)
        # Fit a density model using the samples
        agent_emp_states = samples[0].copy()
        agent_emp_states = agent_emp_states.reshape(-1,agent_emp_states.shape[2]) # n*T states
        print(f'collect trajs {time.time() - start:.0f}s', flush=True)
        # print(agent_emp_states.shape)

        start = time.time()

        # optimization w.r.t. reward
        reward_losses = []
        for _ in range(v['reward']['gradient_step']):
            if v['irl']['resample_episodes'] > v['irl']['expert_episodes']:
                expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=True)
                expert_trajs_train = expert_trajs[expert_res_indices].copy() # resampling the expert trajectories
            elif v['irl']['resample_episodes'] > 0:
                expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=False)
                expert_trajs_train = expert_trajs[expert_res_indices].copy()
            else:
                expert_trajs_train = None # not use expert trajs

            if v['obj'] == 'maxentirl':
                loss = ML_loss(v['obj'], samples, expert_samples, reward_func, device)
            elif v['obj'] == 'maxentirl_sa':
                loss = ML_sa_loss(v['obj'], samples, expert_samples_sa, reward_func, device) 
            
            reward_losses.append(loss.item())
            print(f"{v['obj']} loss: {loss}")
            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()

        # evaluating the learned reward
        real_return_det, real_return_sto = try_evaluate(itr, "Running", sac_info)
        if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
            max_real_return_det, max_real_return_sto = real_return_det, real_return_sto
            torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), 
                    f"model/reward_model_itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}.pkl"))

        logger.record_tabular("Itration", itr)
        logger.record_tabular("Reward Loss", loss.item())
        if v['sac']['automatic_alpha_tuning']:
            logger.record_tabular("alpha", sac_agent.alpha.item())
        
        if v['irl']['save_interval'] > 0 and (itr % v['irl']['save_interval'] == 0 or itr == v['irl']['n_itrs']-1):
            torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), f"model/reward_model_{itr}.pkl"))

        logger.dump_tabular()