import numpy as np
import gym
import torch
import pickle
import os
from numpy import *
import pandas as pd
from torch.distributions.normal import Normal
from sac import SAC
import argparse


def traj_generator(pi, env, render=False):
    rewards = []
    tra_list = []
    done = False
    rets = 0.0
    state = env.reset()
    obs = []
    acs = []
    length = 0
    dones = []
    while not done:
        length += 1
        action = pi.select_action(state, evaluate=True)
        obs.append(state)
        acs.append(action)

        tra_list.append({"observation": state, "action": action})
        # env.render()
        next_state, reward, done, _ = env.step(action)
        dones.append(done)

        if render:
            env.render()
        rewards.append(reward)
        rets += reward
        state = next_state

    return obs, acs, rewards, dones, rets, length


def runner(env, model_path, timesteps_per_batch, number_trajs,
           return_low=None, save=False, reuse=False, args=None, render=True):
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    # initial network
    pi = SAC(ob_space.shape[0], ac_space, args)
    # pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    # U.initialize()

    # Prepare for rollouts load model
    # ----------------------------------------
    pi.load_model(model_path)
    # u.load_variables(load_model_path)

    episode_len_list = []
    episode_return_list = []
    current_traj_num = 0
    current_abandon_traj_num = 0
    expert_demonstrations = []

    obs_list, acs_list, rewards_list, rets_list, dones_list, length_list = [], [], [], [], [], []
    while current_traj_num <= number_trajs:
        obs, acs, rewards, dones, returns, trj_len = traj_generator(pi, env)
        if returns < int(suffix) - 200 or returns > int(suffix) + 200:
            print("return not in standard scale")
            continue
        if trj_len < timesteps_per_batch:
            current_abandon_traj_num += 1
            print("abandon episode number:{}!, episode len:{}".format(current_abandon_traj_num, trj_len))
            continue

        current_traj_num += 1

        if current_traj_num >= number_trajs:
            break

        if return_low is not None:
            if return_low + 100 < returns or return_low - 100 > returns:
                print("return low {}".format(sum(rewards)))
                continue
        if current_traj_num % 1 == 0:  # control the print frequency
            print("accept episode number:{}, len:{}, returns:{}".format(current_traj_num, trj_len, returns))
        print("returns".format(returns))
        obs_list.append(obs)
        acs_list.append(acs)
        rewards_list.append(rewards)
        dones_list.append(dones)
        rets_list.append(returns)
        length_list.append(trj_len)

    if save:
        # f = open(os.path.join(save_dir, env_name) + '/export_demonstrations_{}.pkl'.format(suffix), "wb")
        # pickle.dump(traj_data_list, f)
        # f.close()
        # print(os.path.join(save_dir, env_name) + '/export_demonstrations_{}.pkl'.format(suffix))
        #
        # #returns = [np.sum(ret) for ret in episode_return_list]
        # dataframe = pd.DataFrame(data={"mean return":mean([d["ep_ret"] for d in traj_data_list])},index=[0])
        # dataframe.to_csv(os.path.join(save_dir, env_name) + '/mean_{}.csv'.format(suffix), index=False, sep=',')
        # print("return mean: {}".format(mean([d["ep_ret"] for d in traj_data_list])))
        export_demonstration_path = "data/{}".format(args.env_name)
        if not os.path.exists(export_demonstration_path):
            os.makedirs(export_demonstration_path)

        file_name = '/export_demonstrations_{}_model_{}_mean_{}.npz'.format(args.env_name, suffix,
                                                                            str(int(np.mean(rets_list))))
        file_path = os.path.join(export_demonstration_path, file_name)
        print(file_path)
        print("return mean: {}, return std: {}".format(np.mean(rets_list), np.std(rets_list)))
        # Save the gathered data collections to the filesystem
        np.savez(export_demonstration_path + file_name, obs=np.array(obs_list),
                 acs=np.array(acs_list),
                 lens=np.array(length_list), returns=np.array(rets_list),
                 done=np.array(dones_list), reward=np.array(rewards_list))

        print("saving demonstrations")
        print(": {}".format(file_path))


def get_passer():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    # actor and critic model patch
    parser.add_argument('--actor_path', type=str,
                        default="sac_gather_expert_demonstartions/models/HalfCheetah/sac_actor_HalfCheetah",
                        # "./models/Hopper/",
                        help='the path save actor model')
    parser.add_argument('--critic_path', type=str,
                        default="sac_gather_expert_demonstartions/models/HalfCheetah/sac_critic_HalfCheetah",
                        # "./models/Hopper/",
                        help='the path save critic model')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true", default=True,
                        help='run on CUDA (default: False)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get args
    args = get_passer()
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    # fixed seed
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # parameter
    timesteps_per_trajs = 1000  # 1000
    number_trajs = 1500  # 1500
    stochastic_policy = False  # use stochastic/deterministic policy to evaluate
    save_trajectory = True  # save the trajectories or not
    render = False  # True  # show mujoco image
    suffix = '3494'  # add additional information, default= ''

    model_path = "models/{}/sac_model_{}_{}.pt".format(args.env_name, args.env_name, suffix)
    # generate and save expert demonstration
    runner(env=env, model_path=model_path,
           timesteps_per_batch=timesteps_per_trajs, number_trajs=number_trajs,
           return_low=None, save=save_trajectory, args=args,
           render=render)
