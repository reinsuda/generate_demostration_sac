import numpy as np
import gym
import torch
import pickle
from arguments import get_args
from models import cnn_net, mlp_net
import os
from numpy import *
import pandas as pd
from torch.distributions.normal import Normal


# denormalize
def normalize(x, mean, std, clip=10):
    x -= mean
    x /= (std + 1e-8)
    return np.clip(x, -clip, clip)


# get tensors for the agent
def get_tensors(obs, env_type, filters=None):
    if env_type == 'atari':
        tensor = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    elif env_type == 'mujoco':
        tensor = torch.tensor(normalize(obs, filters.rs.mean, filters.rs.std), dtype=torch.float32).unsqueeze(0)
    return tensor


class Policy_PPO(object):
    def __init__(self, state_shape, action_shape, dist, model_path):
        # load policy network
        self.network = mlp_net(state_shape, action_shape, dist)
        net_models, self.filters = torch.load(model_path, map_location=lambda storage, loc: storage)

        # load models
        self.network.load_state_dict(net_models)

    def select_action(self, state):
        obs_tensor = get_tensors(state, args.env_type, self.filters)
        _, pi = self.network(obs_tensor)
        if args.env_type == 'atari':
            actions = torch.argmax(pi, dim=1).item()
        elif args.env_type == 'mujoco':
            if args.dist == 'gauss':
                mean, std = pi
                actions = Normal(mean, std).sample().detach().numpy().squeeze()
                # mean, _ = pi
                # actions = mean.detach().numpy().squeeze()
            elif args.dist == 'beta':
                alpha, beta = pi
                actions = (alpha - 1) / (alpha + beta - 2)
                actions = actions.numpy().squeeze()
                actions = -1 + 2 * actions
        return actions


def traj_generator(pi, env, render=False):
    rewards = []
    tra_list = []
    done = False
    rets = 0.
    state = env.reset()
    obs = []
    acs = []
    length = 0
    dones = []
    while not done:
        length += 1
        action = pi.select_action(state)
        obs.append(state)
        acs.append(action)
        tra_list.append({"observation": state, "action": action})
        next_state, reward, done, _ = env.step(action)
        dones.append(done)

        if render:
            env.render()
        rewards.append(reward)
        rets += reward
        state = next_state

    return obs, acs, rewards, dones, rets, length


def runner(env_name, dist, save_dir, timesteps_per_batch, number_trajs, seed, return_low_flag=False, return_low=5000,
           save=False, suffix="", render=False):
    env = gym.make(env_name)
    env.seed(seed)
    model_path = os.path.join(save_dir, env_name) + '/model_{}.pt'.format(suffix)
    pi = Policy_PPO(env.observation_space.shape[0], env.action_space.shape[0], dist, model_path)

    obs_list, acs_list, rewards_list, rets_list, dones_list, length_list = [], [], [], [], [], []

    traj_data_list = []
    current_traj_num = 0
    current_abandon_traj_num = 0

    while current_traj_num <= number_trajs:
        traj_data = {}
        obs, acs, rewards, dones, rets, trj_len = traj_generator(pi, env)
        traj_data["ob"] = obs
        traj_data["ac"] = acs
        traj_data["length"] = trj_len
        traj_data["ep_ret"] = rets
        traj_data["dones"] = dones
        traj_data["rews"] = rewards

        traj_data_list.append(traj_data)
        if trj_len < timesteps_per_batch:
            current_abandon_traj_num += 1
            print("abandon traj num: {}, trj len".format(current_abandon_traj_num, trj_len))
            continue
        if return_low_flag:
            if return_low > sum(rewards):
                print("return low {}".format(sum(rewards)))
                continue

        obs_list.append(obs)
        acs_list.append(acs)
        rewards_list.append(rewards)
        dones_list.append(dones)
        rets_list.append(rets)
        length_list.append(trj_len)

        current_traj_num += 1
        if current_traj_num % 1 == 0:
            print("accept episode number:{}, len:{}, returns:{}".format(current_traj_num, trj_len, np.sum(rewards)))
        if current_traj_num > number_trajs:
            break

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

        file_path = os.path.join(save_dir, env_name)
        file_name = '/export_demonstrations_{}_model_{}_mean_{}.npz'.format(args.env_name, suffix,
                                                                            str(int(np.mean(rets_list))))
        path = os.path.join(file_path, file_name)
        print(file_path + file_name)
        # Save the gathered data collections to the filesystem
        np.savez(file_path + file_name, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(length_list), returns=np.array(rets_list),
                 done=np.array(dones_list), reward=np.array(rewards_list))

        print("saving demonstrations")
        print("  @: {}.npz".format(path))


if __name__ == '__main__':
    args = get_args()
    print("222")
    # parameter
    timesteps_per_trajs = 1000  # 1000
    number_trajs = 1500  # 1500
    stochastic_policy = False  # use stochastic/deterministic policy to evaluate
    save_trajectory = True  # save the trajectories or not
    render = True  # True  # show mujoco image

    runner(env_name=args.env_name, dist=args.dist, save_dir=args.save_dir, \
           timesteps_per_batch=timesteps_per_trajs, number_trajs=number_trajs, \
           seed=np.random.randint(1000000), return_low_flag=True, return_low=3500, \
           save=True, suffix="2128", render=False)

    # hopper 3786跑专家样本比较好

    print("end")
    # load policy network

    # generate and save expert demonstration
