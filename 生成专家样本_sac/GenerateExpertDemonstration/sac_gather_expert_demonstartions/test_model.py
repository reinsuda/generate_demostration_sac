import pandas as pd
import cv2
import numpy as np
import gym
import os
from mpi4py import MPI
import time
from sac import SAC
import torch
from passer import get_passer

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


def compute_all_file(file_name, args):
    # get the arguments
    # create the environment

    env = gym.make(args.env_name)
    # get the model path
    # model_path = args.save_dir + args.env_name + '/model_2327.pt'
    model_path = file_name
    # create the network
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_model(file_name)
    # start to play the demo

    reward_total = []
    # just one episode
    for _ in range(500):
        obs = env.reset()
        episode_rewards = 0.

        while True:
            # env.render()
            with torch.no_grad():
                # get actions
                actions = agent.select_action(obs)
            obs_, reward, done, _ = env.step(actions)
            episode_rewards += reward

            if done:
                break
            obs = obs_
        # print("episode return: {}".format(episode_rewards))
        reward_total.append(episode_rewards)
    mean, std = np.mean(reward_total), np.std(reward_total)
    print('the rewrads mean: {}, the rewards sqrt: {}'.format(mean, std))
    return mean, std


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]


if __name__ == "__main__":
    args = get_passer()
    file_names = get_imlist("models/" + args.env_name)
    time_start = time.time()

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    if comm_rank == 0:
        data = []
        for i in range(len(file_names)):
            index = i % comm_size
            if index > len(data) - 1:
                data.append([file_names[i]])
            else:
                data[index].append(file_names[i])
    else:
        data = None
    local_data = comm.scatter(data, root=0)

    mean_list, std_list, = [], []
    result = []
    for file_name in local_data:
        mean, std = compute_all_file(file_name, args)
        result.append([file_name, mean, std])
        mean_list.append(mean)
        std_list.append(std)
    gather_result = comm.gather(result, root=0)
    if comm_rank == 0:
        print("times: {}".format(time.time() - time_start))

        col_names = ["file_name", "mean", "std"]
        gather_result = np.concatenate(gather_result, axis=0)
        df = pd.DataFrame(columns=col_names, data=gather_result)

        df.to_csv("models/" + args.env_name + "/test_model.csv", encoding="utf-8", index=False)
