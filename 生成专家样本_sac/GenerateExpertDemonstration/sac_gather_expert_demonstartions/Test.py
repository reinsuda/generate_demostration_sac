import gym
import numpy as np

import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import argparse


# env_name, actor_path, critic_path, Gaussian,
def get_passer():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env_name', default="Ant-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')

    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--file_name', type=str, default=" ", metavar='N',
                        help='random seed (default: 123456)')

    args = parser.parse_args()
    return args


# get args
args = get_passer()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

model_path = "models/{}/{}".format(args.env_name, args.file_name)
agent.load_model(model_path)

total_numsteps = 0
updates = 0
return_list = []

for i in range(20):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        action = agent.select_action(state, evaluate=True)  # Sample action from policy
        next_state, reward, done, _ = env.step(action)  # Step
        # env.render()
        episode_reward += reward
        total_numsteps += 1
        episode_steps += 1
        state = next_state
    print("episode: {}".format(episode_steps))
    print("episode: {},reward: {}".format(i, episode_reward))
    return_list.append(episode_reward)
print("episode return mean: {} episode return std: {}".format(np.mean(return_list), np.std(return_list)))
