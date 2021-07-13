import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from passer import get_passer
from sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory


def traj_generator(pi, args, render=False):
    envl = gym.make(args.env_name)
    #envl.seed(args.seed)

    d = False
    rts = 0.0
    s = envl.reset()

    length = 0

    while not d:
        length += 1
        ac = pi.select_action(s, evaluate=True)
        next_s, r, d, _ = envl.step(ac)

        if render:
            envl.render()

        rts += r
        s = next_s

    return rts, length


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

# Tensor board
# writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime(
# "%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
total_episode = 0
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)

                # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                # writer.add_scalar('loss/policy', policy_loss, updates)
                # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action)  # Step

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask)  # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, env_name:{}".format
          (i_episode, total_numsteps, episode_steps, round(episode_reward, 2), args.env_name))
    rets, lens = [], []
    for i in range(10):
        rt, le = traj_generator(agent, args, False)
        rets.append(rt)
        lens.append(le)
    print("std " + str(np.std(rets)))
    if np.mean(lens) > 970:
        agent.save_model(args.env_name, suffix=str(int(np.mean(rets))))
        print("save model mean return: {} mean length: {}".format(np.mean(rets), np.mean(lens)))


    # if i_episode % 10 == 0 and args.eval is True:
    #     avg_reward = 0.
    #     episodes = 10
    #     for _ in range(episodes):
    #         state = env.reset()
    #         episode_reward = 0
    #         done = False
    #         while not done:
    #             action = agent.select_action(state, evaluate=True)
    #
    #             next_state, reward, done, _ = env.step(action)
    #             episode_reward += reward
    #
    #             state = next_state
    #         avg_reward += episode_reward
    #     avg_reward /= episodes
    #
    #     # writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    #     total_episode += episodes
        # print("----------------------------------------")
        # # print("Test Episodes: {}, Avg. Reward: {} total_episode: {}".format(episodes, round(avg_reward, 2), total_episode))
        # # if 8000 < np.mean(rets) < 8500 and np.mean(lens) == 1000:
        # #     agent.save_model(args.env_name, suffix=str(int(np.mean(rets))))
        # print("Test Episodes: {}, Avg. Return: {} total_episode: {} Mean Length: {}".format(episodes,
        #                                                                                         round(np.mean(rets), 2),
        #                                                                                         total_episode,
        #                                                                                         np.mean(lens)))
        # print("----------------------------------------")

env.close()
print(args.actor_path, args.critic_path)
# save model
agent.save_model(args.env_name, suffix=str(int(avg_reward)) + "_new")
