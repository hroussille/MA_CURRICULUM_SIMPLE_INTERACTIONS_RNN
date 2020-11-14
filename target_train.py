import numpy as np
import MADDPG
import argparse
import yaml
import copy
import random

from SimpleInteractions import SimpleInteractions
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

def load_config(path="config.yaml"):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config

def get_learners_subpolicies(env, maddpg):
    return [random.randrange(0, maddpg.subpolicies) for _ in range(env.n)]

def explore(env, maddpg):
    step_count = 0
    Done = False
    timestep = 0

    s = env.reset()

    while timestep < max_timestep and not (Done):
        subs = get_learners_subpolicies(env, maddpg)
        timestep = timestep + 1
        actions_detached = maddpg.act(s, subs)
        s_t, r, Done, i = env.step(copy.deepcopy(actions_detached))

        if timestep >= max_timestep:
            Done = True

        r = [-1 * 0.1 * int(Done) * timestep] * env.n

        maddpg.push_sample(s, actions_detached, r, Done, s_t, subs)

        s = s_t


def train(env, maddpg, max_timestep, episode, writer):
    timestep = 1
    total_reward = 0

    subs = get_learners_subpolicies(env, maddpg)
    s = env.reset()
    Done = False

    while timestep <= max_timestep and not Done:
        timestep = timestep + 1

        input = s
        actions_detached = maddpg.act(input, subs)
        s_t, r, Done, _ = env.step(copy.deepcopy(actions_detached))

        total_reward += int(Done)

        if timestep > max_timestep:
            Done = True

        r = [-1 * 0.1 * int(Done) * timestep] * env.n

        maddpg.push_sample(s, actions_detached, r, Done, s_t, subs)
        s = s_t

        maddpg.train(subs)

    writer.add_scalars('Target play reward', {'Reward': total_reward}, episode)

def test(env, maddpg, n_episodes, max_episode_timestep):
    results = []

    for episode in range(n_episodes):

        episode_reward = 0

        done = False
        timestep = 0
        subs = get_learners_subpolicies(env, maddpg)

        s = env.reset()

        while timestep < max_episode_timestep and not done:
            timestep = timestep + 1
            actions = []

            actions_detached = maddpg.act(s, subs, noise=False)
            s_t, r, done, _ = env.step(copy.deepcopy(actions_detached))

            if timestep >= max_episode_timestep:
                done = True

            episode_reward += r[0]
            s = s_t

        results.append(episode_reward)

    return np.mean(results, axis=0) , np.std(results, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="runs/final/config.yaml", help="path to config file")
    parser.add_argument('--output', type=str, required=True, help="path to output folder")

    args = parser.parse_args()

    config = load_config(args.config)

    print(config['target_play'])

    n_agents = len(config['env']['env_obs_learners'])
    with_finish_zone = config['env']['with_finish_zone']
    synchronized_activations = config['env']['synchronized_activations']

    env = SimpleInteractions(n_agents=n_agents, with_finish_zone=with_finish_zone, synchronized_activation=synchronized_activations)

    config['learners']['replay_buffer_type'] = 'per'
    config['learners']['replay_buffer_size'] = 500000
    config['learners']['batch_size'] = 32
    config['learners']['lr_actor'] = 0.00001
    config['learners']['lr_critic'] = 0.00001
    config['learners']['noise'] = 0.01

    maddpg = MADDPG.MADDPG(env, config['env']['env_obs_learners'], **config['learners'])

    """
    for agent_rb in maddpg.replay_buffers:
        for rb in agent_rb:
            rb.beta = 1
    
    """

    maddpg.load(args.output + "/models_1")
    #maddpg.load_optim(args.output + "/models_1")

    writer = SummaryWriter(args.output + "/Summary_target_play_2")

    n_exploration = config['target_play']['exploration_episodes']
    n_train = config['target_play']['episodes']
    test_freq = config['target_play']['test_freq']
    test_episodes = config['target_play']['test_episodes']
    max_timestep = config['target_play']['max_timestep']

    print(args.config)
    print(n_exploration)
    print(n_train)

    increment = 1 / n_train

    #actors_schedulers = [torch.optim.lr_scheduler.StepLR(maddpg.actors_optimizers[i][j], 20000, gamma=0.5) for i in range(len(maddpg.env_obs)) for j in range(maddpg.subpolicies)]
    #critics_schedulers = [torch.optim.lr_scheduler.StepLR(maddpg.critics_optimisers[i], 20000, gamma=0.5) for i in range(len(maddpg.env_obs))]

    for episode in tqdm(range(n_exploration)):
        explore(env, maddpg)

    for episode in tqdm(range(n_train)):
        train(env, maddpg, max_timestep, episode, writer)

        maddpg.increment_per(increment)

        if episode % 100:
            maddpg.apply_noise_decay()

        #for scheduler in actors_schedulers + critics_schedulers:
        #    scheduler.step()

        if episode % test_freq == 0:
            mean, std = test(env, maddpg, test_episodes, max_timestep)
            writer.add_scalars("Target_play", {'average reward': np.mean(mean)}, episode)

        maddpg.apply_noise_decay()
