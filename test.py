
import numpy as np
import torch
import time
import copy
import utils
from MADDPG import MADDPG
import random
import argparse
import yaml
import SimpleInteractions
import itertools

from ActorCritic import Actor

def get_learners_subpolicies(learner, env):
    return [random.randrange(0, learner.subpolicies) for _ in range(env.n)]

def load_config(path="config.yaml"):
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help="path to output folder")

    args = parser.parse_args()
    config = load_config(args.path + "/config.yaml")
    n_agents = len(config['env']['env_obs_learners'])
    with_finish_zone = config['env']['with_finish_zone']
    synchronized_activations = config['env']['synchronized_activations']

    env = SimpleInteractions.SimpleInteractions(n_agents=n_agents, with_finish_zone=with_finish_zone,
                                                synchronized_activation=synchronized_activations)

    learner = MADDPG(env, config['env']['env_obs_learners'], **config['learners'])
    learner.load(args.path + "/models_1/")

    affectations = []

    o = env.reset()
    reward = []

    subs = np.random.randint(0, len(config['env']['env_obs_learners']), config['learners']['subpolicies'])


    average_reward = []

    success_rate = 0
    success_time = []
    total = 0

    for _ in range(500):

        subs = np.random.randint(0, len(config['env']['env_obs_learners']), config['learners']['subpolicies'])
        state = env.reset()

        episode_reward = 0
        #subs = get_learners_subpolicies(learner, env)

        for step in range(50):

            with torch.no_grad():
                actors_input = state
                action = learner.act(actors_input, subs, noise=False)

            state, _, done, _ = env.step(action)

            r = int(done)
            episode_reward += r / (step + 1e-5)
            success_rate += r

            if done:
                break

        #average_reward.append(episode_reward)

            env.render(mode="human")
            time.sleep(0.1)

        total += 1

    print("SUB : {} Success rate : {}".format(subs, success_rate / total))
    #print("Average time to success : {}".format(np.mean(success_time)))

