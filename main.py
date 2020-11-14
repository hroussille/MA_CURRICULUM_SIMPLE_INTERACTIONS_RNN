import numpy as np
import argparse
from MACuriculum import MACuriculum
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import SimpleInteractions
import errno
import yaml
import torch

def set_mode(mode):
    if mode == "GPU":
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print("Set GPU mode")
        else:
            print("GPU mode unavailable, fallback to CPU")
    else:
        print("Set CPU mode")

def load_config(path="config.yaml"):
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config

def save_config(config, path):

    with open(path, 'w') as file:
        documents = yaml.dump(config, file)

def train(config, path):
    n_train = config['training']['n_train']
    n_agents = len(config['env']['env_obs_learners'])
    with_finish_zone = config['env']['with_finish_zone']
    synchronized_activations = config['env']['synchronized_activations']
    means = []
    stds = []

    for i in range(n_train):
        print("Training : {} on {}".format(i + 1, n_train))
        env = SimpleInteractions.SimpleInteractions(n_agents=n_agents, with_finish_zone=with_finish_zone, synchronized_activation=synchronized_activations)
        learner = MACuriculum(env, writer, i + 1, config, path)
        mean, std = learner.run()
        means.append(mean)
        stds.append(std)

    averages = np.mean(means, axis=0)
    deviations = np.mean(stds, axis=0)

    return np.array(averages), (deviations)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="config.yaml", help="path to config file")
    parser.add_argument('--output', type=str, required=True, help="path to output folder")

    args = parser.parse_args()
    config = load_config(args.config)

    set_mode(config['training']['mode'])

    try:
        os.makedirs(args.output)

        for i in range(config['training']['n_train']):
            os.makedirs(args.output + "/models_{}".format(i + 1))

    except OSError as e:
        if e.errno == errno.EEXIST:
            print("Output folder : {} already exist".format(args.output))
            exit()

    writer = SummaryWriter(args.output + "/Summary")
    save_config(config, args.output + "/config.yaml")

    path = args.output + "/"
    averages, deviations = train(config, path)

    writer.close()

    np.save(args.output + "/averages", averages)
    np.save(args.output + "/deviations", deviations)