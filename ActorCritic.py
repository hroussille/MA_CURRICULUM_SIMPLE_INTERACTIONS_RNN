import numpy as np
import torch

from torch import nn

class Actor(nn.Module):

    tanh = torch.nn.Tanh()

    def __init__(self, input_size, output_size, hidden_size = 128, hidden_act=torch.nn.LeakyReLU(), output_act=torch.nn.Tanh()):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.network = torch.nn.Sequential(
                                               torch.nn.Linear(input_size, hidden_size),
                                               hidden_act,
                                               torch.nn.Linear(hidden_size, hidden_size),
                                               hidden_act,
                                               torch.nn.Linear(hidden_size, output_size))
    def forward(self, input):
        return self.output_act(self.network(input))


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size = 128):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.network = torch.nn.Sequential(
                                           torch.nn.Linear(input_size, hidden_size),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(hidden_size, hidden_size),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(hidden_size, 1))
    def forward(self, input):
        return self.network(input)
