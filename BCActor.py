import torch
from torch import nn

class BCActor(nn.Module):

    tanh = torch.nn.Tanh()

    def __init__(self, input_size, output_size, hidden_size = 128, hidden_act=torch.nn.LeakyReLU(), output_act=torch.nn.Tanh()):
        super(BCActor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.network = torch.nn.Sequential(
                                               torch.nn.Linear(input_size, hidden_size),
                                               hidden_act,
                                               torch.nn.Dropout(0.25),
                                               torch.nn.Linear(hidden_size, 2 * hidden_size),
                                               hidden_act,
                                               torch.nn.Dropout(0.25),
                                               torch.nn.Linear(2 * hidden_size, output_size))
    def forward(self, input):
        return self.output_act(self.network(input))
