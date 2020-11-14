import numpy as np
import random

class Noise():

    def __init__(self, mu=0, sigma=1, output_shape=(1)):
        self.output_shape = output_shape
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        pass


class RandomNoise(Noise):

    def __init__(self, mu=0, sigma=1, output_shape=(1)):
        super(Noise, self).__init__(mu, sigma, output_shape)

    def sample(self):
        return

