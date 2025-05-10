# We base ourselves on the EfficientNet coefficients in order to have well parameterized networks.
# alpha^N, beta^N, gamma^N are the coefficients for the number of channels, depth and resolution respectively.

from activation_func import *
from module import *


def init_network(input_size, output_size, depth, width, resolution, middle_activation=ReLU, last_activation=Softmax):
    layers = []
    for i in range(depth):
        layers.append(Linear(input_size, width))
        layers.append(middle_activation)
        input_size = width

    layers.append(Linear(width, output_size))
    layers.append(last_activation)

    return Sequential(layers)

def create_networks(input_size, output_size, alpha=1.2, beta=1.1, gamma=1.15, phi_values=[1, 2, 3, 4, 5]):
    networks = []
    for phi in phi_values:
        width = int(32 * alpha ** phi)
        depth = int(3 * beta ** phi)
        resolution = int(224 * gamma ** phi)
        networks.append(init_network(input_size, output_size, depth, width, resolution))
    return networks
