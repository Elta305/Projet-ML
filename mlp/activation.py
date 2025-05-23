import numpy as np

from .module import Module


class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.tanh(x)

    def backward_delta(self, x, delta):
        fw = self.forward(x)
        return delta * (1.0 - fw**2)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward_delta(self, x, delta):
        fw = self.forward(x)
        return delta * fw * (1 - fw)


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def backward_delta(self, x, delta):
        fw = self.forward(x)
        return delta * fw * (1 - fw)


class LogSoftmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_max = x - np.max(x, axis=-1, keepdims=True)
        return x_max - np.log(np.sum(np.exp(x_max), axis=-1, keepdims=True))

    def backward_delta(self, x, delta):
        fw = self.forward(x)
        return delta - np.exp(fw) * np.sum(delta, axis=-1, keepdims=True)


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def backward_delta(self, x, delta):
        fw = self.forward(x)
        return delta * (fw > 0)
