import numpy as np

from module import Module


class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_delta(self, input, delta):
        return delta * (1 - self.forward(input) ** 2)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_delta(self, input, delta):
        fw = self.forward(input)
        return delta * fw * (1 - fw)

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_delta(self, input, delta):
        fw = self.forward(input)
        return delta * fw * (1 - fw)

class LogSoftmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X_max = X - np.max(X, axis=-1, keepdims=True)
        return X_max - np.log(np.sum(np.exp(X_max), axis=-1, keepdims=True))

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_delta(self, input, delta):
        fw = self.forward(input)
        return delta - np.exp(fw) * np.sum(delta, axis=-1, keepdims=True)

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.maximum(0, X)

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_delta(self, input, delta):
        return delta * (input > 0)
