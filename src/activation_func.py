from module import Module
import numpy as np

class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)
    
    def backward_delta(self, input, delta):
        return delta * (1 - self.forward(input) ** 2)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1 / (1 + np.exp(-X))
    
    def backward_delta(self, input, delta):
        fw = self.forward(input)
        return delta * fw * (1 - fw)

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)
    
    def backward_delta(self, input, delta):
        fw = self.forward(input)
        return delta * fw * (1 - fw)

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.maximum(0, X)
    
    def backward_delta(self, input, delta):
        return delta * (self.forward(input) > 0)