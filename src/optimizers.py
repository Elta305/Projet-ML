import numpy as np
from tqdm import tqdm
from module import *

class Optim:
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        output = self.net.forward(batch_x)
        loss = np.mean(self.loss.forward(batch_y, output))
        self.net.zero_grad()
        delta = self.loss.backward(batch_y, output)
        self.net.backward(delta)
        self.net.update_parameters(self.eps)
        return loss

    def SGD(self, data_x, data_y, batch_size, num_iterations):
        losses = []
        for _ in tqdm(range(num_iterations)):
            indices = np.random.permutation(len(data_x))
            loss = []
            for i in range(0, len(data_x), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_x = data_x[batch_indices]
                batch_y = data_y[batch_indices]
                loss.append(self.step(batch_x, batch_y))
            loss = np.mean(loss)
            losses.append(loss)
        return losses

class AdamOptimizer:
    def __init__(self, net, loss_fn, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.net = net
        self.loss = loss_fn
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentums = {}
        self.caches = {}

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, module):
        if isinstance(module, Linear):
            if module not in self.momentums:
                self.momentums[module] = {
                    'parameters': np.zeros_like(module._parameters),
                    'bias': np.zeros_like(module._bias) if module._has_bias else None
                }
                self.caches[module] = {
                    'parameters': np.zeros_like(module._parameters),
                    'bias': np.zeros_like(module._bias) if module._has_bias else None
                }
            self.momentums[module]['parameters'] = self.beta_1 * self.momentums[module]['parameters'] + (1 - self.beta_1) * module._gradient
            if module._has_bias:
                self.momentums[module]['bias'] = self.beta_1 * self.momentums[module]['bias'] + (1 - self.beta_1) * module._gradient_bias
            
            corrected_momentums_parameters = self.momentums[module]['parameters'] / (1 - self.beta_1 ** (self.iterations + 1))
            corrected_momentums_bias = None
            if module._has_bias:
                corrected_momentums_bias = self.momentums[module]['bias'] / (1 - self.beta_1 ** (self.iterations + 1))
            
            self.caches[module]['parameters'] = self.beta_2 * self.caches[module]['parameters'] + (1 - self.beta_2) * (module._gradient ** 2)
            if module._has_bias:
                self.caches[module]['bias'] = self.beta_2 * self.caches[module]['bias'] + (1 - self.beta_2) * (module._gradient_bias ** 2)

            corrected_caches_parameters = self.caches[module]['parameters'] / (1 - self.beta_2 ** (self.iterations + 1))
            corrected_caches_bias = None
            if module._has_bias:
                corrected_caches_bias = self.caches[module]['bias'] / (1 - self.beta_2 ** (self.iterations + 1))

            module._parameters -= self.current_learning_rate * corrected_momentums_parameters / (np.sqrt(corrected_caches_parameters) + self.epsilon)
            if module._has_bias:
                module._bias -= self.current_learning_rate * corrected_momentums_bias / (np.sqrt(corrected_caches_bias) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

    def step(self, batch_x, batch_y):
        output = self.net.forward(batch_x)
        loss = np.mean(self.loss.forward(batch_y, output))
        self.net.zero_grad()
        delta = self.loss.backward(batch_y, output)
        self.net.backward(delta)
        for module in self.net.modules:
            self.update_params(module)
        return loss

    def SGD(self, data_x, data_y, batch_size, num_iterations):
        losses = []
        for _ in tqdm(range(num_iterations)):
            indices = np.random.permutation(len(data_x))
            loss = []
            for i in range(0, len(data_x), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_x = data_x[batch_indices]
                batch_y = data_y[batch_indices]
                self.pre_update_params()
                loss.append(self.step(batch_x, batch_y))
                self.post_update_params()
            loss = np.mean(loss)
            losses.append(loss)
        return losses