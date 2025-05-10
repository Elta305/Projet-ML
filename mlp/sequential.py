import logging

from .linear import Linear
from .module import Module

logger = logging.getLogger(__name__)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        self.inputs = []

    def forward(self, x):
        self.inputs = [x]
        for module in self.modules:
            x = module.forward(x)
            self.inputs.append(x)

        return x

    def backward(self, delta):
        self.inputs.reverse()
        for i, module in enumerate(reversed(self.modules)):
            module.backward_update_gradient(self.inputs[i + 1], delta)
            delta = module.backward_delta(self.inputs[i + 1], delta)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def update_parameters(self, learning_rate):
        for module in self.modules:
            module.update_parameters(learning_rate)

    def state_dict(self):
        state = {}
        for module in self.modules:
            if isinstance(module, Linear):
                state[module] = {
                    "parameters": module.parameters.copy(),
                    "gradient": module.gradient.copy(),
                }
                if module.use_bias:
                    state[module]["bias"] = module.bias.copy()
                    state[module]["gradient_bias"] = module.gradient_bias.copy()
        return state

    def load_state_dict(self, state_dict):
        for module in self.modules:
            if isinstance(module, Linear):
                module.parameters = state_dict[module]["parameters"]
                module.gradient = state_dict[module]["gradient"]
                if module.use_bias:
                    module.bias = state_dict[module]["bias"]
                    module.gradient_bias = state_dict[module]["gradient_bias"]
