import numpy as np


class Module:
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        pass

    def forward(self, X):
        pass

    def update_parameters(self, learning_rate=1e-3):
        self._parameters -= learning_rate * self._gradient

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass

class Linear(Module):
    def __init__(self, input_dim, output_dim, init="he_normal", has_bias=True):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._has_bias = has_bias
        self._parameters, self._gradient, self._bias, self._gradient_bias = self.init_parameters(input_dim, output_dim, init, has_bias)

    def init_parameters(self, input_dim, output_dim, init, has_bias):
        if init == "random":
            parameters = np.random.rand(input_dim, output_dim)
            gradient = np.zeros_like(parameters)
            bias = np.random.rand(1, output_dim) if has_bias else None
            gradient_bias = np.zeros_like(bias) if has_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "normal":
            parameters = np.random.randn(input_dim, output_dim)
            gradient = np.zeros_like(parameters)
            bias = np.random.randn(1, output_dim) if has_bias else None
            gradient_bias = np.zeros_like(bias) if has_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "uniform":
            limit = np.sqrt(6 / (input_dim + output_dim))
            parameters = np.random.uniform(-limit, limit, (input_dim, output_dim))
            gradient = np.zeros_like(parameters)
            bias = np.random.uniform(-limit, limit, (1, output_dim)) if has_bias else None
            gradient_bias = np.zeros_like(bias) if has_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "he_normal":
            std_dev = np.sqrt(2 / self._input_dim)
            parameters = np.random.normal(0, std_dev, (input_dim, output_dim))
            gradient = np.zeros_like(parameters)
            bias = np.random.normal(0, std_dev, (1, output_dim)) if has_bias else None
            gradient_bias = np.zeros_like(bias) if has_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "he_uniform":
            limit = np.sqrt(6 / (input_dim + output_dim))
            parameters = np.random.uniform(-limit, limit, (input_dim, output_dim))
            gradient = np.zeros_like(parameters)
            bias = np.random.uniform(-limit, limit, (1, output_dim)) if has_bias else None
            gradient_bias = np.zeros_like(bias) if has_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "xavier_normal":
            std_dev = np.sqrt(2 / (input_dim + output_dim))
            parameters = np.random.normal(0, std_dev, (input_dim, output_dim))
            gradient = np.zeros_like(parameters)
            bias = np.random.normal(0, std_dev, (1, output_dim)) if has_bias else None
            gradient_bias = np.zeros_like(bias) if has_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "xavier_uniform":
            limit = np.sqrt(6 / (input_dim + output_dim))
            parameters = np.random.uniform(-limit, limit, (input_dim, output_dim))
            gradient = np.zeros_like(parameters)
            bias = np.random.uniform(-limit, limit, (1, output_dim)) if has_bias else None
            gradient_bias = np.zeros_like(bias) if has_bias else None
            return parameters, gradient, bias, gradient_bias

    def forward(self, X):
        assert X.shape[1] == self._input_dim, "Input dimensions must match weight dimensions"
        self.output = X @ self._parameters
        if self._has_bias:
            return self.output + self._bias
        return self.output

    def zero_grad(self):
        self._gradient.fill(0)
        if self._has_bias:
            self._gradient_bias.fill(0)

    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self._input_dim, "Input dimensions must match weight dimensions"
        assert delta.shape[1] == self._output_dim, "Delta dimensions must match weight dimensions"
        self._gradient += input.T @ delta
        if self._has_bias:
            self._gradient_bias += np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
        assert input.shape[1] == self._input_dim, "Input dimensions must match weight dimensions"
        assert delta.shape[1] == self._output_dim, "Delta dimensions must match weight dimensions"
        return delta @ self._parameters.T

    def update_parameters(self, learning_rate=1e-3):
        self._parameters -= learning_rate * self._gradient
        if self._has_bias:
            self._bias -= learning_rate * self._gradient_bias

class Sequential:
    def __init__(self, *modules):
        self.modules = modules
        self.inputs = []

    def get_parameters(self):
        parameters = []
        for module in self.modules:
            if isinstance(module, Linear):
                parameters.append(module._parameters)
                if module._has_bias:
                    parameters.append(module._bias)
        return parameters

    def get_gradient(self):
        gradients = []
        for module in self.modules:
            if isinstance(module, Linear):
                gradients.append(module._gradient)
                if module._has_bias:
                    gradients.append(module._gradient_bias)
        return gradients

    def forward(self, X):
        self.inputs = [X]
        for module in self.modules:
            X = module.forward(X)
            self.inputs.append(X)
        return X

    def backward(self, delta):
        self.inputs.reverse()
        for i, module in enumerate(reversed(self.modules)):
            module.backward_update_gradient(self.inputs[i+1], delta)
            delta = module.backward_delta(self.inputs[i+1], delta)
        return delta

    def update_parameters(self, learning_rate=1e-3):
        for module in self.modules:
            module.update_parameters(learning_rate)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def state_dict(self):
        state = {}
        for idx, module in enumerate(self.modules):
            if isinstance(module, Linear):
                state[idx] = {
                    '_parameters': module._parameters.copy(),
                    '_gradient': module._gradient.copy()
                }
                if module._has_bias:
                    state[idx]['_bias'] = module._bias.copy()
                    state[idx]['_gradient_bias'] = module._gradient_bias.copy()
        return state

    def load_state_dict(self, state_dict):
        for idx, module in enumerate(self.modules):
            if isinstance(module, Linear):
                module._parameters = state_dict[idx]['_parameters']
                module._gradient = state_dict[idx]['_gradient']
                if module._has_bias:
                    module._bias = state_dict[idx]['_bias']
                    module._gradient_bias = state_dict[idx]['_gradient_bias']

class AutoEncoder:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder.forward(x)
        x_hat = self.decoder.forward(z)
        return x_hat

    def backward(self, delta):
        delta = self.decoder.backward_delta(self.encoder.inputs[-1], delta)
        self.decoder.backward_update_gradient(self.encoder.inputs[-1], delta)
        delta = self.encoder.backward_delta(self.encoder.inputs[-2], delta)
        self.encoder.backward_update_gradient(self.encoder.inputs[-2], delta)
        return delta

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def encode(self, x):
        return self.encoder.forward(x)

    def update_parameters(self, learning_rate=1e-3):
        self.encoder.update_parameters(learning_rate)
        self.decoder.update_parameters(learning_rate)
