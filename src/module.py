import numpy as np

class Module(object):
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
    def __init__(self, input_dim, output_dim, has_bias=True):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._has_bias = has_bias

        std_dev = np.sqrt(2 / self._input_dim)
        self._parameters = np.random.normal(0, std_dev, (self._input_dim, self._output_dim))
        self._gradient = np.zeros_like(self._parameters)

        if self._has_bias:
            self._bias = np.random.normal(0, std_dev, (1, self._output_dim))
            self._gradient_bias = np.zeros_like(self._bias)

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
        for module in self.modules:
            if isinstance(module, Linear):
                state[module] = {
                    '_parameters': module._parameters.copy(),
                    '_gradient': module._gradient.copy()
                }
                if module._has_bias:
                    state[module]['_bias'] = module._bias.copy()
                    state[module]['_gradient_bias'] = module._gradient_bias.copy()
        return state

    def load_state_dict(self, state_dict):
        for module in self.modules:
            if isinstance(module, Linear):
                module._parameters = state_dict[module]['_parameters']
                module._gradient = state_dict[module]['_gradient']
                if module._has_bias:
                    module._bias = state_dict[module]['_bias']
                    module._gradient_bias = state_dict[module]['_gradient_bias']

class AutoEncoder:
    def __init__(self, encoder, decoder, loss_fn):
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, x):
        z = self.encoder.forward(x)
        x_hat = self.decoder.forward(z)
        return x_hat

    def backward(self, x, x_hat):
        loss = self.loss_fn.forward(x, x_hat)
        dloss = self.loss_fn.backward(x, x_hat)
        dz = self.decoder.backward(dloss)
        self.encoder.backward(dz)
        return loss

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def encode(self, x):
        return self.encoder.forward(x)
    
    def update_parameters(self, learning_rate=1e-3):
        self.encoder.update_parameters(learning_rate)
        self.decoder.update_parameters(learning_rate)