import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = np.array([])
        self._gradient = np.array([])

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
        self._parameters = np.random.randn(input_dim, output_dim)
        self._bias = np.random.randn(1, output_dim)
        self._gradient = np.zeros_like(self._parameters)
        self._gradient_bias = np.zeros_like(self._bias)

        if not self._has_bias:
            self._bias = None
            self._gradient_bias = None

    def forward(self, X):
        assert X.shape[1] == self._input_dim, "Input dimensions must match weight dimensions"
        if self._has_bias:
            return X @ self._parameters + self._bias
        return X @ self._parameters
    
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
    
    def update_parameters(self, learning_rate=1e-3):
        for module in self.modules:
            module.update_parameters(learning_rate)
    
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()
