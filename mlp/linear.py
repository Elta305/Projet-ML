import logging

import numpy as np

from .module import Module

logger = logging.getLogger(__name__)


class Linear(Module):
    def __init__(self, input_size, output_size, init="he_normal", use_bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias

        self._parameters, self._gradient, self._bias, self._gradient_bias = self.init_parameters(input_size, output_size, init, use_bias)
        
        logger.debug(
            f"Initialized linear layer with "
            f"input size: {input_size}, "
            f"output size: {output_size}, "
            f"bias: {use_bias}"
        )

    def init_parameters(self, input_size, output_size, init, use_bias):
        if init == "random":
            parameters = np.random.rand(input_size, output_size)
            gradient = np.zeros_like(parameters)
            bias = np.random.rand(1, output_size) if use_bias else None
            gradient_bias = np.zeros_like(bias) if use_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "normal":
            parameters = np.random.randn(input_size, output_size)
            gradient = np.zeros_like(parameters)
            bias = np.random.randn(1, output_size) if use_bias else None
            gradient_bias = np.zeros_like(bias) if use_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "uniform":
            limit = np.sqrt(6 / (input_size + output_size))
            parameters = np.random.uniform(-limit, limit, (input_size, output_size))
            gradient = np.zeros_like(parameters)
            bias = np.random.uniform(-limit, limit, (1, output_size)) if use_bias else None
            gradient_bias = np.zeros_like(bias) if use_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "he_normal":
            std_dev = np.sqrt(2 / self._input_size)
            parameters = np.random.normal(0, std_dev, (input_size, output_size))
            gradient = np.zeros_like(parameters)
            bias = np.random.normal(0, std_dev, (1, output_size)) if use_bias else None
            gradient_bias = np.zeros_like(bias) if use_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "he_uniform":
            limit = np.sqrt(6 / (input_size + output_size))
            parameters = np.random.uniform(-limit, limit, (input_size, output_size))
            gradient = np.zeros_like(parameters)
            bias = np.random.uniform(-limit, limit, (1, output_size)) if use_bias else None
            gradient_bias = np.zeros_like(bias) if use_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "xavier_normal":
            std_dev = np.sqrt(2 / (input_size + output_size))
            parameters = np.random.normal(0, std_dev, (input_size, output_size))
            gradient = np.zeros_like(parameters)
            bias = np.random.normal(0, std_dev, (1, output_size)) if use_bias else None
            gradient_bias = np.zeros_like(bias) if use_bias else None
            return parameters, gradient, bias, gradient_bias
        if init == "xavier_uniform":
            limit = np.sqrt(6 / (input_size + output_size))
            parameters = np.random.uniform(-limit, limit, (input_size, output_size))
            gradient = np.zeros_like(parameters)
            bias = np.random.uniform(-limit, limit, (1, output_size)) if use_bias else None
            gradient_bias = np.zeros_like(bias) if use_bias else None
            return parameters, gradient, bias, gradient_bias

    def forward(self, x):
        assert x.shape[1] == self.input_size, (
            f"Input dimension mismatch: "
            f"got {x.shape[1]}, "
            f"expected {self.input_size}"
        )

        output = x @ self.parameters

        if self.use_bias:
            output += self.bias

        logger.debug(
            f"Forward pass though linear layer: "
            f"input shape: {x.shape}, "
            f"output shape: {output.shape}"
        )

        return output

    def zero_grad(self):
        self.gradient.fill(0.0)

        if self.use_bias:
            self.gradient_bias.fill(0.0)

        logger.debug("Zeroed gradients in linear layer")

    def backward_update_gradient(self, x, delta):
        assert x.shape[1] == self.input_size, (
            f"Input dimension mismatch in backward: "
            f"got {x.shape[1]}, "
            f"expected {self.input_size}"
        )
        assert delta.shape[1] == self.output_size, (
            f"Delta dimension mismatch: "
            f"got {delta.shape[1]}, "
            f"expected {self.output_size}"
        )

        self.gradient += x.T @ delta

        if self.use_bias:
            self.gradient_bias += np.sum(delta, axis=0)

        logger.debug(
            f"Updated gradients in linear layer: "
            f"input shape {x.shape}, "
            f"delta shape {delta.shape}"
        )

    def backward_delta(self, x, delta):
        assert x.shape[1] == self.input_size, (
            f"Input dimension mismatch in backward_delta: "
            f"got {x.shape[1]}, "
            f"expected {self.input_size}"
        )
        assert delta.shape[1] == self.output_size, (
            f"Delta dimension mismatch: "
            f"got {delta.shape[1]}, "
            f"expected {self.output_size}"
        )

        input_grad = delta @ self.parameters.T

        logger.debug(
            f"Computed input gradient in Linear layer: "
            f"delta shape {delta.shape}, "
            f"result shape {input_grad.shape}"
        )
        return input_grad

    def update_parameters(self, learning_rate):
        self.parameters -= learning_rate * self.gradient

        if self.use_bias:
            self.bias -= learning_rate * self.gradient_bias

        logger.debug(
            f"Updated parameters in linear layer: learning rate {learning_rate}"
        )
