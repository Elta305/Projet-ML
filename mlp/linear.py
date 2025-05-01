import numpy as np
import logging
from .module import Module

logger = logging.getLogger(__name__)


class Linear(Module):
    def __init__(self, input_size, output_size, use_bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias

        weight_scale = np.sqrt(1.0 / input_size)
        self._parameters = (
            np.random.randn(input_size, output_size) * weight_scale
        )
        self._gradient = np.zeros((input_size, output_size))

        if self.use_bias:
            self.bias = np.zeros((1, output_size))
            self.bias_gradient = np.zeros((1, output_size))
        else:
            self.bias = None
            self.bias_gradient = None

        logger.debug(
            f"Initialized linear layer with input size: {input_size}, output size: {output_size}, bias: {use_bias}"
        )

    def forward(self, input_data):
        assert input_data.shape[1] == self.input_size, (
            f"Input dimension mismatch: got {input_data.shape[1]}, expected {self.input_size}"
        )
        output = np.dot(input_data, self._parameters)
        if self.use_bias:
            output += self.bias

        logger.debug(
            f"Forward pass though linear layer: input shape: {input_data.shape}, output shape: {output.shape}"
        )
        return output

    def zero_grad(self):
        self._gradient.fill(0.0)
        if self.use_bias:
            self.bias_gradient.fill(0.0)
        logger.debug("Zeroed gradients in linear layer")

    def backward_update_gradient(self, input_data, delta):
        assert input_data.shape[1] == self.input_size, (
            f"Input dimension mismatch in backward: got {input_data.shape[1]}, expected {self.input_size}"
        )
        assert delta.shape[1] == self.output_size, (
            f"Delta dimension mismatch: got {delta.shape[1]}, expected {self.output_size}"
        )

        weight_gradient = np.dot(input_data.T, delta)
        self._gradient += weight_gradient

        if self.use_bias:
            bias_gradient = np.sum(delta, axis=0, keepdims=True)
            self.bias_gradient += bias_gradient

        logger.debug(
            f"Updated gradients in linear layer: input shape {input_data.shape}, delta shape {delta.shape}"
        )

    def backward_delta(self, input_data, delta):
        assert input_data.shape[1] == self.input_size, (
            f"Input dimension mismatch in backward_delta: got {input_data.shape[1]}, expected {self.input_size}"
        )
        assert delta.shape[1] == self.output_size, (
            f"Delta dimension mismatch: got {delta.shape[1]}, expected {self.output_size}"
        )

        input_grad = np.dot(delta, self._parameters.T)
        logger.debug(
            f"Computed input gradient in Linear layer: delta shape {delta.shape}, result shape {input_grad.shape}"
        )
        return input_grad

    def update_parameters(self, learning_rate=1e-3):
        gradient_clip_value = 1.0

        np.clip(
            self._gradient,
            -gradient_clip_value,
            gradient_clip_value,
            out=self._gradient,
        )
        self._parameters -= learning_rate * self._gradient

        if self.use_bias:
            np.clip(
                self.bias_gradient,
                -gradient_clip_value,
                gradient_clip_value,
                out=self.bias_gradient,
            )
            self.bias -= learning_rate * self.bias_gradient
        logger.debug(
            f"Updated parameters in linear layer with learning rate {learning_rate}"
        )
