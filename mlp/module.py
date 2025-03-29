import logging

logger = logging.getLogger(__name__)


class Module(object):
    """
    Module represents a generic neural network layer with parameters, forward
    propagation, and backpropagation capabilities. This abstract class serves
    as the foundation for specific layer implementations in the neural network.
    """

    def __init__(self):
        """
        Initialize a new Module instance with empty parameters and gradient.
        The specific parameters will be defined in child classes.
        """
        self._parameters = None
        self._gradient = None
        logger.debug("Initialized base module")

    def forward(self, x):
        """
        forward performs the forward pass computation through this module.

        * x: input data to the module
        """
        raise NotImplementedError(
            "forward method must be implemented by subclasses"
        )

    def zero_grad(self):
        """
        zero_grad resets the accumulated gradient to zero, typically called
        before a new backprobagation pass to clear previous gradients.
        """
        raise NotImplementedError(
            "zero_grad method must be implemented by subclasses"
        )

    def backward_update_gradient(self, input, delta):
        """
        backward_update_gradient computes and accumulates the gradient of the
        loss with respect to this module's parameters.

        * input : the original input that was passed to this module during
        forward pass
        * delta: the gradient of the loss with respect to the module's ouput
        """
        raise NotImplementedError(
            "backward_update_gradient method must be implemented by subclasses"
        )

    def backward_delta(self, input, delta):
        """
        backward_delta computes the gradient of the loss with respect to the
        module's input. This gradient will be passed to the previous layer in
        the network.

        * input: the original input that was passed to this module during
        forward pass
        * delta: the gradient of the loss with respect to the module's output
        pass
        """
        raise NotImplementedError(
            "backward_delta method must be implemented by subclasses"
        )

    def update_parameters(self, learning_rate=1e-3):
        """
        update_parameters adjusts the module's parameters using the accumulated
        gradient and specific learning rate. This method implements the
        gradient descent update step.
        """
        self._parameters -= learning_rate * self._gradient
        logger.debug(f"Updated parameters with learning rate {learning_rate}")
