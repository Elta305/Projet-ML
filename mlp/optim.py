import logging

import numpy as np

logger = logging.getLogger(__name__)


class Optim:
    """Base optimizer class that handles the forward/backward passes for
    training."""

    def __init__(self, net, loss, eps):
        """Initialize the optimizer with a network and loss function.

        Args:
            net: The neural network to optimize
            loss: The loss function to use for training
            eps: Learning rate
        """
        self.net = net
        self.loss = loss
        self.eps = eps
        logger.debug(f"Initialized base optimizer with learning rate {eps}")

    def step(self, batch_x, batch_y):
        """Perform a single optimization step.

        Args:
            batch_x: Input batch data
            batch_y: Target batch data

        Returns:
            Loss value for this batch
        """
        output = self.net.forward(batch_x)

        loss_val = self.loss.forward(batch_y, output)

        grad = self.loss.backward(batch_y, output)
        self.net.zero_grad()
        self.net.backward(grad)

        self._update_parameters()

        return loss_val

    def _update_parameters(self):
        """Update parameters using the optimizer's strategy.
        This method should be implemented by subclasses.
        """
        self.net.update_parameters(self.eps)


class SGD(Optim):
    """Standard Stochastic Gradient Descent optimizer."""

    def __init__(self, net, loss, eps=1e-3):
        super().__init__(net, loss, eps)
        logger.debug("Initialized SGD optimizer")

    def _update_parameters(self):
        self.net.update_parameters(self.eps)


class SGDMomentum(Optim):
    """SGD with momentum for faster convergence."""

    def __init__(self, net, loss, eps=1e-3, momentum=0.9):
        super().__init__(net, loss, eps)
        self.momentum = momentum
        self.velocity = {}

        # Initialize velocity for each module with parameters
        for module in self.net.modules:
            if hasattr(module, "parameters") and module.parameters is not None:
                self.velocity[module] = {
                    "parameters": np.zeros_like(module.parameters)
                }
                if hasattr(module, "bias") and module.bias is not None:
                    self.velocity[module]["bias"] = np.zeros_like(module.bias)

        logger.debug(f"Initialized SGD with momentum {momentum}")

    def _update_parameters(self):
        for module in self.net.modules:
            if hasattr(module, "parameters") and module.parameters is not None:
                # Update velocity and parameters
                self.velocity[module]["parameters"] = (
                    self.momentum * self.velocity[module]["parameters"]
                    - self.eps * module.gradient
                )
                module.parameters += self.velocity[module]["parameters"]

                # Update bias if it exists
                if hasattr(module, "bias") and module.bias is not None:
                    self.velocity[module]["bias"] = (
                        self.momentum * self.velocity[module]["bias"]
                        - self.eps * module.gradient_bias
                    )
                    module.bias += self.velocity[module]["bias"]


class Adam(Optim):
    """Adam optimizer with adaptive learning rates."""

    def __init__(
        self, net, loss, eps=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        super().__init__(net, loss, eps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = {}
        self.v = {}

        for module in self.net.modules:
            if hasattr(module, "parameters") and module.parameters is not None:
                self.m[module] = {
                    "parameters": np.zeros_like(module.parameters)
                }
                self.v[module] = {
                    "parameters": np.zeros_like(module.parameters)
                }

                if hasattr(module, "bias") and module.bias is not None:
                    self.m[module]["bias"] = np.zeros_like(module.bias)
                    self.v[module]["bias"] = np.zeros_like(module.bias)

        logger.debug(
            f"Initialized Adam optimizer with beta1={beta1}, beta2={beta2}"
        )

    def _update_parameters(self):
        self.t += 1

        for module in self.net.modules:
            if hasattr(module, "parameters") and module.parameters is not None:
                self.m[module]["parameters"] = (
                    self.beta1 * self.m[module]["parameters"]
                    + (1 - self.beta1) * module.gradient
                )
                self.v[module]["parameters"] = self.beta2 * self.v[module][
                    "parameters"
                ] + (1 - self.beta2) * np.square(module.gradient)

                m_hat = self.m[module]["parameters"] / (1 - self.beta1**self.t)
                v_hat = self.v[module]["parameters"] / (1 - self.beta2**self.t)

                module.parameters -= (
                    self.eps * m_hat / (np.sqrt(v_hat) + self.epsilon)
                )

                if hasattr(module, "bias") and module.bias is not None:
                    self.m[module]["bias"] = (
                        self.beta1 * self.m[module]["bias"]
                        + (1 - self.beta1) * module.gradient_bias
                    )
                    self.v[module]["bias"] = self.beta2 * self.v[module][
                        "bias"
                    ] + (1 - self.beta2) * np.square(module.gradient_bias)

                    m_bias_hat = self.m[module]["bias"] / (
                        1 - self.beta1**self.t
                    )
                    v_bias_hat = self.v[module]["bias"] / (
                        1 - self.beta2**self.t
                    )

                    module.bias -= (
                        self.eps
                        * m_bias_hat
                        / (np.sqrt(v_bias_hat) + self.epsilon)
                    )
