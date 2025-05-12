import logging

import numpy as np

logger = logging.getLogger(__name__)


class Loss:
    """Loss represents a loss function that measures the difference beween
    predicted values and target values in a neural network.
    """

    def __init__(self):
        logger.debug("Intitialized base loss")

    def forward(self, targets, predictions):
        """Forward computes the loss value between true labels and predictions.

        * y: true labels/targets
        * yhat: predicted values from the model
        """
        error_msg = "forward method must be implemented by subclasses"
        raise NotImplementedError(error_msg)

    def backward(self, targets, predictions):
        """Backward computes the gradient of the loss with respect to model
        predictions. This gradient is the starting point for backpropagation
        through the network.

        * y: true labels/targets
        * yhat: predicted values from the models
        """
        error_msg = "backward method must be implemented by subclasses"
        raise NotImplementedError(error_msg)


class MSELoss(Loss):
    def __init__(self):
        super().__init__()
        logger.debug("Initialized MSELoss")

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, (
            "Targets and predictions must have the same shape"
        )
        batch_size = y.shape[0]

        mse = np.mean(np.square(y - yhat))
        logger.debug(f"Computed MSE for batch size {batch_size}")

        return mse

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, (
            "Targets and predictions must have the same shape"
        )
        batch_size = y.shape[0]
        gradient = -2 * (y - yhat)
        logger.debug(f"Computed MSE gradient for batch size {batch_size}")
        return gradient


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, (
            "Targets and predictions must have the same shape"
        )
        return 1 - (yhat * y).sum(axis=1)

    def backward(self, targets, predictions):
        assert targets.shape == predictions.shape, (
            "Targets and predictions must have the same shape"
        )
        return predictions - targets
