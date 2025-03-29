import numpy as np
import logging

logger = logging.getLogger(__name__)


class Loss(object):
    """
    Loss represents a loss function that measures the difference beween
    predicted values and target values in a neural network.
    """

    def __init__(self):
        logger.debug("Intitialized base loss")

    def forward(self, targets, predictions):
        """
        forward computes the loss value between true labels and predictions.

        * y: true labels/targets
        * yhat: predicted values from the model
        """
        raise NotImplementedError(
            "forward method must be implemented by subclasses"
        )

    def backward(self, targets, predictions):
        """
        backward computes the gradient of the loss with respect to model
        predictions. This gradient is the starting point for backpropagation
        through the network.

        * y: true labels/targets
        * yhat: predicted values from the models
        """
        raise NotImplementedError(
            "backward method must be implemented by subclasses"
        )


class MSELoss(Loss):
    def __init__(self):
        super().__init__()
        logger.debug("Initialized MSELoss")

    def forward(self, targets, predictions):
        assert targets.shape == predictions.shape, (
            "Targets and predictions must have the same shape"
        )
        batch_size = targets.shape[0]
        squared_errors = np.sum((targets - predictions) ** 2, axis=1)
        logger.debug(f"Computed MSE for batch size {batch_size}")
        return squared_errors

    def backward(self, targets, predictions):
        assert targets.shape == predictions.shape, (
            "Targets and predictions must have the same shape"
        )
        batch_size = targets.shape[0]
        gradient = -2 * (targets - predictions)
        logger.debug(f"Computed MSE gradient for batch size {batch_size}")
        return gradient


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, "Shapes of y and yhat must match"
        return 1 - (yhat * y).sum(axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, "Shapes of y and yhat must match"
        return yhat - y
