import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, "Shapes of y and yhat must match"
        return np.linalg.norm(y - yhat) ** 2
    
    def backward(self, y, yhat):
        assert y.shape == yhat.shape, "Shapes of y and yhat must match"
        return -2 * (y - yhat)

class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape"
        return 1 - (yhat * y).sum(axis=1)
    
    def backward(self, y, yhat):
        return yhat - y
