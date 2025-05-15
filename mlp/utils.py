import numpy as np


def one_hot_encoding(y, num_classes):
    """Convert class indices to one-hot encoding.

    Args:
        y: Array of class indices (shape: n_samples,)
        num_classes: Number of classes

    Returns:
        One-hot encoded array (shape: n_samples, num_classes)
    """
    n_samples = len(y)
    one_hot = np.zeros((n_samples, num_classes))
    for i in range(n_samples):
        one_hot[i, int(y[i])] = 1
    return one_hot
