import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute statistical measures from an array of values."""
    values = np.array(values)
    q1, q3 = np.percentile(values, [25, 75])
    mask = (values >= q1) & (values <= q3)
    interquartile_values = values[mask]
    iqm = np.mean(interquartile_values)
    mins = np.min(values)
    maxs = np.max(values)
    return {"iqm": iqm, "q1": q1, "q3": q3, "min": mins, "max": maxs}


def get_batches(x, y, batch_size):
    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)

    x_shuffled = x[indices]
    y_shuffled = y[indices]

    n_batches = int(np.ceil(n_samples / batch_size))
    batches = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        batch_x = x_shuffled[start_idx:end_idx]
        batch_y = y_shuffled[start_idx:end_idx]

        batches.append((batch_x, batch_y))

    return batches


def generate_linear_data(n_samples=1000, noise=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.random.rand(n_samples, 1) * 10
    true_weight = np.random.uniform(-5, 5)
    true_bias = np.random.uniform(-10, 10)
    y = true_weight * x + true_bias + noise * np.random.randn(n_samples, 1)

    return x, y, true_weight, true_bias


def generate_classification_data(
    n_samples=1000, n_features=2, n_classes=2, separation=1.0, seed=None
):
    """Generate synthetic data for classification with optional random seed."""
    if seed is not None:
        np.random.seed(seed)

    centers = np.random.uniform(-5, 5, size=(n_classes, n_features))

    for i in range(n_classes):
        centers[i] = centers[i] * separation

    x = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, 1))

    samples_per_class = n_samples // n_classes
    for i in range(n_classes):
        start_idx = i * samples_per_class
        end_idx = (
            start_idx + samples_per_class if i < n_classes - 1 else n_samples
        )

        class_samples = end_idx - start_idx
        x[start_idx:end_idx] = centers[i] + np.random.randn(
            class_samples, n_features
        )
        y[start_idx:end_idx] = i

    indices = np.random.permutation(n_samples)
    x = x[indices]
    y = y[indices]

    y = 2 * y - 1

    return x, y, centers


def generate_xor_classification_data(
    n_samples=1000, n_features=2, n_classes=2, separation=1.0, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    centers = np.array(
        [
            [separation, separation],
            [-separation, -separation],
            [separation, -separation],
            [-separation, separation],
        ]
    )

    x = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, 1))

    samples_per_quadrant = n_samples // 4

    x[0:samples_per_quadrant] = centers[0] + np.random.randn(
        samples_per_quadrant, n_features
    )
    y[0:samples_per_quadrant] = 1

    x[samples_per_quadrant : 2 * samples_per_quadrant] = centers[
        1
    ] + np.random.randn(samples_per_quadrant, n_features)
    y[samples_per_quadrant : 2 * samples_per_quadrant] = 1

    x[2 * samples_per_quadrant : 3 * samples_per_quadrant] = centers[
        2
    ] + np.random.randn(samples_per_quadrant, n_features)
    y[2 * samples_per_quadrant : 3 * samples_per_quadrant] = -1

    x[3 * samples_per_quadrant :] = centers[3] + np.random.randn(
        n_samples - 3 * samples_per_quadrant, n_features
    )
    y[3 * samples_per_quadrant :] = -1

    indices = np.random.permutation(n_samples)
    x = x[indices]
    y = y[indices]

    return x, y, centers


def get_mnist():
    """
    Load MNIST dataset from local files or download if needed.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test: numpy arrays
    """
    dataset_dir = Path("datasets")
    dataset_dir.mkdir(exist_ok=True)

    files = [
        (
            "train-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        ),
        (
            "train-labels-idx1-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        ),
        (
            "t10k-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        ),
        (
            "t10k-labels-idx1-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        ),
    ]

    for filename, url in files:
        filepath = dataset_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                print("Alternative download options:")
                print(
                    "1. Kaggle: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data"
                )
                print(
                    "2. GitHub: https://github.com/golbin/tensorflow-mnist-tutorial/tree/master/mnist/data"
                )
                raise

    def read_images(filepath):
        with gzip.open(filepath, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                -1, rows * cols
            )
            return images / 255.0

    def read_labels(filepath):
        with gzip.open(filepath, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    try:
        x_train_full = read_images(dataset_dir / "train-images-idx3-ubyte.gz")
        y_train_full = read_labels(dataset_dir / "train-labels-idx1-ubyte.gz")
        x_test = read_images(dataset_dir / "t10k-images-idx3-ubyte.gz")
        y_test = read_labels(dataset_dir / "t10k-labels-idx1-ubyte.gz")

        val_size = 10000
        x_train = x_train_full[:-val_size]
        y_train = y_train_full[:-val_size]
        x_val = x_train_full[-val_size:]
        y_val = y_train_full[-val_size:]

        print(
            f"Dataset loaded: {x_train.shape[0]} training, {x_val.shape[0]} validation, {x_test.shape[0]} test samples"
        )
    except Exception as e:
        print(f"Error reading MNIST data: {e}")
        raise
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test
