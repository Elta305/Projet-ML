import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mlp.activation import Sigmoid
from mlp.linear import Linear
from mlp.loss import MSELoss


def generate_classification_data(
    n_samples=1000, n_features=2, n_classes=2, separation=1.0, seed=None
):
    """Generate synthetic data for classification with optional random seed."""
    if seed is not None:
        np.random.seed(seed)

    centers = np.random.uniform(-5, 5, size=(n_classes, n_features))

    for i in range(n_classes):
        centers[i] = centers[i] * separation

    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, 1))

    samples_per_class = n_samples // n_classes
    for i in range(n_classes):
        start_idx = i * samples_per_class
        end_idx = (
            start_idx + samples_per_class if i < n_classes - 1 else n_samples
        )

        class_samples = end_idx - start_idx
        X[start_idx:end_idx] = centers[i] + np.random.randn(
            class_samples, n_features
        )
        y[start_idx:end_idx] = i

    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    if n_classes == 2:
        y = 2 * y - 1

    return X, y, centers


def train_sigmoid_model(X, y, learning_rate=0.01, n_epochs=500):
    """Train a perceptron with sigmoid activation for classification."""
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    linear_layer = Linear(input_dim, output_dim)
    sigmoid = Sigmoid()
    loss_fn = MSELoss()
    losses = []
    accuracies = []

    for epoch in range(n_epochs):
        linear_output = linear_layer.forward(X)
        y_pred = sigmoid.forward(linear_output)

        y_pred_scaled = 2 * y_pred - 1

        loss_values = loss_fn.forward(y, y_pred_scaled)
        avg_loss = np.mean(loss_values)
        losses.append(avg_loss)

        predictions = np.sign(y_pred_scaled)
        accuracy = np.mean(predictions == y)
        accuracies.append(accuracy)

        grad = loss_fn.backward(y, y_pred_scaled)
        grad_sigmoid = grad * 2
        delta = sigmoid.backward_delta(linear_output, grad_sigmoid)
        linear_layer.zero_grad()
        linear_layer.backward_update_gradient(X, delta)
        linear_layer.update_parameters(learning_rate)

    return (linear_layer, sigmoid), losses, accuracies


def main():
    n_runs = 100
    n_samples = 400
    n_features = 2
    n_classes = 2
    separation = 1.2
    learning_rate = 0.01
    n_epochs = 1000
    random_seed = 42

    np.random.seed(random_seed)

    run_seeds = np.random.randint(0, 10000, size=n_runs)

    all_losses = []
    all_accuracies = []
    all_weights = []
    all_biases = []
    all_centers = []

    for run in tqdm(range(n_runs), desc="Running classification trials"):
        X, y, centers = generate_classification_data(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            separation=separation,
            seed=run_seeds[run],
        )

        all_centers.append(centers.tolist())

        model, losses, accuracies = train_sigmoid_model(
            X, y, learning_rate=learning_rate, n_epochs=n_epochs
        )

        all_losses.append(losses)
        all_accuracies.append(accuracies)

        linear_layer, _ = model
        weights = linear_layer._parameters.copy()
        bias = linear_layer.bias.copy()

        all_weights.append(weights.tolist())
        all_biases.append(bias.tolist())

    results = {
        "parameters": {
            "n_runs": n_runs,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "separation": separation,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "random_seed": random_seed,
            "run_seeds": run_seeds.tolist(),
        },
        "centers": all_centers,
        "weights": all_weights,
        "biases": all_biases,
        "all_losses": all_losses,
        "all_accuracies": all_accuracies,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/linear_classification_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
