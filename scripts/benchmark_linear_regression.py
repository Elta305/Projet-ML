import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mlp.linear import Linear
from mlp.loss import MSELoss


def generate_linear_data(n_samples=1000, noise=0.5, seed=None):
    """Generate synthetic data for linear regression with optional random seed."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.rand(n_samples, 1) * 10
    true_weight = np.random.uniform(-5, 5)
    true_bias = np.random.uniform(-10, 10)
    y = true_weight * X + true_bias + noise * np.random.randn(n_samples, 1)

    return X, y, true_weight, true_bias


def train_linear_model(X, y, learning_rate=0.01, n_epochs=500):
    """Train a linear regression model."""
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = Linear(input_dim, output_dim)
    loss_fn = MSELoss()
    losses = []

    for epoch in range(n_epochs):
        y_pred = model.forward(X)
        loss_values = loss_fn.forward(y, y_pred)
        avg_loss = np.mean(loss_values)
        losses.append(avg_loss)

        grad = loss_fn.backward(y, y_pred)
        model.zero_grad()
        model.backward_update_gradient(X, grad)
        model.update_parameters(learning_rate)

    return model, losses


def main():
    n_runs = 100
    n_samples = 200
    noise = 0.5
    learning_rate = 0.01
    n_epochs = 1000
    random_seed = 42

    np.random.seed(random_seed)

    run_seeds = np.random.randint(0, 10000, size=n_runs)

    all_losses = []
    all_weights = []
    all_biases = []
    true_weights = []
    true_biases = []

    for run in tqdm(range(n_runs), desc="Running linear regression trials"):
        X, y, true_weight, true_bias = generate_linear_data(
            n_samples=n_samples, noise=noise, seed=run_seeds[run]
        )

        true_weights.append(true_weight)
        true_biases.append(true_bias)

        model, losses = train_linear_model(
            X, y, learning_rate=learning_rate, n_epochs=n_epochs
        )

        all_losses.append(losses)
        recovered_weight = model._parameters[0, 0]
        recovered_bias = model.bias[0, 0] if hasattr(model, "bias") else 0
        all_weights.append(recovered_weight)
        all_biases.append(recovered_bias)

    results = {
        "parameters": {
            "n_runs": n_runs,
            "n_samples": n_samples,
            "noise": noise,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "random_seed": random_seed,
            "run_seeds": run_seeds.tolist(),
        },
        "true_weights": true_weights,
        "true_biases": true_biases,
        "recovered_weights": all_weights,
        "recovered_biases": all_biases,
        "all_losses": np.array(all_losses).tolist(),
    }

    Path("results").mkdir(exist_ok=True)

    with open("results/linear_regression_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
