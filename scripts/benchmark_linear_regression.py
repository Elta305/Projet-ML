import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mlp.linear import Linear
from mlp.loss import MSELoss


def generate_linear_data(n_samples=1000, noise=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.random.rand(n_samples, 1) * 10
    true_weight = np.random.uniform(-5, 5)
    true_bias = np.random.uniform(-10, 10)
    y = true_weight * x + true_bias + noise * np.random.randn(n_samples, 1)

    return x, y, true_weight, true_bias


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


def train_linear_model(x, y, batch_size=32, learning_rate=0.01, n_epochs=500):
    """Train a linear regression model."""
    input_dim = x.shape[1]
    output_dim = y.shape[1]

    model = Linear(input_dim, output_dim)

    loss_fn = MSELoss()

    losses = []

    for _ in range(n_epochs):
        epoch_losses = []

        batches = get_batches(x, y, batch_size)

        for batch_x, batch_y in batches:
            y_pred = model.forward(batch_x)

            loss_values = loss_fn.forward(batch_y, y_pred)
            avg_loss = np.mean(loss_values)
            epoch_losses.append(avg_loss)

            grad = loss_fn.backward(batch_y, y_pred)

            model.zero_grad()
            model.backward_update_gradient(batch_x, grad)
            model.update_parameters(learning_rate)

        avg_epoch_loss = np.mean(epoch_losses)
        losses.append(avg_epoch_loss)

    return model, losses


def main():
    n_runs = 25
    n_samples = 400
    noise = 0.5
    learning_rate = 0.01
    n_epochs = 500
    random_seed = 42

    np.random.seed(random_seed)

    run_seeds = np.random.randint(0, 10000, size=n_runs)

    all_losses = []
    all_weights = []
    all_biases = []
    true_weights = []
    true_biases = []

    for run in tqdm(range(n_runs), desc="Running linear regression trials"):
        x, y, true_weight, true_bias = generate_linear_data(
            n_samples=n_samples, noise=noise, seed=run_seeds[run]
        )

        true_weights.append(true_weight)
        true_biases.append(true_bias)

        model, losses = train_linear_model(
            x, y, learning_rate=learning_rate, n_epochs=n_epochs
        )

        all_losses.append(losses)
        recovered_weight = model.parameters[0, 0]
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
