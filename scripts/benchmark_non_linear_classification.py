import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mlp.activation import Sigmoid
from mlp.linear import Linear
from mlp.loss import MSELoss
from mlp.sequential import Sequential

from .utils import generate_xor_classification_data, get_batches


def train_model(x, y, batch_size=32, learning_rate=0.01, n_epochs=500):
    """Train a perceptron with sigmoid activation for classification using
    mini-batches."""
    input_dim = x.shape[1]
    hidden_dim = 8
    output_dim = y.shape[1]

    model = Sequential(
        Linear(input_dim, hidden_dim),
        Sigmoid(),
        Linear(hidden_dim, output_dim),
        Sigmoid(),
    )

    loss_fn = MSELoss()

    losses = []
    accuracies = []

    for _ in range(n_epochs):
        epoch_losses = []
        epoch_accuracies = []

        batches = get_batches(x, y, batch_size)

        for batch_x, batch_y in batches:
            output = model.forward(batch_x)
            prediction = 2 * output - 1

            loss_values = loss_fn.forward(batch_y, prediction)
            batch_loss = np.mean(loss_values)
            epoch_losses.append(batch_loss)

            predictions = np.sign(prediction)
            batch_accuracy = np.mean(predictions == batch_y)
            epoch_accuracies.append(batch_accuracy)

            grad = loss_fn.backward(batch_y, prediction)
            grad_scaled = 2 * grad

            model.zero_grad()
            model.backward(grad_scaled)
            model.update_parameters(learning_rate)

        avg_epoch_loss = np.mean(epoch_losses)
        avg_epoch_accuracy = np.mean(epoch_accuracies)
        losses.append(avg_epoch_loss)
        accuracies.append(avg_epoch_accuracy)

    return model, losses, accuracies


def main():
    n_runs = 25
    n_samples = 400
    n_features = 2
    n_classes = 2
    separation = 3
    learning_rate = 0.01
    n_epochs = 256
    batch_size = 32
    random_seed = 42

    np.random.seed(random_seed)

    run_seeds = np.random.randint(0, 10000, size=n_runs)

    all_losses = []
    all_accuracies = []
    all_weights = []
    all_biases = []
    all_centers = []

    for run in tqdm(range(n_runs), desc="Running classification trials"):
        x, y, centers = generate_xor_classification_data(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            separation=separation,
            seed=run_seeds[run],
        )

        all_centers.append(centers.tolist())

        model, losses, accuracies = train_model(
            x,
            y,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
        )

        all_losses.append(losses)
        all_accuracies.append(accuracies)

        model_state = model.state_dict()

        layer_weights = []
        layer_biases = []

        for module, module_state in model_state.items():
            if isinstance(module, Linear):
                layer_weights.append(module_state["parameters"].tolist())
                if "bias" in module_state:
                    layer_biases.append(module_state["bias"].tolist())

        all_weights.append(layer_weights)
        all_biases.append(layer_biases)

    results = {
        "parameters": {
            "n_runs": n_runs,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "separation": separation,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
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

    with open("results/non_linear_classification_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
