import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils import get_batches, get_mnist

from mlp.activation import ReLU, Softmax
from mlp.linear import Linear
from mlp.loss import CrossEntropyLoss
from mlp.optim import Adam
from mlp.sequential import Sequential
from mlp.utils import one_hot_encoding


def train_mnist_classifier(
    x_train,
    y_train,
    x_val,
    y_val,
    batch_size,
    learning_rate,
    n_epochs,
    hidden_dims,
    seed,
):
    """Train a MNIST classifier."""
    np.random.seed(seed)

    num_classes = 10
    y_train_onehot = one_hot_encoding(y_train, num_classes)
    y_val_onehot = one_hot_encoding(y_val, num_classes)

    input_dim = x_train.shape[1]
    output_dim = num_classes

    # Build the model
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(Linear(prev_dim, hidden_dim))
        layers.append(ReLU())
        prev_dim = hidden_dim
    layers.append(Linear(prev_dim, output_dim))
    layers.append(Softmax())

    model = Sequential(*layers)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model, loss_fn, eps=learning_rate)

    # Track training metrics
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Track best model state based on validation accuracy
    best_val_accuracy = 0
    best_model_state = None

    with tqdm(total=n_epochs, desc=f"Training run (seed={seed})") as pbar:
        for epoch in range(n_epochs):
            train_losses = []
            train_accuracies = []

            # Train on batches
            batches = get_batches(x_train, y_train_onehot, batch_size)
            for batch_x, batch_y in batches:
                loss = optimizer.step(batch_x, batch_y)
                train_losses.append(loss)

                output = model.forward(batch_x)
                predictions = np.argmax(output, axis=1)
                batch_y_labels = np.argmax(batch_y, axis=1)
                batch_accuracy = np.mean(predictions == batch_y_labels)
                train_accuracies.append(batch_accuracy)

            # Evaluate on validation set
            val_output = model.forward(x_val)
            val_loss = loss_fn.forward(y_val_onehot, val_output)
            val_predictions = np.argmax(val_output, axis=1)
            val_true_labels = np.argmax(y_val_onehot, axis=1)
            val_accuracy = np.mean(val_predictions == val_true_labels)

            # Calculate averages
            avg_train_loss = np.mean(train_losses)
            avg_train_accuracy = np.mean(train_accuracies)

            # Store metrics
            history["train_loss"].append(avg_train_loss)
            history["train_accuracy"].append(avg_train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(
                {
                    "train_loss": f"{avg_train_loss:.4f}",
                    "val_acc": f"{val_accuracy:.4f}",
                }
            )

            # Keep track of best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, best_val_accuracy


def main():
    """Train MNIST classifier with multiple runs and save results."""
    # Hyperparameters
    n_runs = 15
    batch_size = 128
    learning_rate = 0.001
    n_epochs = 20
    hidden_dims = [64, 32]
    random_seed = 42

    # Generate random seeds for each run
    np.random.seed(random_seed)
    run_seeds = np.random.randint(0, 10000, size=n_runs)

    # Get MNIST data
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    # Prepare for storing results
    all_histories = []
    all_val_accuracies = []
    best_model = None
    best_accuracy = 0

    # Run training multiple times
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        model, history, val_accuracy = train_mnist_classifier(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            hidden_dims=hidden_dims,
            seed=run_seeds[run],
        )

        all_histories.append(history)
        all_val_accuracies.append(val_accuracy)

        # Keep track of best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model

    # Evaluate best model on test set
    num_classes = 10
    y_test_onehot = one_hot_encoding(y_test, num_classes)
    test_output = best_model.forward(x_test)
    test_predictions = np.argmax(test_output, axis=1)
    test_true_labels = np.argmax(y_test_onehot, axis=1)
    test_accuracy = np.mean(test_predictions == test_true_labels)
    print(f"\nBest model test accuracy: {test_accuracy:.4f}")

    # Save results
    results = {
        "parameters": {
            "n_runs": n_runs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "hidden_dims": hidden_dims,
            "random_seed": random_seed,
            "run_seeds": run_seeds.tolist(),
        },
        "best_model_state": best_model.state_dict(),
        "all_histories": all_histories,
        "all_val_accuracies": all_val_accuracies,
        "best_val_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/mnist_classification_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Results saved to results/mnist_classification_results.pkl")


if __name__ == "__main__":
    main()
