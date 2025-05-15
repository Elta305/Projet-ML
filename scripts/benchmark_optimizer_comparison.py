import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import get_batches, get_mnist

from mlp.activation import ReLU
from mlp.linear import Linear
from mlp.loss import CrossEntropyLoss
from mlp.optim import SGD, Adam, SGDMomentum
from mlp.sequential import Sequential
from mlp.utils import one_hot_encoding


def train_with_early_stopping(
    model,
    optimizer,
    x_train,
    y_train,
    x_val,
    y_val,
    batch_size,
    patience=10,
    max_epochs=200,
):
    """Train a model with early stopping.

    Args:
        model: Neural network model
        optimizer: Optimizer instance
        x_train: Training features
        y_train: Training labels (one-hot encoded)
        x_val: Validation features
        y_val: Validation labels (one-hot encoded)
        batch_size: Mini-batch size
        patience: Number of epochs to wait for improvement
        max_epochs: Maximum number of epochs to train

    Returns:
        Dictionary with training history and metadata
    """
    loss_fn = optimizer.loss
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_times": [],
    }

    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        epoch_start_time = time.time()

        # Training phase
        train_losses = []
        train_accuracies = []

        model.train = True  # Set model to training mode
        batches = get_batches(x_train, y_train, batch_size)

        for batch_x, batch_y in batches:
            # Forward pass and optimization
            loss = optimizer.step(batch_x, batch_y)
            train_losses.append(loss)

            # Calculate accuracy
            output = model.forward(batch_x)
            predictions = np.argmax(output, axis=1)
            batch_y_labels = np.argmax(batch_y, axis=1)
            batch_accuracy = np.mean(predictions == batch_y_labels)
            train_accuracies.append(batch_accuracy)

        # Validation phase
        model.train = False  # Set model to evaluation mode
        val_output = model.forward(x_val)
        val_loss = loss_fn.forward(y_val, val_output)

        val_predictions = np.argmax(val_output, axis=1)
        val_true_labels = np.argmax(y_val, axis=1)
        val_accuracy = np.mean(val_predictions == val_true_labels)

        # Calculate epoch averages
        avg_train_loss = np.mean(train_losses)
        avg_train_accuracy = np.mean(train_accuracies)

        # Store metrics
        epoch_time = time.time() - epoch_start_time
        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(avg_train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["epoch_times"].append(epoch_time)

        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    history["total_epochs"] = len(history["train_loss"])
    history["best_val_accuracy"] = best_val_accuracy

    return history, model


def run_optimizer_benchmark():
    """Run benchmark comparing different optimizers."""
    # Load MNIST dataset
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    # Parameters
    num_classes = 10
    y_train_onehot = one_hot_encoding(y_train, num_classes)
    y_val_onehot = one_hot_encoding(y_val, num_classes)
    y_test_onehot = one_hot_encoding(y_test, num_classes)

    input_dim = x_train.shape[1]  # 784 for MNIST
    hidden_dim = 64
    output_dim = num_classes  # 10 for MNIST

    # Hyperparameters
    batch_size = 64
    n_runs = 15
    patience = 10
    max_epochs = 100

    # Random seeds for reproducibility
    np.random.seed(42)
    run_seeds = np.random.randint(0, 10000, size=n_runs)

    # Optimizer configurations
    optimizer_configs = [
        {"name": "SGD", "lr": 0.01, "class": SGD, "kwargs": {}},
        {
            "name": "SGD with Momentum",
            "lr": 0.01,
            "class": SGDMomentum,
            "kwargs": {"momentum": 0.9},
        },
        {
            "name": "Adam",
            "lr": 0.001,
            "class": Adam,
            "kwargs": {"beta1": 0.9, "beta2": 0.999},
        },
    ]

    results = {
        "parameters": {
            "n_runs": n_runs,
            "batch_size": batch_size,
            "patience": patience,
            "max_epochs": max_epochs,
            "hidden_dim": hidden_dim,
            "seeds": run_seeds.tolist(),
        },
        "optimizers": {},
    }

    # Run benchmark for each optimizer
    for config in optimizer_configs:
        optimizer_name = config["name"]
        print(f"\n{'-' * 40}\nBenchmarking {optimizer_name}\n{'-' * 40}")

        optimizer_results = {
            "histories": [],
            "test_accuracies": [],
            "convergence_epochs": [],
            "total_times": [],
        }

        for run_idx, seed in enumerate(
            tqdm(run_seeds, desc=f"Running {optimizer_name}")
        ):
            np.random.seed(seed)

            # Create model
            model = Sequential(
                Linear(input_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, output_dim),
                ReLU(),  # Final activation should actually be Softmax, but we'll apply that in the loss
            )

            # Create loss and optimizer
            loss_fn = CrossEntropyLoss()
            optimizer_class = config["class"]
            optimizer = optimizer_class(
                model, loss_fn, eps=config["lr"], **config["kwargs"]
            )

            # Train with early stopping
            start_time = time.time()
            history, trained_model = train_with_early_stopping(
                model,
                optimizer,
                x_train,
                y_train_onehot,
                x_val,
                y_val_onehot,
                batch_size,
                patience,
                max_epochs,
            )
            total_time = time.time() - start_time

            # Evaluate on test set
            test_output = trained_model.forward(x_test)
            test_predictions = np.argmax(test_output, axis=1)
            test_true_labels = np.argmax(y_test_onehot, axis=1)
            test_accuracy = np.mean(test_predictions == test_true_labels)

            # Store results
            optimizer_results["histories"].append(history)
            optimizer_results["test_accuracies"].append(float(test_accuracy))
            optimizer_results["convergence_epochs"].append(
                history["total_epochs"]
            )
            optimizer_results["total_times"].append(total_time)

            print(
                f"Run {run_idx + 1}/{n_runs} - Test Accuracy: {test_accuracy:.4f}, Epochs: {history['total_epochs']}"
            )

        # Calculate summary statistics
        optimizer_results["mean_test_accuracy"] = float(
            np.mean(optimizer_results["test_accuracies"])
        )
        optimizer_results["median_test_accuracy"] = float(
            np.median(optimizer_results["test_accuracies"])
        )
        optimizer_results["std_test_accuracy"] = float(
            np.std(optimizer_results["test_accuracies"])
        )

        optimizer_results["mean_epochs"] = float(
            np.mean(optimizer_results["convergence_epochs"])
        )
        optimizer_results["median_epochs"] = float(
            np.median(optimizer_results["convergence_epochs"])
        )

        optimizer_results["mean_time"] = float(
            np.mean(optimizer_results["total_times"])
        )
        optimizer_results["median_time"] = float(
            np.median(optimizer_results["total_times"])
        )

        results["optimizers"][optimizer_name] = optimizer_results

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save results
    with open("results/optimizer_benchmark_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Print summary table
    print("\n" + "=" * 80)
    print("OPTIMIZER BENCHMARK SUMMARY")
    print("=" * 80)
    summary_data = []

    for optimizer_name, optimizer_results in results["optimizers"].items():
        summary_data.append(
            {
                "Optimizer": optimizer_name,
                "Mean Test Accuracy": f"{optimizer_results['mean_test_accuracy']:.4f}",
                "Median Test Accuracy": f"{optimizer_results['median_test_accuracy']:.4f}",
                "Mean Epochs": f"{optimizer_results['mean_epochs']:.1f}",
                "Median Epochs": f"{optimizer_results['median_epochs']:.1f}",
                "Mean Time (s)": f"{optimizer_results['mean_time']:.1f}",
                "Median Time (s)": f"{optimizer_results['median_time']:.1f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print("Detailed results saved to: results/optimizer_benchmark_results.pkl")


if __name__ == "__main__":
    run_optimizer_benchmark()
