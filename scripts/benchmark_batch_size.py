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
from mlp.optim import Adam
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
    """Train a model with early stopping and measure time per epoch."""
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
        val_output = model.forward(x_val)
        val_loss = loss_fn.forward(y_val, val_output)

        val_predictions = np.argmax(val_output, axis=1)
        val_true_labels = np.argmax(y_val, axis=1)
        val_accuracy = np.mean(val_predictions == val_true_labels)

        # Calculate epoch averages
        avg_train_loss = np.mean(train_losses)
        avg_train_accuracy = np.mean(train_accuracies)

        # Measure epoch time
        epoch_time = time.time() - epoch_start_time

        # Store metrics
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
    history["total_training_time"] = sum(history["epoch_times"])

    return history, model


def run_batch_size_benchmark():
    """Run benchmark comparing different batch sizes."""
    # Load MNIST dataset
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    # Parameters
    num_classes = 10
    y_train_onehot = one_hot_encoding(y_train, num_classes)
    y_val_onehot = one_hot_encoding(y_val, num_classes)
    y_test_onehot = one_hot_encoding(y_test, num_classes)

    input_dim = x_train.shape[1]  # 784 for MNIST
    hidden_dims = [64, 32]  # Standard 2-layer MLP
    output_dim = num_classes  # 10 for MNIST

    # Hyperparameters
    learning_rate = 0.001  # Default for Adam
    n_runs = 15
    patience = 10
    max_epochs = 100

    # Random seeds for reproducibility
    np.random.seed(42)
    run_seeds = np.random.randint(0, 10000, size=n_runs)

    # Batch sizes to test
    batch_sizes = [16, 32, 64, 128, 256]

    results = {
        "parameters": {
            "n_runs": n_runs,
            "learning_rate": learning_rate,
            "patience": patience,
            "max_epochs": max_epochs,
            "hidden_dims": hidden_dims,
            "seeds": run_seeds.tolist(),
        },
        "batch_sizes": {},
    }

    # Run benchmark for each batch size
    for batch_size in batch_sizes:
        print(
            f"\n{'-' * 40}\nBenchmarking batch size: {batch_size}\n{'-' * 40}"
        )

        batch_results = {
            "histories": [],
            "test_accuracies": [],
            "convergence_epochs": [],
            "training_times": [],
        }

        for run_idx, seed in enumerate(
            tqdm(run_seeds, desc=f"Running batch_size={batch_size}")
        ):
            np.random.seed(seed)

            # Create model (standard MLP with 2 hidden layers and ReLU)
            model = Sequential(
                Linear(input_dim, hidden_dims[0]),
                ReLU(),
                Linear(hidden_dims[0], hidden_dims[1]),
                ReLU(),
                Linear(hidden_dims[1], output_dim),
            )

            # Create loss and optimizer (Adam with default parameters)
            loss_fn = CrossEntropyLoss()
            optimizer = Adam(model, loss_fn, eps=learning_rate)

            # Train with early stopping and measure time
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

            # Evaluate on test set
            test_output = trained_model.forward(x_test)
            test_predictions = np.argmax(test_output, axis=1)
            test_true_labels = np.argmax(y_test_onehot, axis=1)
            test_accuracy = np.mean(test_predictions == test_true_labels)

            # Store results
            batch_results["histories"].append(history)
            batch_results["test_accuracies"].append(float(test_accuracy))
            batch_results["convergence_epochs"].append(history["total_epochs"])
            batch_results["training_times"].append(
                history["total_training_time"]
            )

            print(
                f"Run {run_idx + 1}/{n_runs} - Test Accuracy: {test_accuracy:.4f}, "
                f"Epochs: {history['total_epochs']}, "
                f"Time: {history['total_training_time']:.2f}s"
            )

        # Calculate summary statistics
        batch_results["mean_test_accuracy"] = float(
            np.mean(batch_results["test_accuracies"])
        )
        batch_results["median_test_accuracy"] = float(
            np.median(batch_results["test_accuracies"])
        )
        batch_results["std_test_accuracy"] = float(
            np.std(batch_results["test_accuracies"])
        )

        batch_results["mean_epochs"] = float(
            np.mean(batch_results["convergence_epochs"])
        )
        batch_results["median_epochs"] = float(
            np.median(batch_results["convergence_epochs"])
        )

        batch_results["mean_training_time"] = float(
            np.mean(batch_results["training_times"])
        )
        batch_results["median_training_time"] = float(
            np.median(batch_results["training_times"])
        )

        results["batch_sizes"][str(batch_size)] = batch_results

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save results
    with open("results/batch_size_benchmark_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Print summary table
    print("\n" + "=" * 80)
    print("BATCH SIZE BENCHMARK SUMMARY")
    print("=" * 80)
    summary_data = []

    for batch_size, batch_results in results["batch_sizes"].items():
        summary_data.append(
            {
                "Batch Size": batch_size,
                "Mean Test Accuracy": f"{batch_results['mean_test_accuracy']:.4f}",
                "Mean Epochs": f"{batch_results['mean_epochs']:.1f}",
                "Mean Training Time (s)": f"{batch_results['mean_training_time']:.1f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(
        "Detailed results saved to: results/batch_size_benchmark_results.pkl"
    )


if __name__ == "__main__":
    run_batch_size_benchmark()
