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


def calculate_parameters(input_dim, hidden_dim, output_dim):
    """Calculate the number of parameters in a 2-layer MLP."""
    # First layer: input_dim * hidden_dim weights + hidden_dim biases
    params_layer1 = input_dim * hidden_dim + hidden_dim

    # Second layer: hidden_dim * hidden_dim weights + hidden_dim biases
    params_layer2 = hidden_dim * hidden_dim + hidden_dim

    # Output layer: hidden_dim * output_dim weights + output_dim biases
    params_output = hidden_dim * output_dim + output_dim

    return params_layer1 + params_layer2 + params_output


def train_with_early_stopping(
    model,
    optimizer,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    batch_size,
    patience=10,
    max_epochs=200,
):
    """Train a model with early stopping and measure train-test gap."""
    loss_fn = optimizer.loss
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_accuracy": [],
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

        # Test phase (to track generalization gap)
        test_output = model.forward(x_test)
        test_predictions = np.argmax(test_output, axis=1)
        test_true_labels = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_predictions == test_true_labels)

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
        history["test_accuracy"].append(test_accuracy)
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

    # Calculate final generalization gap
    final_train_accuracy = history["train_accuracy"][-1]
    final_test_accuracy = history["test_accuracy"][-1]
    history["generalization_gap"] = final_train_accuracy - final_test_accuracy

    return history, model


def run_hidden_size_benchmark():
    """Run benchmark comparing different hidden layer sizes."""
    # Load MNIST dataset
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    # Parameters
    num_classes = 10
    y_train_onehot = one_hot_encoding(y_train, num_classes)
    y_val_onehot = one_hot_encoding(y_val, num_classes)
    y_test_onehot = one_hot_encoding(y_test, num_classes)

    input_dim = x_train.shape[1]  # 784 for MNIST
    output_dim = num_classes  # 10 for MNIST

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001  # Default for Adam
    n_runs = 15
    patience = 10
    max_epochs = 100

    # Random seeds for reproducibility
    np.random.seed(42)
    run_seeds = np.random.randint(0, 10000, size=n_runs)

    # Hidden layer sizes to test (same size for both layers)
    hidden_sizes = [8, 16, 32, 64, 128]

    results = {
        "parameters": {
            "n_runs": n_runs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "max_epochs": max_epochs,
            "seeds": run_seeds.tolist(),
        },
        "hidden_sizes": {},
    }

    # Run benchmark for each hidden size
    for hidden_size in hidden_sizes:
        hidden_dims = [hidden_size, hidden_size]  # Same size for both layers
        print(
            f"\n{'-' * 40}\nBenchmarking hidden size: ({hidden_size},{hidden_size})\n{'-' * 40}"
        )

        # Calculate number of parameters for this architecture
        num_params = calculate_parameters(input_dim, hidden_size, output_dim)

        size_results = {
            "hidden_dims": hidden_dims,
            "num_parameters": num_params,
            "histories": [],
            "test_accuracies": [],
            "generalization_gaps": [],
            "convergence_epochs": [],
            "training_times": [],
        }

        for run_idx, seed in enumerate(
            tqdm(
                run_seeds,
                desc=f"Running hidden_size=({hidden_size},{hidden_size})",
            )
        ):
            np.random.seed(seed)

            # Create model with specified hidden layer sizes
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

            # Train with early stopping and measure train-test gap
            history, trained_model = train_with_early_stopping(
                model,
                optimizer,
                x_train,
                y_train_onehot,
                x_val,
                y_val_onehot,
                x_test,
                y_test_onehot,
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
            size_results["histories"].append(history)
            size_results["test_accuracies"].append(float(test_accuracy))
            size_results["generalization_gaps"].append(
                float(history["generalization_gap"])
            )
            size_results["convergence_epochs"].append(history["total_epochs"])
            size_results["training_times"].append(
                history["total_training_time"]
            )

            print(
                f"Run {run_idx + 1}/{n_runs} - Test Accuracy: {test_accuracy:.4f}, "
                f"Gap: {history['generalization_gap']:.4f}, "
                f"Epochs: {history['total_epochs']}"
            )

        # Calculate summary statistics
        size_results["mean_test_accuracy"] = float(
            np.mean(size_results["test_accuracies"])
        )
        size_results["std_test_accuracy"] = float(
            np.std(size_results["test_accuracies"])
        )

        size_results["mean_generalization_gap"] = float(
            np.mean(size_results["generalization_gaps"])
        )
        size_results["std_generalization_gap"] = float(
            np.std(size_results["generalization_gaps"])
        )

        size_results["mean_epochs"] = float(
            np.mean(size_results["convergence_epochs"])
        )
        size_results["mean_training_time"] = float(
            np.mean(size_results["training_times"])
        )

        results["hidden_sizes"][str(hidden_size)] = size_results

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save results
    with open("results/hidden_size_benchmark_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Print summary table
    print("\n" + "=" * 100)
    print("HIDDEN LAYER SIZE BENCHMARK SUMMARY")
    print("=" * 100)
    summary_data = []

    for hidden_size, size_results in results["hidden_sizes"].items():
        summary_data.append(
            {
                "Hidden Size": f"({hidden_size},{hidden_size})",
                "Parameters": f"{size_results['num_parameters']:,}",
                "Mean Test Accuracy": f"{size_results['mean_test_accuracy']:.4f}",
                "Gen. Gap": f"{size_results['mean_generalization_gap']:.4f}",
                "Mean Epochs": f"{size_results['mean_epochs']:.1f}",
                "Mean Time (s)": f"{size_results['mean_training_time']:.1f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 100)
    print(
        "Detailed results saved to: results/hidden_size_benchmark_results.pkl"
    )


if __name__ == "__main__":
    run_hidden_size_benchmark()
