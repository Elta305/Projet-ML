import pickle
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
    """Train a model with early stopping."""
    loss_fn = optimizer.loss
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(max_epochs):
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

        # Store metrics
        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(avg_train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

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


def run_initialization_benchmark():
    """Run benchmark comparing different parameter initializations."""
    # Load MNIST dataset
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    # Parameters
    num_classes = 10
    y_train_onehot = one_hot_encoding(y_train, num_classes)
    y_val_onehot = one_hot_encoding(y_val, num_classes)
    y_test_onehot = one_hot_encoding(y_test, num_classes)

    input_dim = x_train.shape[1]  # 784 for MNIST
    hidden_dims = [64, 32]
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

    # Initialization configurations
    initialization_configs = [
        {"name": "Random", "init": "random"},
        {"name": "Normal", "init": "normal"},
        {"name": "He Normal", "init": "he_normal"},
        {"name": "Xavier Normal", "init": "xavier_normal"},
        {"name": "Uniform", "init": "uniform"},
        {"name": "He Uniform", "init": "he_uniform"},
        {"name": "Xavier Uniform", "init": "xavier_uniform"},
    ]

    results = {
        "parameters": {
            "n_runs": n_runs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "max_epochs": max_epochs,
            "hidden_dims": hidden_dims,
            "seeds": run_seeds.tolist(),
        },
        "initializations": {},
    }

    # Run benchmark for each initialization method
    for config in initialization_configs:
        init_name = config["name"]
        init_method = config["init"]
        print(
            f"\n{'-' * 40}\nBenchmarking {init_name} initialization\n{'-' * 40}"
        )

        init_results = {
            "histories": [],
            "test_accuracies": [],
            "convergence_epochs": [],
        }

        for run_idx, seed in enumerate(
            tqdm(run_seeds, desc=f"Running {init_name}")
        ):
            np.random.seed(seed)

            # Create model with specified initialization method
            model = Sequential(
                Linear(input_dim, hidden_dims[0], init=init_method),
                ReLU(),
                Linear(hidden_dims[0], hidden_dims[1], init=init_method),
                ReLU(),
                Linear(hidden_dims[1], output_dim, init=init_method),
            )

            # Create loss and optimizer (Adam with default parameters)
            loss_fn = CrossEntropyLoss()
            optimizer = Adam(model, loss_fn, eps=learning_rate)

            # Train with early stopping
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
            init_results["histories"].append(history)
            init_results["test_accuracies"].append(float(test_accuracy))
            init_results["convergence_epochs"].append(
                history["total_epochs"]
            )

            print(
                f"Run {run_idx + 1}/{n_runs} - Test Accuracy: {test_accuracy:.4f}, Epochs: {history['total_epochs']}"
            )

        # Calculate summary statistics
        init_results["mean_test_accuracy"] = float(
            np.mean(init_results["test_accuracies"])
        )
        init_results["median_test_accuracy"] = float(
            np.median(init_results["test_accuracies"])
        )
        init_results["std_test_accuracy"] = float(
            np.std(init_results["test_accuracies"])
        )

        init_results["mean_epochs"] = float(
            np.mean(init_results["convergence_epochs"])
        )
        init_results["median_epochs"] = float(
            np.median(init_results["convergence_epochs"])
        )

        results["initializations"][init_name] = init_results

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save results
    with open("results/initialization_benchmark_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Print summary table
    print("\n" + "=" * 80)
    print("PARAMETER INITIALIZATION BENCHMARK SUMMARY")
    print("=" * 80)
    summary_data = []

    for init_name, init_results in results["initializations"].items():
        summary_data.append(
            {
                "Initialization": init_name,
                "Mean Test Accuracy": f"{init_results['mean_test_accuracy']:.4f}",
                "Median Test Accuracy": f"{init_results['median_test_accuracy']:.4f}",
                "Std Dev": f"{init_results['std_test_accuracy']:.4f}",
                "Mean Epochs": f"{init_results['mean_epochs']:.1f}",
                "Median Epochs": f"{init_results['median_epochs']:.1f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(
        "Detailed results saved to: results/initialization_benchmark_results.pkl"
    )


if __name__ == "__main__":
    run_initialization_benchmark()