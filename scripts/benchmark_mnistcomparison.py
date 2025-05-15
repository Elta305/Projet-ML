import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import get_batches, get_mnist

from mlp.activation import TanH, ReLU, Softmax
from mlp.linear import Linear
from mlp.loss import CrossEntropyLoss
from mlp.optim import Adam
from mlp.sequential import Sequential
from mlp.utils import one_hot_encoding
import pandas as pd


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
    activation_cls,
):
    np.random.seed(seed)

    num_classes = 10
    y_train_onehot = one_hot_encoding(y_train, num_classes)
    y_val_onehot = one_hot_encoding(y_val, num_classes)

    input_dim = x_train.shape[1]
    output_dim = num_classes

    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(Linear(prev_dim, hidden_dim))
        layers.append(activation_cls())
        prev_dim = hidden_dim
    layers.append(Linear(prev_dim, output_dim))
    layers.append(Softmax())

    model = Sequential(*layers)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model, loss_fn, eps=learning_rate)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0
    best_model_state = None

    for _ in range(n_epochs):
        train_losses = []
        train_accuracies = []

        batches = get_batches(x_train, y_train_onehot, batch_size)
        for batch_x, batch_y in batches:
            loss = optimizer.step(batch_x, batch_y)
            train_losses.append(loss)

            output = model.forward(batch_x)
            predictions = np.argmax(output, axis=1)
            batch_y_labels = np.argmax(batch_y, axis=1)
            batch_accuracy = np.mean(predictions == batch_y_labels)
            train_accuracies.append(batch_accuracy)

        val_output = model.forward(x_val)
        val_loss = loss_fn.forward(y_val_onehot, val_output)
        val_predictions = np.argmax(val_output, axis=1)
        val_true_labels = np.argmax(y_val_onehot, axis=1)
        val_accuracy = np.mean(val_predictions == val_true_labels)

        avg_train_loss = np.mean(train_losses)
        avg_train_accuracy = np.mean(train_accuracies)

        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(avg_train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, best_val_accuracy


def main():
    n_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    hidden_dims = [64, 32]
    random_seed = 42

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    # Train with TanH
    _, tanh_history, _ = train_mnist_classifier(
        x_train, y_train, x_val, y_val,
        batch_size, learning_rate, n_epochs, hidden_dims, random_seed, TanH
    )

    # Train with ReLU
    _, relu_history, _ = train_mnist_classifier(
        x_train, y_train, x_val, y_val,
        batch_size, learning_rate, n_epochs, hidden_dims, random_seed, ReLU
    )

    # Plot Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tanh_history["train_loss"], label="TanH Train Loss")
    plt.plot(tanh_history["val_loss"], label="TanH Val Loss")
    plt.plot(relu_history["train_loss"], label="ReLU Train Loss")
    plt.plot(relu_history["val_loss"], label="ReLU Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(tanh_history["train_accuracy"], label="TanH Train Acc")
    plt.plot(tanh_history["val_accuracy"], label="TanH Val Acc")
    plt.plot(relu_history["train_accuracy"], label="ReLU Train Acc")
    plt.plot(relu_history["val_accuracy"], label="ReLU Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Example: Collect results for demonstration (replace with actual results)
    results = {
        "initializations": {
            "TanH": {
                "mean_test_accuracy": np.mean(tanh_history["val_accuracy"]),
                "median_test_accuracy": np.median(tanh_history["val_accuracy"]),
                "std_test_accuracy": np.std(tanh_history["val_accuracy"]),
                "mean_epochs": len(tanh_history["val_accuracy"]),
                "median_epochs": len(tanh_history["val_accuracy"]),
            },
            "ReLU": {
                "mean_test_accuracy": np.mean(relu_history["val_accuracy"]),
                "median_test_accuracy": np.median(relu_history["val_accuracy"]),
                "std_test_accuracy": np.std(relu_history["val_accuracy"]),
                "mean_epochs": len(relu_history["val_accuracy"]),
                "median_epochs": len(relu_history["val_accuracy"]),
            },
        }
    }

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save results
    with open("results/benchmark_mnistcomparison_results.pkl", "wb") as f:
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
        "Detailed results saved to: results/benchmark_mnistcomparison_results.pkl"
    )

if __name__ == "__main__":
    main()
