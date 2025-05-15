import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils import compute_stats


def visualize_activation_benchmark():
    """Create simple visualizations for activation benchmark results."""
    # Load results
    with open("results/activation_benchmark_results.pkl", "rb") as f:
        results = pickle.load(f)

    activation_names = list(results["activations"].keys())

    # Create figure directory if it doesn't exist
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Prepare data for learning curves
    max_epochs = max(
        [
            max(
                [
                    h["total_epochs"]
                    for h in results["activations"][act]["histories"]
                ]
            )
            for act in activation_names
        ]
    )

    # Colors for different activations
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    # Figure 1: Validation Accuracy Curves (simple version)
    plt.figure(figsize=(10, 6))

    for i, activation_name in enumerate(activation_names):
        activation_results = results["activations"][activation_name]
        histories = activation_results["histories"]

        # Prepare validation accuracy data
        val_accs = np.zeros((len(histories), max_epochs))

        for j, history in enumerate(histories):
            epochs_run = len(history["val_accuracy"])
            val_accs[j, :epochs_run] = history["val_accuracy"]

            # Pad with the last value
            if epochs_run < max_epochs:
                val_accs[j, epochs_run:] = history["val_accuracy"][-1]

        # Calculate statistics for each epoch
        epochs = np.arange(1, max_epochs + 1)
        val_acc_stats = [
            compute_stats(val_accs[:, e]) for e in range(max_epochs)
        ]

        # Extract IQM and IQR
        val_acc_iqm = [s["iqm"] for s in val_acc_stats]
        val_acc_q1 = [s["q1"] for s in val_acc_stats]
        val_acc_q3 = [s["q3"] for s in val_acc_stats]

        # Plot validation accuracy curves
        color = colors[i]
        alpha = 0.2  # Alpha for IQR shading

        plt.plot(
            epochs, val_acc_iqm, color=color, linewidth=2, label=activation_name
        )
        plt.fill_between(
            epochs, val_acc_q1, val_acc_q3, color=color, alpha=alpha
        )

    plt.title("Validation Accuracy by Activation Function")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/activation_validation_accuracy.png", dpi=300)

    # Figure 2: Test Accuracy Boxplot (simple version)
    plt.figure(figsize=(8, 6))

    test_accuracies = [
        results["activations"][act]["test_accuracies"]
        for act in activation_names
    ]

    plt.boxplot(test_accuracies, labels=activation_names)
    plt.title("Test Accuracy by Activation Function")
    plt.ylabel("Accuracy")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("figures/activation_test_accuracy.png", dpi=300)

    # Figure 3: Simple bar chart of mean test accuracies
    plt.figure(figsize=(8, 6))

    mean_accuracies = [
        results["activations"][act]["mean_test_accuracy"]
        for act in activation_names
    ]
    std_accuracies = [
        results["activations"][act]["std_test_accuracy"]
        for act in activation_names
    ]

    bars = plt.bar(
        activation_names, mean_accuracies, yerr=std_accuracies, capsize=10
    )

    # Add color to bars
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])

    plt.title("Mean Test Accuracy by Activation Function")
    plt.ylabel("Accuracy")
    plt.ylim(
        0.8 * min(mean_accuracies), 1.02
    )  # Adjust y-axis to highlight differences
    plt.grid(axis="y")

    # Add text labels on bars
    for i, v in enumerate(mean_accuracies):
        plt.text(i, v + 0.005, f"{v:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig("figures/activation_mean_accuracy.png", dpi=300)

    # Print summary table
    print("\nACTIVATION FUNCTION BENCHMARK SUMMARY")
    print("=" * 50)
    print(
        f"{'Activation':<10} {'Mean Acc.':<10} {'Median Acc.':<12} {'Std Dev':<10} {'Epochs':<10}"
    )
    print("-" * 50)

    for act_name in activation_names:
        act_results = results["activations"][act_name]
        print(
            f"{act_name:<10} "
            f"{act_results['mean_test_accuracy']:.4f}    "
            f"{act_results['median_test_accuracy']:.4f}      "
            f"{act_results['std_test_accuracy']:.4f}    "
            f"{act_results['mean_epochs']:.1f}"
        )

    print("=" * 50)
    print("Visualizations created in 'figures' directory:")
    print("1. activation_validation_accuracy.png - Validation accuracy curves")
    print("2. activation_test_accuracy.png - Test accuracy boxplot")
    print("3. activation_mean_accuracy.png - Mean test accuracy bar chart")


if __name__ == "__main__":
    visualize_activation_benchmark()
