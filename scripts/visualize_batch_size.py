import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils import compute_stats


def visualize_batch_size_benchmark():
    """Create visualizations for batch size benchmark results."""
    # Load results
    with open("results/batch_size_benchmark_results.pkl", "rb") as f:
        results = pickle.load(f)

    batch_sizes = list(results["batch_sizes"].keys())
    batch_sizes = [
        int(bs) for bs in batch_sizes
    ]  # Convert to integers for proper sorting
    batch_sizes.sort()  # Ensure they're sorted
    batch_sizes = [str(bs) for bs in batch_sizes]  # Convert back to strings

    # Create figure directory if it doesn't exist
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Prepare data for learning curves
    max_epochs = max(
        [
            max(
                [
                    h["total_epochs"]
                    for h in results["batch_sizes"][bs]["histories"]
                ]
            )
            for bs in batch_sizes
        ]
    )

    # Colors for different batch sizes
    colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

    # Figure 1: Validation Accuracy Curves
    plt.figure(figsize=(10, 6))

    for i, batch_size in enumerate(batch_sizes):
        batch_results = results["batch_sizes"][batch_size]
        histories = batch_results["histories"]

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
            epochs,
            val_acc_iqm,
            color=color,
            linewidth=2,
            label=f"Batch {batch_size}",
        )
        plt.fill_between(
            epochs, val_acc_q1, val_acc_q3, color=color, alpha=alpha
        )

    plt.title("Validation Accuracy by Batch Size")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/batch_size_validation_accuracy.png", dpi=300)

    # Figure 2: Scatter plot of test accuracy vs. training time
    plt.figure(figsize=(10, 6))

    # Prepare data for scatter plot
    mean_accuracies = []
    mean_times = []
    marker_sizes = []

    for batch_size in batch_sizes:
        batch_results = results["batch_sizes"][batch_size]
        mean_accuracies.append(batch_results["mean_test_accuracy"])
        mean_times.append(batch_results["mean_training_time"])
        marker_sizes.append(int(batch_size))  # Size marker based on batch size

    # Normalize marker sizes for better visualization
    marker_sizes = [
        100 * (float(ms) / float(batch_sizes[-1])) for ms in marker_sizes
    ]

    # Create scatter plot
    scatter = plt.scatter(
        mean_times,
        mean_accuracies,
        s=marker_sizes,
        c=range(len(batch_sizes)),
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
    )

    # Add batch size labels
    for i, batch_size in enumerate(batch_sizes):
        plt.annotate(
            f"Batch {batch_size}",
            (mean_times[i], mean_accuracies[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("Test Accuracy vs. Training Time")
    plt.xlabel("Mean Training Time (seconds)")
    plt.ylabel("Mean Test Accuracy")
    plt.grid(True)

    # Add trend line
    z = np.polyfit(mean_times, mean_accuracies, 1)
    p = np.poly1d(z)
    plt.plot(mean_times, p(mean_times), "r--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("figures/batch_size_accuracy_vs_time.png", dpi=300)

    # Create summary table
    print("\nBATCH SIZE BENCHMARK SUMMARY")
    print("=" * 60)
    print(
        f"{'Batch Size':<10} {'Test Acc.':<10} {'Epochs':<10} {'Time (s)':<10}"
    )
    print("-" * 60)

    for batch_size in batch_sizes:
        batch_results = results["batch_sizes"][batch_size]
        print(
            f"{batch_size:<10} "
            f"{batch_results['mean_test_accuracy']:.4f}    "
            f"{batch_results['mean_epochs']:.1f}       "
            f"{batch_results['mean_training_time']:.1f}"
        )

    print("=" * 60)
    print("Visualizations created in 'figures' directory:")
    print("1. batch_size_validation_accuracy.png - Validation accuracy curves")
    print("2. batch_size_accuracy_vs_time.png - Accuracy vs. training time")


if __name__ == "__main__":
    visualize_batch_size_benchmark()
