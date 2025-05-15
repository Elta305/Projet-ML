import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils import compute_stats


def visualize_hidden_size_benchmark():
    """Create visualizations for hidden layer size benchmark results."""
    # Load results
    with open("results/hidden_size_benchmark_results.pkl", "rb") as f:
        results = pickle.load(f)

    hidden_sizes = list(results["hidden_sizes"].keys())
    hidden_sizes = [
        int(hs) for hs in hidden_sizes
    ]  # Convert to integers for proper sorting
    hidden_sizes.sort()  # Ensure they're sorted
    hidden_sizes = [str(hs) for hs in hidden_sizes]  # Convert back to strings

    # Create figure directory if it doesn't exist
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Prepare data for learning curves
    max_epochs = max(
        [
            max(
                [
                    h["total_epochs"]
                    for h in results["hidden_sizes"][hs]["histories"]
                ]
            )
            for hs in hidden_sizes
        ]
    )

    # Colors for different hidden sizes
    colors = plt.cm.viridis(np.linspace(0, 1, len(hidden_sizes)))

    # Figure 1: Validation Accuracy Curves
    plt.figure(figsize=(10, 6))

    for i, hidden_size in enumerate(hidden_sizes):
        size_results = results["hidden_sizes"][hidden_size]
        histories = size_results["histories"]

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
            label=f"Hidden Size ({hidden_size},{hidden_size})",
        )
        plt.fill_between(
            epochs, val_acc_q1, val_acc_q3, color=color, alpha=alpha
        )

    plt.title("Validation Accuracy by Hidden Layer Size")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/hidden_size_validation_accuracy.png", dpi=300)

    # Figure 2: Performance vs. Number of Parameters
    plt.figure(figsize=(10, 6))

    # Extract data
    param_counts = [
        results["hidden_sizes"][hs]["num_parameters"] for hs in hidden_sizes
    ]
    mean_accuracies = [
        results["hidden_sizes"][hs]["mean_test_accuracy"] for hs in hidden_sizes
    ]
    std_accuracies = [
        results["hidden_sizes"][hs]["std_test_accuracy"] for hs in hidden_sizes
    ]

    # Plot performance vs parameters
    plt.errorbar(
        param_counts,
        mean_accuracies,
        yerr=std_accuracies,
        fmt="o-",
        capsize=5,
        linewidth=2,
        markersize=10,
    )

    # Add hidden size labels
    for i, hidden_size in enumerate(hidden_sizes):
        plt.annotate(
            f"({hidden_size},{hidden_size})",
            (param_counts[i], mean_accuracies[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("Test Accuracy vs. Number of Parameters")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Mean Test Accuracy")
    plt.xscale("log")  # Log scale for parameter count
    plt.grid(True)

    # Highlight diminishing returns
    plt.tight_layout()
    plt.savefig("figures/hidden_size_parameters_vs_accuracy.png", dpi=300)

    # Figure 3: Generalization Gap
    plt.figure(figsize=(10, 6))

    gen_gaps = [
        results["hidden_sizes"][hs]["mean_generalization_gap"]
        for hs in hidden_sizes
    ]
    gap_stds = [
        results["hidden_sizes"][hs]["std_generalization_gap"]
        for hs in hidden_sizes
    ]

    bars = plt.bar(
        range(len(hidden_sizes)),
        gen_gaps,
        yerr=gap_stds,
        capsize=5,
        color=colors,
        alpha=0.7,
    )

    plt.xticks(
        range(len(hidden_sizes)), [f"({hs},{hs})" for hs in hidden_sizes]
    )
    plt.title("Generalization Gap (Train-Test Accuracy Difference)")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Generalization Gap")
    plt.grid(axis="y")

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.002,
            f"{gen_gaps[i]:.4f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    plt.tight_layout()
    plt.savefig("figures/hidden_size_generalization_gap.png", dpi=300)

    # Create summary table
    print("\nHIDDEN LAYER SIZE BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"{'Hidden Size':<15} {'Parameters':<12} {'Test Acc.':<10} {'Gen. Gap':<10} {'Epochs':<8}"
    )
    print("-" * 70)

    for hidden_size in hidden_sizes:
        size_results = results["hidden_sizes"][hidden_size]
        print(
            f"({hidden_size},{hidden_size})       "
            f"{size_results['num_parameters']:,}      "
            f"{size_results['mean_test_accuracy']:.4f}     "
            f"{size_results['mean_generalization_gap']:.4f}     "
            f"{size_results['mean_epochs']:.1f}"
        )

    print("=" * 70)
    print("Visualizations created in 'figures' directory:")
    print("1. hidden_size_validation_accuracy.png - Validation accuracy curves")
    print("2. hidden_size_parameters_vs_accuracy.png - Accuracy vs. parameters")
    print("3. hidden_size_generalization_gap.png - Generalization gap analysis")


if __name__ == "__main__":
    visualize_hidden_size_benchmark()
