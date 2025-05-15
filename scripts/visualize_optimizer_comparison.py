import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import compute_stats


def visualize_optimizer_benchmark():
    """Create visualizations for optimizer benchmark results."""
    # Load results
    with open("results/optimizer_benchmark_results.pkl", "rb") as f:
        results = pickle.load(f)

    optimizer_names = list(results["optimizers"].keys())

    # Set up plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )

    # Create figure directory if it doesn't exist
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Prepare data for learning curves
    max_epochs = max(
        [
            max(
                [
                    h["total_epochs"]
                    for h in results["optimizers"][opt]["histories"]
                ]
            )
            for opt in optimizer_names
        ]
    )

    # Colors for different optimizers
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    # Figure 1: Training and Validation Loss Curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    for i, optimizer_name in enumerate(optimizer_names):
        optimizer_results = results["optimizers"][optimizer_name]
        histories = optimizer_results["histories"]

        # Pad histories to the same length
        train_losses = np.zeros((len(histories), max_epochs))
        val_losses = np.zeros((len(histories), max_epochs))
        train_accs = np.zeros((len(histories), max_epochs))
        val_accs = np.zeros((len(histories), max_epochs))

        for j, history in enumerate(histories):
            epochs_run = len(history["train_loss"])
            train_losses[j, :epochs_run] = history["train_loss"]
            val_losses[j, :epochs_run] = history["val_loss"]
            train_accs[j, :epochs_run] = history["train_accuracy"]
            val_accs[j, :epochs_run] = history["val_accuracy"]

            # Pad with the last value
            if epochs_run < max_epochs:
                train_losses[j, epochs_run:] = history["train_loss"][-1]
                val_losses[j, epochs_run:] = history["val_loss"][-1]
                train_accs[j, epochs_run:] = history["train_accuracy"][-1]
                val_accs[j, epochs_run:] = history["val_accuracy"][-1]

        # Calculate statistics for each epoch
        epochs = np.arange(1, max_epochs + 1)
        train_loss_stats = [
            compute_stats(train_losses[:, e]) for e in range(max_epochs)
        ]
        val_loss_stats = [
            compute_stats(val_losses[:, e]) for e in range(max_epochs)
        ]
        train_acc_stats = [
            compute_stats(train_accs[:, e]) for e in range(max_epochs)
        ]
        val_acc_stats = [
            compute_stats(val_accs[:, e]) for e in range(max_epochs)
        ]

        # Extract IQM and IQR
        train_loss_iqm = [s["iqm"] for s in train_loss_stats]
        train_loss_q1 = [s["q1"] for s in train_loss_stats]
        train_loss_q3 = [s["q3"] for s in train_loss_stats]

        val_loss_iqm = [s["iqm"] for s in val_loss_stats]
        val_loss_q1 = [s["q1"] for s in val_loss_stats]
        val_loss_q3 = [s["q3"] for s in val_loss_stats]

        train_acc_iqm = [s["iqm"] for s in train_acc_stats]
        train_acc_q1 = [s["q1"] for s in train_acc_stats]
        train_acc_q3 = [s["q3"] for s in train_acc_stats]

        val_acc_iqm = [s["iqm"] for s in val_acc_stats]
        val_acc_q1 = [s["q1"] for s in val_acc_stats]
        val_acc_q3 = [s["q3"] for s in val_acc_stats]

        # Plot loss curves
        color = colors[i]
        alpha = 0.2  # Alpha for IQR shading

        # Plot training loss
        ax1.plot(
            epochs,
            train_loss_iqm,
            color=color,
            linestyle="-",
            linewidth=2,
            label=f"{optimizer_name} (Train)",
        )
        ax1.fill_between(
            epochs, train_loss_q1, train_loss_q3, color=color, alpha=alpha
        )

        # Plot validation loss
        ax1.plot(
            epochs,
            val_loss_iqm,
            color=color,
            linestyle="--",
            linewidth=2,
            label=f"{optimizer_name} (Val)",
        )
        ax1.fill_between(
            epochs, val_loss_q1, val_loss_q3, color=color, alpha=alpha
        )

        # Plot accuracy curves
        ax2.plot(
            epochs,
            train_acc_iqm,
            color=color,
            linestyle="-",
            linewidth=2,
            label=f"{optimizer_name} (Train)",
        )
        ax2.fill_between(
            epochs, train_acc_q1, train_acc_q3, color=color, alpha=alpha
        )

        ax2.plot(
            epochs,
            val_acc_iqm,
            color=color,
            linestyle="--",
            linewidth=2,
            label=f"{optimizer_name} (Val)",
        )
        ax2.fill_between(
            epochs, val_acc_q1, val_acc_q3, color=color, alpha=alpha
        )

    # Customize loss plot
    ax1.set_title("Loss Curves")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # Customize accuracy plot
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(
        "figures/optimizer_learning_curves.png", dpi=300, bbox_inches="tight"
    )

    # Figure 2: Boxplot of Test Accuracies
    plt.figure(figsize=(10, 6))
    test_accuracies = [
        results["optimizers"][opt]["test_accuracies"] for opt in optimizer_names
    ]

    plt.boxplot(
        test_accuracies,
        labels=optimizer_names,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red"),
    )

    plt.title("Test Accuracy Distribution")
    plt.ylabel("Accuracy")
    plt.xlabel("Optimizer")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(
        "figures/optimizer_test_accuracy_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Figure 3: Convergence Time Comparison
    plt.figure(figsize=(12, 6))

    epochs_data = []
    for optimizer_name in optimizer_names:
        epochs_data.append(
            results["optimizers"][optimizer_name]["convergence_epochs"]
        )

    plt.boxplot(
        epochs_data,
        labels=optimizer_names,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red"),
    )

    plt.title("Convergence Epochs Comparison")
    plt.ylabel("Number of Epochs")
    plt.xlabel("Optimizer")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(
        "figures/optimizer_convergence_epochs.png", dpi=300, bbox_inches="tight"
    )

    # Create summary table
    summary_data = []
    for optimizer_name in optimizer_names:
        optimizer_results = results["optimizers"][optimizer_name]
        summary_data.append(
            {
                "Optimizer": optimizer_name,
                "Mean Test Accuracy": f"{optimizer_results['mean_test_accuracy']:.4f}",
                "Median Test Accuracy": f"{optimizer_results['median_test_accuracy']:.4f}",
                "Test Acc. Std Dev": f"{optimizer_results['std_test_accuracy']:.4f}",
                "Mean Epochs": f"{optimizer_results['mean_epochs']:.1f}",
                "Median Epochs": f"{optimizer_results['median_epochs']:.1f}",
                "Mean Time (s)": f"{optimizer_results['mean_time']:.1f}",
                "Median Time (s)": f"{optimizer_results['median_time']:.1f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("figures/optimizer_benchmark_summary.csv", index=False)

    # Create a styled HTML table for the report
    html_table = summary_df.to_html(index=False)
    with open("figures/optimizer_benchmark_summary.html", "w") as f:
        f.write(html_table)

    print("Visualizations created in 'figures' directory:")
    print("1. optimizer_learning_curves.png - Learning curves with IQM and IQR")
    print("2. optimizer_test_accuracy_boxplot.png - Boxplot of test accuracies")
    print("3. optimizer_convergence_epochs.png - Boxplot of convergence epochs")
    print("4. optimizer_benchmark_summary.csv/html - Summary tables")


if __name__ == "__main__":
    visualize_optimizer_benchmark()
