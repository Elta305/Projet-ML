import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from utils import compute_stats


def main():
    """Create and save visualization of classification benchmark results."""
    with open("results/linear_classification_results.pkl", "rb") as f:
        results = pickle.load(f)

    params = results["parameters"]
    n_epochs = params["n_epochs"]
    n_features = params["n_features"]
    all_losses = np.array(
        [np.array(losses) for losses in results["all_losses"]]
    )
    all_accuracies = np.array(
        [np.array(accs) for accs in results["all_accuracies"]]
    )
    all_weights = results["weights"]
    all_biases = results["biases"]

    epochs = np.arange(n_epochs)
    loss_stats = []
    acc_stats = []

    for epoch_idx in range(n_epochs):
        epoch_losses = all_losses[:, epoch_idx]
        epoch_accs = all_accuracies[:, epoch_idx]
        loss_stats.append(compute_stats(epoch_losses))
        acc_stats.append(compute_stats(epoch_accs))

    iqm_losses = np.array([stats["iqm"] for stats in loss_stats])
    q1_losses = np.array([stats["q1"] for stats in loss_stats])
    q3_losses = np.array([stats["q3"] for stats in loss_stats])

    iqm_accs = np.array([stats["iqm"] for stats in acc_stats])
    q1_accs = np.array([stats["q1"] for stats in acc_stats])
    q3_accs = np.array([stats["q3"] for stats in acc_stats])

    final_accs = all_accuracies[:, -1]
    iqm_value = np.mean(np.percentile(final_accs, [25, 75]))
    q1_value = np.percentile(final_accs, 25)

    iqm_idx = np.argmin(np.abs(final_accs - iqm_value))
    final_accs_masked = np.delete(final_accs, iqm_idx)
    indices = np.delete(np.arange(len(final_accs)), iqm_idx)
    q1_idx = indices[np.argmin(np.abs(final_accs_masked - q1_value))]

    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )

    fig = plt.figure(figsize=(6, 9))

    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.1, 1.1, 1])
    gs.update(
        left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.6, wspace=0.4
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(
        epochs,
        q1_losses,
        q3_losses,
        color="#3498db",
        alpha=0.2,
        label="Loss IQR",
    )
    ax1.plot(
        epochs,
        iqm_losses,
        color="#3498db",
        linewidth=3,
        label="Loss IQM",
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax1.legend(loc="upper right", frameon=True)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(
        epochs,
        q1_accs,
        q3_accs,
        color="#3498db",
        alpha=0.2,
        label="Accuracy IQR",
    )
    ax2.plot(
        epochs,
        iqm_accs,
        color="#3498db",
        linewidth=3,
        label="Accuracy IQM",
    )
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification Accuracy")

    ax2.legend(loc="lower right", frameon=True)

    def generate_synthetic_data(seed_idx):
        np.random.seed(results["parameters"]["run_seeds"][seed_idx])
        centers = np.array(results["centers"][seed_idx])
        n_classes = centers.shape[0]
        n_samples = params["n_samples"]

        x = np.zeros((n_samples, n_features))
        y = np.zeros((n_samples, 1))

        samples_per_class = n_samples // n_classes
        for i in range(n_classes):
            start_idx = i * samples_per_class
            end_idx = (
                start_idx + samples_per_class
                if i < n_classes - 1
                else n_samples
            )

            class_samples = end_idx - start_idx
            x[start_idx:end_idx] = centers[i] + np.random.randn(
                class_samples, n_features
            )
            y[start_idx:end_idx] = i

        indices = np.random.permutation(n_samples)
        x = x[indices]
        y = y[indices]

        y = 2 * y - 1

        return x, y

    x_median, y_median = generate_synthetic_data(iqm_idx)
    x_worst, y_worst = generate_synthetic_data(q1_idx)

    def plot_decision_boundary(ax, x, y, weights, bias):
        w = np.array(weights).flatten()
        b = np.array(bias).flatten()[0]

        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )

        z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
        z = z.reshape(xx.shape)

        ax.contour(
            xx, yy, z, levels=[0], colors="black", linestyles="--", linewidths=2
        )

        y_pred = np.sign(np.dot(x, w) + b)

        mask = (y.flatten() == 1) & (y_pred.flatten() == 1)
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            color="#3498db",
            marker="o",
            alpha=0.15,
        )

        # True class +1, predicted -1 (red circle)
        mask = (y.flatten() == 1) & (y_pred.flatten() == -1)
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            color="#d66b6a",
            marker="o",
            alpha=0.25,
        )

        # True class -1, predicted +1 (blue cross)
        mask = (y.flatten() == -1) & (y_pred.flatten() == 1)
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            color="#3498db",
            marker="x",
            alpha=0.25,
        )

        # True class -1, predicted -1 (red cross)
        mask = (y.flatten() == -1) & (y_pred.flatten() == -1)
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            color="#d66b6a",
            marker="x",
            alpha=0.15,
        )

    ax3 = fig.add_subplot(gs[2, 0])
    plot_decision_boundary(
        ax3, x_median, y_median, all_weights[iqm_idx], all_biases[iqm_idx]
    )
    ax3.set_xlabel(r"$x_1$")
    ax3.set_ylabel(r"$x_2$")
    ax3.set_title("Median Run")

    ax4 = fig.add_subplot(gs[2, 1])
    plot_decision_boundary(
        ax4, x_worst, y_worst, all_weights[q1_idx], all_biases[q1_idx]
    )
    ax4.set_xlabel(r"$x_1$")
    ax4.set_ylabel(r"$x_2$")
    ax4.set_title("Worst Run")

    plt.savefig(
        "paper/figures/linear_classification.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


if __name__ == "__main__":
    main()
