import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute statistical measures from an array of values."""
    values = np.array(values)
    q1, q3 = np.percentile(values, [25, 75])
    mask = (values >= q1) & (values <= q3)
    interquartile_values = values[mask]
    iqm = np.mean(interquartile_values)
    mins = np.min(values)
    maxs = np.max(values)
    return {"iqm": iqm, "q1": q1, "q3": q3, "min": mins, "max": maxs}


def main():
    """Create and save visualization of linear regression benchmark results."""
    with open("results/linear_regression_results.pkl", "rb") as f:
        results = pickle.load(f)

    params = results["parameters"]
    n_samples = params["n_samples"]
    n_epochs = params["n_epochs"]
    all_losses = np.array(results["all_losses"])
    all_weights = results["recovered_weights"]
    all_biases = results["recovered_biases"]
    true_weights = results["true_weights"]
    true_biases = results["true_biases"]

    epochs = np.arange(n_epochs)
    epoch_stats = []
    for epoch_idx in range(n_epochs):
        epoch_losses = all_losses[:, epoch_idx]
        epoch_stats.append(compute_stats(epoch_losses))

    iqm_losses = np.array([stats["iqm"] for stats in epoch_stats])
    q1_losses = np.array([stats["q1"] for stats in epoch_stats])
    q3_losses = np.array([stats["q3"] for stats in epoch_stats])

    final_losses = all_losses[:, -1]

    median_value = np.median(final_losses)
    median_idx = np.argmin(np.abs(final_losses - median_value))
    worst_idx = np.argmax(final_losses)

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

    fig = plt.figure(figsize=(6, 6))

    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1])

    gs.update(
        left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.5, wspace=0.4
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

    ax1.legend(loc="upper right", frameon=True)

    def generate_synthetic_data(seed):
        np.random.seed(seed)
        X = np.random.rand(n_samples, 1) * 10
        true_weight = true_weights[seed]
        true_bias = true_biases[seed]
        noise = params["noise"]

        y = true_weight * X + true_bias + noise * np.random.randn(n_samples, 1)

        return X, y

    X_median, y_median = generate_synthetic_data(median_idx)
    X_worst, y_worst = generate_synthetic_data(worst_idx)
    X_viz = np.linspace(0, 10, 100).reshape(-1, 1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(X_median, y_median, color="#3498db", alpha=0.2)
    ax2.plot(
        X_viz,
        X_viz * all_weights[median_idx] + all_biases[median_idx],
        "--",
        color="black",
        linewidth=2,
    )
    ax2.set_xlabel(r"$\textbf{x}$")
    ax2.set_ylabel(r"$y$")
    ax2.set_title("Median Run")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(X_worst, y_worst, color="#3498db", alpha=0.2)
    ax3.plot(
        X_viz,
        X_viz * all_weights[worst_idx] + all_biases[worst_idx],
        "--",
        color="black",
        linewidth=2,
    )
    ax3.set_xlabel(r"$\textbf{x}$")
    ax3.set_ylabel("")
    ax3.set_title("Worst Run")

    plt.savefig(
        "paper/figures/linear_regression.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


if __name__ == "__main__":
    main()
