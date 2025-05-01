import logging
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mlp.linear import Linear
from mlp.loss import MSELoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_linear_data(n_samples=1000, noise=0.5, seed=None):
    """Generate synthetic data for linear regression with optional random seed."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.rand(n_samples, 1) * 10
    true_weight = np.random.uniform(-5, 5)
    true_bias = np.random.uniform(-10, 10)
    y = true_weight * X + true_bias + noise * np.random.randn(n_samples, 1)

    logger.info(
        f"Generated {n_samples} data points with weight={true_weight:.4f}, bias={true_bias:.4f}"
    )
    return X, y, true_weight, true_bias


def train_linear_model(X, y, learning_rate=0.01, n_epochs=500):
    """Train a linear regression model."""
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = Linear(input_dim, output_dim)
    loss_fn = MSELoss()
    losses = []

    for epoch in range(n_epochs):
        y_pred = model.forward(X)
        loss_values = loss_fn.forward(y, y_pred)
        avg_loss = np.mean(loss_values)
        losses.append(avg_loss)

        grad = loss_fn.backward(y, y_pred)
        model.zero_grad()
        model.backward_update_gradient(X, grad)
        model.update_parameters(learning_rate)

        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.6f}")

    logger.info(f"Final loss: {losses[-1]:.6f}")
    return model, losses


def main():
    # Parameters for reproducibility
    n_runs = 15
    n_samples = 200
    noise = 0.5
    learning_rate = 0.01
    n_epochs = 1000
    random_seed = 42  # Set a master seed for reproducibility

    # Create a comprehensive title - short enough to fit
    title = f"Linear Regression Ensemble \n (n={n_runs}, samples={n_samples}, lr={learning_rate}, epochs={n_epochs})"

    logger.info(f"Starting linear regression ensemble: {title}")

    # Set the master seed
    np.random.seed(random_seed)

    # Generate run-specific seeds
    run_seeds = np.random.randint(0, 10000, size=n_runs)

    # Store results from all runs
    all_losses = []
    all_weights = []
    all_biases = []

    # Run multiple trials
    for run in range(n_runs):
        logger.info(f"\nRun {run + 1}/{n_runs} (seed={run_seeds[run]})")

        # Generate data with run-specific seed
        X, y, true_weight, true_bias = generate_linear_data(
            n_samples=n_samples, noise=noise, seed=run_seeds[run]
        )

        # Train model
        model, losses = train_linear_model(
            X, y, learning_rate=learning_rate, n_epochs=n_epochs
        )

        # Save results
        all_losses.append(losses)
        recovered_weight = model._parameters[0, 0]
        recovered_bias = model.bias[0, 0] if hasattr(model, "bias") else 0
        all_weights.append(recovered_weight)
        all_biases.append(recovered_bias)

        logger.info(
            f"Run {run + 1} - True: weight={true_weight:.4f}, bias={true_bias:.4f} | "
            f"Learned: weight={recovered_weight:.4f}, bias={recovered_bias:.4f}"
        )

    # Convert to arrays for easier processing
    all_losses = np.array(all_losses)

    # Calculate statistics for losses
    q1_losses = np.percentile(all_losses, 25, axis=0)
    q3_losses = np.percentile(all_losses, 75, axis=0)

    # Calculate IQM as the mean of values between Q1 and Q3 (true interquartile mean)
    iqm_values = []
    for epoch in range(n_epochs):
        epoch_losses = all_losses[:, epoch]
        q1 = q1_losses[epoch]
        q3 = q3_losses[epoch]
        # Select values between Q1 and Q3
        interquartile_values = epoch_losses[
            (epoch_losses >= q1) & (epoch_losses <= q3)
        ]
        iqm_values.append(np.mean(interquartile_values))
    iqm_losses = np.array(iqm_values)

    # Find best and worst runs based on final loss
    final_losses = all_losses[:, -1]
    best_idx = np.argmin(final_losses)
    worst_idx = np.argmax(final_losses)

    logger.info(
        f"\nBest run: {best_idx + 1} (loss={final_losses[best_idx]:.6f})"
    )
    logger.info(
        f"Worst run: {worst_idx + 1} (loss={final_losses[worst_idx]:.6f})"
    )

    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "text.usetex": True,
        }
    )

    # Create figure with more space for margins
    plt.figure(figsize=(16, 10))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.1, hspace=0.4)

    # Plot 1: Loss curves with statistics (top subplot)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    epochs = np.arange(n_epochs)

    # Fill between Q1 and Q3
    ax1.fill_between(
        epochs,
        q1_losses,
        q3_losses,
        color="#5C97C0",
        alpha=0.3,
        label="Q1-Q3 Range",
    )

    # Plot IQM line (should be within the Q1-Q3 range now)
    ax1.plot(
        epochs,
        iqm_losses,
        color="#5C97C0",
        linewidth=2,
        label="Interquartile Mean",
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title("Training Loss Over Time", pad=20)
    ax1.legend()

    # Ensure all elements are visible by setting margins
    ax1.margins(x=0.01, y=0.1)

    # Generate data for best and worst runs
    X_best, y_best, _, _ = generate_linear_data(
        n_samples=n_samples, noise=noise, seed=run_seeds[best_idx]
    )
    X_worst, y_worst, _, _ = generate_linear_data(
        n_samples=n_samples, noise=noise, seed=run_seeds[worst_idx]
    )

    # For visualization, use clean X range
    X_viz = np.linspace(0, 10, 100).reshape(-1, 1)

    # Plot 2: Best run (bottom left)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.scatter(X_best, y_best, color="#5C97C0", alpha=0.2)
    ax2.plot(
        X_viz,
        X_viz * all_weights[best_idx] + all_biases[best_idx],
        color="#5C97C0",
        linewidth=2,
    )
    ax2.set_xlabel("$\\mathbf{x}$")
    ax2.set_ylabel("$y$")
    ax2.set_title("Best Run")
    ax2.margins(x=0.05)

    # Plot 3: Worst run (bottom right)
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.scatter(X_worst, y_worst, color="#5C97C0", alpha=0.2)
    ax3.plot(
        X_viz,
        X_viz * all_weights[worst_idx] + all_biases[worst_idx],
        color="#5C97C0",
        linewidth=2,
    )
    ax3.set_xlabel("$\\mathbf{x}$")
    ax3.set_ylabel("$y$")
    ax3.set_title("Worst Run")
    ax3.margins(x=0.05)

    plt.show()

    logger.info("Ensemble test completed successfully")


if __name__ == "__main__":
    main()
