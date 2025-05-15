import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import create_autoencoder, get_batches, get_mnist

from mlp.loss import MSELoss
from mlp.optim import Adam


def train_autoencoder_mnist(
    x_train,
    x_val,
    latent_dim,
    depth,
    loss_fn,
    batch_size,
    learning_rate,
    max_epochs,
    patience,
    seed,
):
    np.random.seed(seed)

    input_dim = x_train.shape[1]

    model = create_autoencoder(input_dim, latent_dim, depth)
    optimizer = Adam(model, loss_fn, eps=learning_rate)

    best_val_loss = float("inf")
    best_model_state = None

    for _ in tqdm(range(max_epochs), leave=False):
        batches = get_batches(x_train, x_train, batch_size)
        for batch_x, _ in batches:
            optimizer.step(batch_x, batch_x)

        val_output = model.forward(x_val)
        val_loss = loss_fn.forward(x_val, val_output)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)

    return model


def evaluate_autoencoder(model, x_test):
    reconstructed = model.forward(x_test)

    # Calculate per-sample reconstruction error
    errors = np.mean((x_test - reconstructed) ** 2, axis=1)

    # Find indices for worst, median, and best reconstructions
    worst_idx = np.argmax(errors)
    median_idx = np.argsort(errors)[len(errors) // 2]
    best_idx = np.argmin(errors)

    # Plot the original and reconstructed images
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    titles = ["Worst", "Median", "Best"]
    indices = [worst_idx, median_idx, best_idx]

    for i, (title, idx) in enumerate(zip(titles, indices, strict=False)):
        # Plot original image
        axs[i, 0].imshow(x_test[idx].reshape(28, 28), cmap="gray")
        axs[i, 0].set_title(f"{title} - Original")
        axs[i, 0].axis("off")

        # Plot reconstructed image
        axs[i, 1].imshow(reconstructed[idx].reshape(28, 28), cmap="gray")
        axs[i, 1].set_title(
            f"{title} - Reconstructed (Error: {errors[idx]:.4f})"
        )
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

    # Return the overall loss
    return MSELoss().forward(x_test, reconstructed)


def main():
    n_runs = 1
    random_seed = 42

    np.random.seed(random_seed)
    run_seeds = np.random.randint(0, 10000, size=n_runs)

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    losses = []
    for run in tqdm(range(n_runs)):
        model = train_autoencoder_mnist(
            x_train=x_train,
            x_val=x_val,
            latent_dim=32,
            depth=3,
            loss_fn=MSELoss(),
            batch_size=128,
            learning_rate=0.001,
            max_epochs=100,
            patience=5,
            seed=run_seeds[run],
        )

        loss = evaluate_autoencoder(model, x_test)
        losses.append(loss)

    print(losses)


if __name__ == "__main__":
    main()
