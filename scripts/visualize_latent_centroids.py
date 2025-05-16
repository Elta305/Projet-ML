import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils import create_autoencoder, get_mnist


def extract_latent_representations(model, x, depth):
    _ = model.forward(x)
    encoder_modules_count = 2 * (depth + 1)
    return model.inputs[encoder_modules_count]


def decode_from_latent(model, z, depth):
    x = z
    decoder_start_idx = 2 * (depth + 1)
    for i in range(decoder_start_idx, len(model.modules)):
        x = model.modules[i].forward(x)
    return x


def main():
    with open("results/mnist_autoencoder_matrix.pkl", "rb") as f:
        results = pickle.load(f)

    key = (64, 5, "MSELoss")
    best_model_state = results[key]["best_model"]

    _, _, _, _, x_test, y_test = get_mnist()
    model = create_autoencoder(x_test.shape[1], key[0], key[1])
    model.load_state_dict(best_model_state)

    latent_vectors = extract_latent_representations(model, x_test, key[1])

    centroids = []
    for digit in range(10):
        digit_indices = np.where(y_test == digit)[0]
        centroid = np.mean(latent_vectors[digit_indices], axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    interpolated_grid = np.zeros((10, 10, key[0]))
    reconstructed_grid = np.zeros((10, 10, 28, 28))

    for i in range(10):
        interpolated_grid[i, i] = centroids[i]
        reconstructed = decode_from_latent(
            model, centroids[i].reshape(1, -1), key[1]
        )
        reconstructed_grid[i, i] = reconstructed.reshape(28, 28)

    for i in range(10):
        for j in range(10):
            if i != j:
                interpolated = 0.5 * centroids[i] + 0.5 * centroids[j]
                interpolated_grid[i, j] = interpolated
                reconstructed = decode_from_latent(
                    model, interpolated.reshape(1, -1), key[1]
                )
                reconstructed_grid[i, j] = reconstructed.reshape(28, 28)

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

    fig, axes = plt.subplots(10, 10, figsize=(15, 15))

    for i in range(10):
        for j in range(10):
            ax = axes[i, j]
            ax.imshow(reconstructed_grid[i, j], cmap="gray")
            ax.axis("off")
            if i == j:
                ax.spines["bottom"].set_color("red")
                ax.spines["top"].set_color("red")
                ax.spines["right"].set_color("red")
                ax.spines["left"].set_color("red")
                ax.spines["bottom"].set_linewidth(2)
                ax.spines["top"].set_linewidth(2)
                ax.spines["right"].set_linewidth(2)
                ax.spines["left"].set_linewidth(2)

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)

    Path("results").mkdir(exist_ok=True)
    plt.savefig(
        "paper/figures/latent_centroids_interpolation.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


if __name__ == "__main__":
    main()
