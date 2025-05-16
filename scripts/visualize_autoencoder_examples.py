import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils import get_mnist, create_autoencoder


def visualize_reconstructions():
    with open("results/mnist_autoencoder_matrix.pkl", "rb") as f:
        results = pickle.load(f)

    key = (4, 1, "MSELoss")
    best_model_state = results[key]["best_model"]

    _, _, _, _, x_test, y_test = get_mnist()

    model = create_autoencoder(x_test.shape[1], key[0], key[1])
    model.load_state_dict(best_model_state)

    digit_examples = {}

    for digit in range(10):
        digit_indices = np.where(y_test == digit)[0]
        if len(digit_indices) > 0:
            idx = digit_indices[0]
            digit_examples[digit] = idx

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 20,
            "font.size": 20,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    fig, axes = plt.subplots(10, 2, figsize=(8, 20))

    for i, digit in enumerate(range(10)):
        idx = digit_examples[digit]

        original = x_test[idx].reshape(28, 28)

        reconstructed = model.forward(x_test[idx : idx + 1]).reshape(28, 28)

        axes[i, 0].imshow(original, cmap="gray")
        axes[i, 0].set_title(f"Original: {digit}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(reconstructed, cmap="gray")
        axes[i, 1].set_title(f"Reconstructed: {digit}")
        axes[i, 1].axis("off")

    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    plt.savefig(
        "paper/figures/mnist_reconstructions.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


if __name__ == "__main__":
    visualize_reconstructions()
