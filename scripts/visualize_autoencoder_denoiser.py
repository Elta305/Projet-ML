import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils import add_gaussian_noise, create_autoencoder, get_mnist


def main():
    with open("results/autoencoder_denoiser.pkl", "rb") as f:
        results = pickle.load(f)

    best_model_state = results["model"]

    _, _, _, _, x_test, y_test = get_mnist()

    model = create_autoencoder(x_test.shape[1], 64, 5)
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

    noise_factor = 0.6

    for i, digit in enumerate(range(10)):
        idx = digit_examples[digit]
        noisy = add_gaussian_noise(x_test[idx : idx + 1], noise_factor).reshape(
            28, 28
        )
        reconstructed = model.forward(
            add_gaussian_noise(x_test[idx : idx + 1], noise_factor)
        ).reshape(28, 28)

        axes[i, 0].imshow(noisy, cmap="gray")
        axes[i, 0].set_title(f"Noisy: {digit}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(reconstructed, cmap="gray")
        axes[i, 1].set_title(f"Reconstructed: {digit}")
        axes[i, 1].axis("off")

    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    plt.savefig(
        "paper/figures/mnist_high_denoising_reconstructions.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    # plt.show()


if __name__ == "__main__":
    main()
