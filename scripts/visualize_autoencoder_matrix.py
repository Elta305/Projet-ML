import pickle

import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("results/mnist_autoencoder_matrix.pkl", "rb") as f:
        results = pickle.load(f)

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

    latent_dims = sorted({key[0] for key in results})
    depths = sorted({key[1] for key in results})

    loss_types = {key[2] for key in results}
    matrices = {}

    for loss_type in loss_types:
        matrix = np.full((len(latent_dims), len(depths)), np.nan)

        for (latent_dim, depth, lt), data in results.items():
            if lt == loss_type:
                row_idx = latent_dims.index(latent_dim)
                col_idx = depths.index(depth)
                matrix[row_idx, col_idx] = data["best_loss"]

        matrices[loss_type] = matrix

    plt.figure(figsize=(12, 6))

    for i, (loss_type, matrix) in enumerate(matrices.items()):
        plt.subplot(1, len(matrices), i + 1)
        im = plt.imshow(matrix, cmap="viridis")

        plt.colorbar(im, label="Average Loss", shrink=0.6)

        plt.title(f"{loss_type}")
        plt.xlabel("Depth")
        plt.ylabel("Latent Dimension")

        plt.xticks(range(len(depths)), depths)
        plt.yticks(range(len(latent_dims)), latent_dims)

    plt.tight_layout()
    plt.savefig(
        "paper/figures/autoencoder_matrix.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


if __name__ == "__main__":
    main()
