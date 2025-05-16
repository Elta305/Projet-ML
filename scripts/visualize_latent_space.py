import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from utils import create_autoencoder, get_mnist


def extract_latent_representations(model, x, depth):
    _ = model.forward(x)
    encoder_modules_count = 2 * (depth + 1)
    return model.inputs[encoder_modules_count]


def main():
    with open("results/mnist_autoencoder_matrix.pkl", "rb") as f:
        results = pickle.load(f)
    key = (4, 1, "MSELoss")
    best_model_state = results[key]["best_model"]

    _, _, _, _, x_test, y_test = get_mnist()

    model = create_autoencoder(x_test.shape[1], key[0], key[1])
    model.load_state_dict(best_model_state)
    latent_vectors = extract_latent_representations(model, x_test, key[1])

    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(latent_vectors, y_test)
    predictions = knn.predict(latent_vectors)

    cm = confusion_matrix(y_test, predictions)

    total_samples = len(y_test)

    print(f"{'Class':<6}{'TP':<6}{'FP':<6}{'FN':<6}{'TN':<6}{'Accuracy':<10}")
    print("-" * 50)

    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
    sum_tn = 0
    total = 0

    for i in range(10):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples - tp - fp - fn

        row_total = tp + fp + fn + tn

        accuracy = (tp + tn) / row_total

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        sum_tn += tn

        total += row_total

        print(f"{i:<6}{tp:<6}{fp:<6}{fn:<6}{tn:<6}{accuracy:.4f}")

    print("-" * 50)
    overall_accuracy = (sum_tp + sum_tn) / total

    print(
        f"{'TOTAL':<6}{sum_tp:<6}{sum_fp:<6}{sum_fn:<6}{sum_tn:<6}{overall_accuracy:.4f}"
    )

    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    latent_2d = (
        2
        * (latent_2d - np.min(latent_2d, axis=0))
        / (np.max(latent_2d, axis=0) - np.min(latent_2d, axis=0))
        - 1
    )

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

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_2d[:, 0], latent_2d[:, 1], c=y_test, cmap="tab10", alpha=0.6, s=5
    )
    plt.colorbar(scatter, label="Digit Class")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(alpha=0.3)

    Path("results").mkdir(exist_ok=True)
    plt.savefig(
        "paper/figures/latent_space_tsne.svg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


if __name__ == "__main__":
    main()
