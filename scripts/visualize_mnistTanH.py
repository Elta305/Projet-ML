import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from utils import compute_stats, get_batches, get_mnist

from mlp.activation import TanH, Softmax
from mlp.linear import Linear
from mlp.loss import CrossEntropyLoss
from mlp.optim import Adam
from mlp.sequential import Sequential
from mlp.utils import one_hot_encoding


def main():
    """Create and save visualization of MNIST classification results with
    misclassified examples."""
    with open("results/mnist_classificationtanh_results.pkl", "rb") as f:
        results = pickle.load(f)

    params = results["parameters"]
    n_runs = params["n_runs"]
    all_histories = results["all_histories"]

    actual_epochs = [len(history["train_loss"]) for history in all_histories]
    max_actual_epochs = max(actual_epochs)

    all_train_losses = np.zeros((n_runs, max_actual_epochs))
    all_val_losses = np.zeros((n_runs, max_actual_epochs))
    all_train_accs = np.zeros((n_runs, max_actual_epochs))
    all_val_accs = np.zeros((n_runs, max_actual_epochs))

    for i, history in enumerate(all_histories):
        n_epochs_run = len(history["train_loss"])
        all_train_losses[i, :n_epochs_run] = history["train_loss"]
        all_val_losses[i, :n_epochs_run] = history["val_loss"]
        all_train_accs[i, :n_epochs_run] = history["train_accuracy"]
        all_val_accs[i, :n_epochs_run] = history["val_accuracy"]

        # Pad with the last value if needed
        if n_epochs_run < max_actual_epochs:
            all_train_losses[i, n_epochs_run:] = history["train_loss"][-1]
            all_val_losses[i, n_epochs_run:] = history["val_loss"][-1]
            all_train_accs[i, n_epochs_run:] = history["train_accuracy"][-1]
            all_val_accs[i, n_epochs_run:] = history["val_accuracy"][-1]

    epochs = np.arange(1, max_actual_epochs + 1)

    train_loss_stats = []
    val_loss_stats = []
    train_acc_stats = []
    val_acc_stats = []

    for epoch_idx in range(max_actual_epochs):
        train_loss_stats.append(compute_stats(all_train_losses[:, epoch_idx]))
        val_loss_stats.append(compute_stats(all_val_losses[:, epoch_idx]))
        train_acc_stats.append(compute_stats(all_train_accs[:, epoch_idx]))
        val_acc_stats.append(compute_stats(all_val_accs[:, epoch_idx]))

    train_loss_iqm = np.array([stats["iqm"] for stats in train_loss_stats])
    train_loss_q1 = np.array([stats["q1"] for stats in train_loss_stats])
    train_loss_q3 = np.array([stats["q3"] for stats in train_loss_stats])

    val_loss_iqm = np.array([stats["iqm"] for stats in val_loss_stats])
    val_loss_q1 = np.array([stats["q1"] for stats in val_loss_stats])
    val_loss_q3 = np.array([stats["q3"] for stats in val_loss_stats])

    train_acc_iqm = np.array([stats["iqm"] for stats in train_acc_stats])
    train_acc_q1 = np.array([stats["q1"] for stats in train_acc_stats])
    train_acc_q3 = np.array([stats["q3"] for stats in train_acc_stats])

    val_acc_iqm = np.array([stats["iqm"] for stats in val_acc_stats])
    val_acc_q1 = np.array([stats["q1"] for stats in val_acc_stats])
    val_acc_q3 = np.array([stats["q3"] for stats in val_acc_stats])

    test_accuracy = results["test_accuracy"]

    _, _, _, _, x_test, y_test = get_mnist()

    hidden_dims = params["hidden_dims"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    n_epochs = params["n_epochs"]

    best_run_idx = np.argmax(results["all_val_accuracies"])
    best_run_seed = params["run_seeds"][best_run_idx]

    x_train, y_train, x_val, y_val, _, _ = get_mnist()

    np.random.seed(best_run_seed)

    num_classes = 10
    y_train_onehot = one_hot_encoding(y_train, num_classes)

    input_dim = x_train.shape[1]
    output_dim = num_classes

    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(Linear(prev_dim, hidden_dim))
        layers.append(TanH())
        prev_dim = hidden_dim
    layers.append(Linear(prev_dim, output_dim))
    layers.append(Softmax())

    model = Sequential(*layers)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model, loss_fn, eps=learning_rate)

    for _ in range(n_epochs):
        batches = get_batches(x_train, y_train_onehot, batch_size)
        for batch_x, batch_y in batches:
            optimizer.step(batch_x, batch_y)

    test_output = model.forward(x_test)
    test_predictions = np.argmax(test_output, axis=1)

    misclassified_indices = np.where(test_predictions != y_test)[0]

    selected_indices = misclassified_indices[:2]
    misclassified_images = x_test[selected_indices].reshape(-1, 28, 28)
    true_labels = y_test[selected_indices]
    predicted_labels = test_predictions[selected_indices]

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
        train_loss_q1,
        train_loss_q3,
        color="#3498db",
        alpha=0.2,
    )
    ax1.plot(
        epochs,
        train_loss_iqm,
        color="#3498db",
        linewidth=2.5,
        label="Train Loss",
    )

    ax1.fill_between(
        epochs,
        val_loss_q1,
        val_loss_q3,
        color="#e74c3c",
        alpha=0.2,
    )
    ax1.plot(
        epochs,
        val_loss_iqm,
        color="#e74c3c",
        linewidth=2.5,
        label="Val Loss",
    )

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend(loc="upper right", frameon=True)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(
        epochs,
        train_acc_q1,
        train_acc_q3,
        color="#3498db",
        alpha=0.2,
    )
    ax2.plot(
        epochs,
        train_acc_iqm,
        color="#3498db",
        linewidth=2.5,
        label="Train Acc",
    )

    ax2.fill_between(
        epochs,
        val_acc_q1,
        val_acc_q3,
        color="#e74c3c",
        alpha=0.2,
    )
    ax2.plot(
        epochs,
        val_acc_iqm,
        color="#e74c3c",
        linewidth=2.5,
        label="Val Acc",
    )

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Classification Accuracy (Test: {test_accuracy:.4f})")
    ax2.legend(loc="lower right", frameon=True)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.imshow(misclassified_images[0], cmap="gray")
    ax3.set_title(f"True: {true_labels[0]}, Pred: {predicted_labels[0]}")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.imshow(misclassified_images[1], cmap="gray")
    ax4.set_title(f"True: {true_labels[1]}, Pred: {predicted_labels[1]}")
    ax4.axis("off")

    plt.savefig(
        "paper/figures/mnist_classification.svg",
        bbox_inches="tight",
        dpi=300,
    )

    print(
        "Enhanced visualization with misclassified examples saved to paper/figures/mnist_classification.svg"
    )


if __name__ == "__main__":
    main()
