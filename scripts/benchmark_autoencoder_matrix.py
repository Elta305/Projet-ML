import pickle
import threading
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils import create_autoencoder, get_batches, get_mnist

from mlp.loss import CrossEntropyLoss, MSELoss
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
    return MSELoss().forward(x_test, reconstructed)


def run_experiment(
    latent_dim,
    depth,
    loss_fn,
    shared_results,
    lock,
    run_seeds,
    x_train,
    x_val,
    x_test,
):
    loss_name = loss_fn.__class__.__name__
    key = (latent_dim, depth, loss_name)

    local_results = {
        "losses": [],
        "best_loss": float("inf"),
        "best_model": None,
    }

    for run in tqdm(
        range(len(run_seeds)),
        leave=False,
        desc=f"latent={latent_dim}, depth={depth}, loss={loss_name}",
    ):
        model = train_autoencoder_mnist(
            x_train=x_train,
            x_val=x_val,
            latent_dim=latent_dim,
            depth=depth,
            loss_fn=loss_fn,
            batch_size=128,
            learning_rate=0.001,
            max_epochs=200,
            patience=5,
            seed=run_seeds[run],
        )
        loss = evaluate_autoencoder(model, x_test)
        local_results["losses"].append(loss)

        if loss < local_results["best_loss"]:
            local_results["best_loss"] = loss
            local_results["best_model"] = model.state_dict()

    local_results["avg_loss"] = np.mean(local_results["losses"])

    with lock:
        shared_results[key] = local_results


def main():
    n_runs = 1
    random_seed = 42

    np.random.seed(random_seed)
    run_seeds = np.random.randint(0, 10000, size=n_runs)

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    results = {}

    threads = []
    results_lock = threading.Lock()

    loss_fns = [MSELoss(), CrossEntropyLoss()]

    for latent_dim in [4, 8, 16, 32, 64]:
        for depth in [1, 2, 3, 4, 5]:
            for loss_fn in loss_fns:
                thread = threading.Thread(
                    target=run_experiment,
                    args=(
                        latent_dim,
                        depth,
                        loss_fn,
                        results,
                        results_lock,
                        run_seeds,
                        x_train,
                        x_val,
                        x_test,
                    ),
                )
                threads.append(thread)
                thread.start()

    for thread in threads:
        thread.join()

    Path("results").mkdir(exist_ok=True)
    with open("results/mnist_hidden.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
