import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils import evaluate_model_mnist, get_mnist, train_mnist_classifier

from mlp.activation import ReLU
from mlp.optim import Adam


def main():
    n_runs = 15
    random_seed = 42

    np.random.seed(random_seed)
    run_seeds = np.random.randint(0, 10000, size=n_runs)

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    configs = {
        "16": {"class": [16, 16], "results": []},
        "32": {"class": [32, 32], "results": []},
        "64": {"class": [64, 64], "results": []},
        "128": {"class": [128, 128], "results": []},
    }

    for name, config in configs.items():
        accuracies = []

        for run in tqdm(range(n_runs), desc=f"{name}"):
            model = train_mnist_classifier(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                optim=Adam,
                activation=ReLU,
                batch_size=128,
                hidden_dims=config["class"],
                learning_rate=0.001,
                max_epochs=50,
                seed=run_seeds[run],
                patience=5,
            )

            accuracy = evaluate_model_mnist(model, x_test, y_test)
            accuracies.append(accuracy)

        config["results"] = accuracies

    results = {name: configs[name]["results"] for name in configs}

    Path("results").mkdir(exist_ok=True)
    with open("results/mnist_hidden.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
