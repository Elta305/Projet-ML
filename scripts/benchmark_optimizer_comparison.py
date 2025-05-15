import pickle
from pathlib import Path

import numpy as np
from utils import evaluate_model_mnist, get_mnist, train_mnist_classifier

from mlp.activation import ReLU
from mlp.optim import SGD, Adam, SGDMomentum


def main():
    n_runs = 15
    random_seed = 42

    np.random.seed(random_seed)
    run_seeds = np.random.randint(0, 10000, size=n_runs)

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    configs = {
        "SGD": {"class": SGD, "lr": 0.01, "results": []},
        "SGDMomentum": {
            "class": SGDMomentum,
            "lr": 0.01,
            "results": [],
        },
        "Adam": {"class": Adam, "lr": 0.001, "results": []},
    }

    for name, config in configs.items():
        accuracies = []

        print(name)

        for run in range(n_runs):
            print(f"Run {run + 1} / {n_runs}")
            model = train_mnist_classifier(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                optim=config["class"],
                activation=ReLU,
                batch_size=128,
                hidden_dims=[64, 32],
                learning_rate=config["lr"],
                max_epochs=50,
                seed=run_seeds[run],
                patience=5,
            )

            accuracy = evaluate_model_mnist(model, x_test, y_test)
            print(accuracy)
            accuracies.append(accuracy)

        config["results"] = accuracies

    results = {name: configs[name]["results"] for name in configs}

    print(results)

    Path("results").mkdir(exist_ok=True)
    with open("results/mnist_optimizer.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
