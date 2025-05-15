import pickle

from utils import compute_stats, simple_whisker_plot


def main():
    with open("results/mnist_activation.pkl", "rb") as f:
        results = pickle.load(f)

    stats = {name: compute_stats(results[name]) for name in results}
    simple_whisker_plot(
        stats,
        ["#6ca247", "#d66b6a", "#5591e1", "#39a985", "#ad75ca", "#c77c1e"],
        "paper/figures/mnist_activation.svg",
    )


if __name__ == "__main__":
    main()
