import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils import create_autoencoder, get_batches, get_mnist

from mlp.loss import CrossEntropyLoss, MSELoss
from mlp.optim import Adam


def add_gaussian_noise(images, noise_factor):
    noisy_images = images.copy()
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=images.shape)
    noisy_images = noisy_images + noise
    return np.clip(noisy_images, 0.0, 1.0)


def train_denoising_autoencoder(
    x_train,
    x_val,
    latent_dim,
    depth,
    noise_factor,
    batch_size,
    learning_rate,
    max_epochs,
    patience,
    seed,
):
    np.random.seed(seed)

    input_dim = x_train.shape[1]

    model = create_autoencoder(input_dim, latent_dim, depth)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model, loss_fn, eps=learning_rate)

    best_val_loss = float("inf")
    best_model_state = None

    for _ in tqdm(range(max_epochs), leave=False):
        noisy_x_train = add_gaussian_noise(x_train, noise_factor)

        batches = get_batches(noisy_x_train, x_train, batch_size)

        for batch_x, batch_y in batches:
            optimizer.step(batch_x, batch_y)

        noisy_x_val = add_gaussian_noise(x_val, noise_factor)
        val_output = model.forward(noisy_x_val)
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


def evaluate_autoencoder(model, x_test, noise_factor):
    noisy_x_test = add_gaussian_noise(x_test, noise_factor)
    reconstructed = model.forward(noisy_x_test)
    return MSELoss().forward(x_test, reconstructed)


def main():
    seed = 42

    np.random.seed(seed)

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()

    model = train_denoising_autoencoder(
        x_train=x_train,
        x_val=x_val,
        latent_dim=64,
        depth=3,
        noise_factor=0.2,
        batch_size=128,
        learning_rate=0.001,
        max_epochs=500,
        patience=5,
        seed=seed,
    )

    results = {
        "loss": evaluate_autoencoder(model, x_train, 0.2),
        "model": model.state_dict(),
    }

    print(results)

    Path("results").mkdir(exist_ok=True)
    with open("results/autoencoder_denoiser.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
