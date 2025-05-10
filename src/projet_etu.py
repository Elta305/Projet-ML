import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from torchvision import datasets, transforms

from activation_func import *
from loss import *
from mltools import *
from module import *
from optimizers import *


def plot_loss(losses, num_epochs, title='Training Loss over Epochs'):
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_linear_regression_result(X, Y, Y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, label='Data')
    plt.plot(X, Y_pred, color='red', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression Result')
    plt.legend()
    plt.show()

def linear_regression():
    X = 2 * np.random.rand(100, 1)
    Y = 4 + 3 * X + np.random.randn(100, 1)

    input_dim = X.shape[1]
    output_dim = 1
    learning_rate = 1e-3
    num_epochs = 100

    loss_fn = MSELoss()
    layer = Linear(input_dim, output_dim)

    losses = []
    for epoch in range(num_epochs):
        Y_pred = layer.forward(X)
        loss = loss_fn.forward(Y, Y_pred).mean()
        losses.append(loss)
        loss_back = loss_fn.backward(Y, Y_pred)
        layer.backward_update_gradient(X, loss_back)
        layer.update_parameters(learning_rate)
        layer.zero_grad()

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}: Loss = {loss:.4f}")

    Y_pred = layer.forward(X)
    plot_loss(losses, num_epochs)
    plot_linear_regression_result(X, Y, Y_pred)

def train_binary_classification_linear():
    X_train, y_train = gen_arti(nbex=1000, data_type=0, epsilon=0.5)
    X_test, y_test = gen_arti(nbex=1000, data_type=0, epsilon=0.5)
    input_dim = X_train.shape[1]
    output_dim = 1

    y_train = np.where(y_train == -1, 0, 1).reshape((-1, 1))
    y_test = np.where(y_test == -1, 0, 1).reshape((-1, 1))

    num_epochs = 100
    learning_rate = 1e-4
    loss_fn = MSELoss()
    layer = Linear(input_dim, output_dim)

    losses = []
    for epoch in range(num_epochs):
        Y_pred = layer.forward(X_train)
        loss_back = loss_fn.backward(y_train, Y_pred)
        loss = loss_back.mean()
        losses.append(loss)
        # delta = layer.backward_delta(X_train, loss_back)
        layer.backward_update_gradient(X_train, loss_back)
        layer.update_parameters(learning_rate)
        layer.zero_grad()

    def predict(X):
        Y_pred = layer.forward(X)
        return np.where(Y_pred >= 0.5, 1, 0)

    plot_classification(X_train, y_train, X_test, y_test, predict, num_epochs, losses)

def train_binary_classification():
    X_train, y_train = gen_arti(nbex=1000, data_type=1, epsilon=0.0)
    X_test, y_test = gen_arti(nbex=1000, data_type=1, epsilon=0.0)
    input_dim = X_train.shape[1]
    output_dim = 1

    y_train = np.where(y_train == -1, 0, 1).reshape((-1, 1))
    y_test = np.where(y_test == -1, 0, 1).reshape((-1, 1))

    num_epochs = 100
    learning_rate = 1e-4
    loss_fn = MSELoss()
    layer1 = Linear(input_dim, 64)
    activation1 = TanH()
    layer2 = Linear(64, output_dim)
    activation2 = Sigmoid()

    losses = []
    for epoch in range(num_epochs):
        hidden1 = layer1.forward(X_train)
        activated1 = activation1.forward(hidden1)
        hidden2 = layer2.forward(activated1)
        Y_pred = activation2.forward(hidden2)
        loss = loss_fn.forward(y_train, Y_pred).mean()
        losses.append(loss)

        grad_loss = loss_fn.backward(y_train, Y_pred)
        grad_hidden2 = activation2.backward_delta(hidden2, grad_loss)
        grad_activated1 = layer2.backward_delta(activated1, grad_hidden2)
        grad_hidden1 = activation1.backward_delta(hidden1, grad_activated1)

        layer2.backward_update_gradient(activated1, grad_hidden2)
        layer1.backward_update_gradient(X_train, grad_hidden1)

        layer2.update_parameters(learning_rate)
        layer1.update_parameters(learning_rate)

        layer2.zero_grad()
        layer1.zero_grad()

    def predict(X):
        hidden1 = layer1.forward(X)
        activated1 = activation1.forward(hidden1)
        hidden2 = layer2.forward(activated1)
        Y_pred = activation2.forward(hidden2)
        return np.where(Y_pred >= 0.5, 1, 0)

    plot_classification(X_train, y_train, X_test, y_test, predict, num_epochs, losses)

def train_binary_classification_seq():
    X_train, y_train = gen_arti(nbex=1000, data_type=1, epsilon=0.0)
    X_test, y_test = gen_arti(nbex=1000, data_type=1, epsilon=0.0)
    input_dim = X_train.shape[1]
    output_dim = 1

    y_train = np.where(y_train == -1, 0, 1).reshape((-1, 1))
    y_test = np.where(y_test == -1, 0, 1).reshape((-1, 1))

    num_epochs = 100
    learning_rate = 1e-4
    loss_fn = MSELoss()
    network = Sequential(
        Linear(input_dim, 64),
        TanH(),
        Linear(64, output_dim),
        Sigmoid()
    )

    optimizer = Optim(network, loss_fn, learning_rate)
    losses = optimizer.SGD(X_train, y_train, batch_size=64, num_iterations=num_epochs)

    def predict(X):
        Y_pred = network.forward(X)
        return np.where(Y_pred >= 0.5, 1, 0)

    plot_classification(X_train, y_train, X_test, y_test, predict, num_epochs, losses, with_batch=True)
    print("Final loss:", losses[-1])

def mnist_classification():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    input_dim, hidden_dim, output_dim = 784, 128, 10
    epochs = 10
    lr = 1e-4
    network = Sequential(
        Linear(input_dim, hidden_dim),
        TanH(),
        Linear(hidden_dim, output_dim),
        LogSoftmax()
    )

    loss_fn = CrossEntropyLoss()
    # optimizer = Optim(network, loss_fn, lr)
    # data_x = np.vstack([images.view(-1, 28 * 28).numpy() for images, _ in train_loader])
    # data_y = np.vstack([np.eye(output_dim)[labels.numpy()] for _, labels in train_loader])
    # losses = optimizer.SGD(data_x, data_y, batch_size=64, num_iterations=epochs)
    losses = []
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28).numpy()
            labels = np.eye(output_dim)[labels.numpy()]

            outputs = network.forward(images)
            loss = loss_fn.forward(labels, outputs).mean()
            losses.append(loss)

            grad_loss = loss_fn.backward(labels, outputs)
            network.backward(grad_loss)
            network.update_parameters(lr)
            network.zero_grad()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    correct, total = 0, 0
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28).numpy()
        labels = labels.numpy()

        outputs = network.forward(images)
        predictions = np.argmax(outputs, axis=1)
        correct += (predictions == labels).sum()
        total += labels.size

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def train_autoencoder():
    # Generate synthetic data
    X_train, _ = gen_arti(nbex=1000, data_type=1, epsilon=0.0)
    X_test, _ = gen_arti(nbex=1000, data_type=1, epsilon=0.0)

    input_dim = X_train.shape[1]
    latent_dim = 2  # Dimension of the latent space

    # Define the autoencoder architecture
    encoder = Sequential(
        Linear(input_dim, 100),
        TanH(),
        Linear(100, latent_dim),
        TanH()
    )
    decoder = Sequential(
        Linear(latent_dim, 100),
        TanH(),
        Linear(100, input_dim),
        Sigmoid()
    )

    # Training parameters
    num_epochs = 100
    learning_rate = 1e-3
    loss_fn = BCELoss()

    losses = []
    for epoch in range(num_epochs):
        # Forward pass
        latent_representations = encoder.forward(X_train)
        X_reconstructed = decoder.forward(latent_representations)
        loss = loss_fn.forward(X_train, X_reconstructed).mean()
        losses.append(loss)

        # Backward pass
        grad_loss = loss_fn.backward(X_train, X_reconstructed)
        grad_latent = decoder.backward(grad_loss)
        encoder.backward(grad_latent)

        decoder.update_parameters(learning_rate)
        encoder.update_parameters(learning_rate)

        decoder.zero_grad()
        encoder.zero_grad()

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # Visualize reconstruction
    X_reconstructed = decoder.forward(encoder.forward(X_test))
    plt.figure(figsize=(10, 5))
    plt.scatter(X_test[:, 0], X_test[:, 1], label="Original Data", alpha=0.5)
    plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], label="Reconstructed Data", alpha=0.5)
    plt.legend()
    plt.title("Original vs Reconstructed Data")
    plt.show()

    # Visualize latent space
    latent_representations = encoder.forward(X_test)
    plt.figure(figsize=(10, 5))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], alpha=0.5)
    plt.title("Latent Space Representations")
    plt.show()

    # Clustering in latent space
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(latent_representations)
    plt.figure(figsize=(10, 5))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title("Clustering in Latent Space")
    plt.show()

if __name__ == "__main__":
    # print("Partie 1")
    # linear_regression()

    # print("Partie 2")
    # train_binary_classification_linear()
    # train_binary_classification()
    # train_binary_classification_seq()

    # print("Partie 4")
    mnist_classification()

    # print("Partie 5")
    # train_autoencoder()
