import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from loss import *
from activation_func import *
from module import *
from optimizers import *
from mltools import *

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
    np.random.seed(42)
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

def plot_classification(X_train, y_train, X_test, y_test, predict, iteration, losses):
    score_train = (y_train == predict(X_train)).mean()
    score_test = (y_test == predict(X_test)).mean()
    print(f"Train accuracy : {score_train}")
    print(f"Test accuracy : {score_test}")

    plot_frontiere(X_train, predict, step=100)
    plot_data(X_test, y_test.reshape(-1))
    plt.title("Train")
    plt.show()

    plot_frontiere(X_test, predict, step=100)
    plot_data(X_test, y_test.reshape(-1))
    plt.title("Test")
    plt.show()

    plt.plot(np.arange(iteration), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss over iteration")
    plt.show()
    print(np.array(losses).shape)

def train_binary_classification():
    np.random.seed(42)
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
        loss = loss_fn.forward(y_train, Y_pred).mean()
        losses.append(loss)
        loss_back = loss_fn.backward(y_train, Y_pred)
        layer.backward_update_gradient(X_train, loss_back)
        layer.update_parameters(learning_rate)
        layer.zero_grad()

    def predict(X):
        Y_pred = layer.forward(X)
        return np.where(Y_pred >= 0.5, 1, 0)
    
    plot_classification(X_train, y_train, X_test, y_test, predict, num_epochs, losses)

def mnist_classification():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    input_dim, hidden_dim, output_dim = 784, 128, 10
    epochs = 10
    lr = 0.01
    network = Sequential(
        Linear(input_dim, hidden_dim),
        TanH(),
        Linear(hidden_dim, output_dim),
        LogSoftmax()
    )

    loss_fn = CrossEntropyLoss()
    X_train = np.concatenate([batch[0].view(-1, 28 * 28).numpy() for batch in train_loader])
    Y_train = np.concatenate([np.eye(10)[batch[1].numpy()] for batch in train_loader])
    batch_size = 64
    optimizer = Optim(network, loss_fn, lr)
    optimizer.SGD(X_train, Y_train, batch_size, epochs)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.view(-1, 28 * 28)
            y_pred_test = network.forward(batch_x)
            predictions = torch.argmax(y_pred_test, axis=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # print("Partie 1")
    # linear_regression()

    # print("Partie 2")
    # train_binary_classification()

    # print("Partie 4")
    mnist_classification()
