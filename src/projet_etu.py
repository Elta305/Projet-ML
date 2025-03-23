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
    output_dim = Y.shape[1]
    learning_rate = 0.01
    num_epochs = 1000

    layer = Linear(input_dim, output_dim)
    loss_fn = MSELoss()

    losses = []
    for epoch in range(num_epochs):
        Y_pred = layer.forward(X)
        loss = loss_fn.forward(Y, Y_pred).mean()
        losses.append(loss)
        loss_back = loss_fn.backward(Y, Y_pred)    
        delta = layer.backward_delta(X, loss_back)
        layer.backward_update_gradient(X, delta)
        layer.update_parameters(learning_rate)
        layer.zero_grad()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    Y_pred = layer.forward(X)
    plot_loss(losses, num_epochs)
    plot_linear_regression_result(X, Y, Y_pred)

def plot_classification(X, Y, model, resolution=0.01):
    X1, X2 = X[:, 0], X[:, 1]
    x1_min, x1_max = X1.min() - 1, X1.max() + 1
    x2_min, x2_max = X2.min() - 1, X2.max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    output = grid
    for layer in model:
        output = layer.forward(output)
    Z = output.reshape(xx1.shape)

    plt.figure(figsize=(10, 5))
    plt.contourf(xx1, xx2, Z, alpha=0.4)
    plt.scatter(X1, X2, c=Y, s=20, edgecolor='k', label='Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Binary Classification')
    plt.legend()
    plt.show()

def train_binary_classification():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    Y = (X[:, 0] + X[:, 1] > 1).astype(float).reshape(-1, 1)
    num_epochs=1000
    learning_rate=1e-2
    
    input_dim, hidden_dim, output_dim = X.shape[1], 5, 1
    layer1 = Linear(input_dim, hidden_dim)
    activation1 = TanH()
    layer2 = Linear(hidden_dim, output_dim)
    activation2 = Sigmoid()
    loss_fn = MSELoss()
    losses = []

    for epoch in range(num_epochs):
        Z1 = layer1.forward(X)
        A1 = activation1.forward(Z1)
        Z2 = layer2.forward(A1)
        Y_pred = activation2.forward(Z2)
        loss = np.mean(loss_fn.forward(Y, Y_pred))
        losses.append(loss)

        layer1.zero_grad()
        layer2.zero_grad()
        delta = loss_fn.backward(Y, Y_pred)
        delta = activation2.backward_delta(Z2, delta)
        layer2.backward_update_gradient(A1, delta)
        delta = layer2.backward_delta(A1, delta)
        delta = activation1.backward_delta(Z1, delta)
        layer1.backward_update_gradient(X, delta)

        layer1.update_parameters(learning_rate)
        layer2.update_parameters(learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    plot_loss(losses, num_epochs)

    model = [layer1, activation1, layer2, activation2]
    plot_classification(X, Y, model)

def multi_class_classification():
    input_dim, hidden_dim, output_dim = 3, 5, 2
    X_test = np.random.rand(10, input_dim)
    Y_test = np.random.rand(10, output_dim)

    network = Sequential(
        Linear(input_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, output_dim),
        Softmax()
    )

    Y_pred = network.forward(X_test)
    print("Predicted output:\n", Y_pred)

    loss_fn = MSELoss()
    loss = loss_fn.forward(Y_test, Y_pred)
    print("Loss =", loss.mean())

    network.zero_grad()
    delta = loss_fn.backward(Y_test, Y_pred)
    network.backward(delta)

    learning_rate = 0.01
    network.update_parameters(learning_rate)

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
        Softmax()
    )

    loss_fn = CrossEntropyLoss()
    X_train = np.concatenate([batch[0].view(-1, 28 * 28).numpy() for batch in train_loader])
    Y_train = np.concatenate([np.eye(10)[batch[1].numpy()] for batch in train_loader])
    batch_size = 64
    optimizer = Optim(network, loss_fn, lr)
    SGD(network, loss_fn, X_train, Y_train, batch_size, epochs, lr, optimizer)

    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.view(-1, 28 * 28).numpy()
            onehot = np.zeros((batch_y.size, 10))
            onehot[np.arange(batch_y.size), batch_y.numpy()] = 1
            batch_y = onehot

            y_pred = network.forward(batch_x)
            loss = loss_fn.forward(batch_y, y_pred).mean()

            network.zero_grad()
            delta = loss_fn.backward(batch_y, y_pred)
            network.backward(delta)
            optimizer.step(batch_x, batch_y)

        print(f"Epoch {epoch+1}, Loss = {loss:.4f}")

    correct = 0
    total = 0
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.view(-1, 28 * 28).numpy()
        batch_y = batch_y.numpy()
        y_pred_test = network.forward(batch_x)
        predictions = np.argmax(y_pred_test, axis=1)
        correct += (predictions == batch_y).sum()
        total += batch_y.size

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    print("Partie 1")
    linear_regression()

    print("Partie 2")
    train_binary_classification()

    print("Partie 3")
    multi_class_classification()

    print("Partie 4")
    mnist_classification()
