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
    # np.random.seed(42)
    # X_train, y_train = gen_arti(nbex=1000, data_type=1, epsilon=0.0)
    # X_test, y_test = gen_arti(nbex=1000, data_type=1, epsilon=0.0)
    # input_dim = X_train.shape[1]
    # output_dim = 1

    # y_train = np.where(y_train == -1, 0, 1).reshape((-1, 1))
    # y_test = np.where(y_test == -1, 0, 1).reshape((-1, 1))

    # num_epochs = 1000
    # learning_rate = 1e-4
    # batch_size = 100
    # num_batches = X_train.shape[0] // batch_size
    # loss_fn = MSELoss()
    # network = Sequential(
    #     Linear(input_dim, 128),
    #     TanH(),
    #     Linear(128, output_dim),
    #     Sigmoid()
    # )

    # losses = []
    # for epoch in range(num_epochs):
    #     epoch_loss = 0
    #     for batch in range(num_batches):
    #         start = batch * batch_size
    #         end = start + batch_size
    #         X_batch = X_train[start:end]
    #         y_batch = y_train[start:end]

    #         Y_pred = network.forward(X_batch)
    #         loss = loss_fn.forward(y_batch, Y_pred).mean()
    #         epoch_loss += loss
    #         grad_loss = loss_fn.backward(y_batch, Y_pred)
    #         network.backward(grad_loss)
    #         network.update_parameters(learning_rate)
    #         network.zero_grad()
        
    #     losses.append(epoch_loss / num_batches)

    # def predict(X):
    #     Y_pred = network.forward(X)
    #     return np.where(Y_pred >= 0.5, 1, 0)
    
    # plot_classification(X_train, y_train, X_test, y_test, predict, num_epochs, losses)

    np.random.seed(42)
    X_train, y_train = gen_arti(nbex=1000, data_type=1, epsilon=0.0)
    X_test, y_test = gen_arti(nbex=1000, data_type=1, epsilon=0.0)
    input_dim = X_train.shape[1]
    output_dim = 1

    y_train = np.where(y_train == -1, 0, 1).reshape((-1, 1))
    y_test = np.where(y_test == -1, 0, 1).reshape((-1, 1))

    num_epochs = 1000
    learning_rate = 1e-4
    loss_fn = MSELoss()
    layer1 = Linear(input_dim, 128)
    activation1 = TanH()
    layer2 = Linear(128, output_dim)
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

if __name__ == "__main__":
    # print("Partie 1")
    # linear_regression()

    # print("Partie 2")
    # train_binary_classification_linear()
    train_binary_classification()

    # print("Partie 4")
    # mnist_classification()
