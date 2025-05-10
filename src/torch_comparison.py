import torch
from torch import nn, optim
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

input_dim, hidden_dim, output_dim = 784, 128, 10

epochs = 10
lr = 0.01

network = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=lr)

for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.view(-1, 28 * 28)

        optimizer.zero_grad()
        y_pred = network(batch_x)
        loss = loss_fn(y_pred, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.view(-1, 28 * 28)
        y_pred_test = network(batch_x)
        predictions = torch.argmax(y_pred_test, axis=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
