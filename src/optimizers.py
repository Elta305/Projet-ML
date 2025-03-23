import numpy as np

class Optim:
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps
    
    def step(self, batch_x, batch_y):
        output = self.net.forward(batch_x)
        loss = np.mean(self.loss.forward(batch_y, output))
        self.net.zero_grad()
        delta = self.loss.backward(batch_y, output)
        self.net.backward(delta)
        self.net.update_parameters(self.eps)
        return loss

    def SGD(self, data_x, data_y, batch_size, num_iterations):
        for _ in range(num_iterations):
            indices = np.random.permutation(len(data_x))
            for i in range(0, len(data_x), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_x = data_x[batch_indices]
                batch_y = data_y[batch_indices]
                self.step(batch_x, batch_y)

class AdamOptimizer(Optim):
    def __init__(self, net, loss, eps, beta1=0.9, beta2=0.999):
        super().__init__(net, loss, eps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [np.zeros_like(p) for p in self.net.parameters()]
        self.v = [np.zeros_like(p) for p in self.net.parameters()]
        self.t = 0
    
    def step(self, batch_x, batch_y):
        self.t += 1
        output = self.net.forward(batch_x)
        loss = np.mean(self.loss.forward(batch_y, output))
        self.net.zero_grad()
        delta = self.loss.backward(batch_y, output)
        self.net.backward(delta)
        self.m = [self.beta1 * m + (1 - self.beta1) * p for m, p in zip(self.m, self.net._gradient)]
        self.v = [self.beta2 * v + (1 - self.beta2) * p ** 2 for v, p in zip(self.v, self.net._gradient)]
        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]
        self.net._parameters = [p - self.eps * m / (np.sqrt(v) + 1e-8) for p, m, v in zip(self.net._parameters, m_hat, v_hat)]
        return loss
