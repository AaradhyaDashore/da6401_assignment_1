import numpy as np

"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

class Optimizer():

    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate
    
class SGD(Optimizer):

    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        super().__init__(learning_rate)
        self.weight_decay = weight_decay
    
    def update(self, layers):
        for layer in layers:
            layer.W = layer.W - self.learning_rate * (layer.dW + self.weight_decay * layer.W)
            layer.b = layer.b - self.learning_rate * layer.db

class Momentum(Optimizer):

    def __init__(self, learning_rate=0.01, decay_rate=0.9, weight_decay=0.0):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay
        self.v_W = []
        self.v_b = []
        

    def update(self, layers):
        if not self.v_W:
            for layer in layers:
                self.v_W.append(np.zeros_like(layer.W))
                self.v_b.append(np.zeros_like(layer.b))
        for i in range(len(layers)):
            self.v_W[i] = self.decay_rate * self.v_W[i] + self.learning_rate * (layers[i].dW + self.weight_decay * layers[i].W)
            layers[i].W -= self.v_W[i]
            self.v_b[i] = self.decay_rate * self.v_b[i] + self.learning_rate * (layers[i].db)
            layers[i].b -= self.v_b[i]


class NAG(Optimizer):
    
    def __init__(self, learning_rate=0.01, decay_rate=0.9, weight_decay=0.0):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay
        self.v_W = []
        self.v_b = []
    
    def update(self, layers):
        if not self.v_W:
            for layer in layers:
                self.v_W.append(np.zeros_like(layer.W))
                self.v_b.append(np.zeros_like(layer.b))
        for i in range(len(layers)):
            self.v_W[i] = self.decay_rate * self.v_W[i] + self.learning_rate * (layers[i].dW + self.weight_decay * layers[i].W)
            layers[i].W -= ((self.decay_rate * self.v_W[i]) + self.learning_rate * (layers[i].dW + self.weight_decay * layers[i].W))
            self.v_b[i] = self.decay_rate * self.v_b[i] + self.learning_rate * (layers[i].db)
            layers[i].b -= ((self.decay_rate * self.v_b[i]) + (self.learning_rate * layers[i].db))

    def get_theta(self, layers):
        for i in range(len(layers)):
            layers[i].W += self.decay_rate * self.v_W[i]
            layers[i].b += self.decay_rate * self.v_b[i]
            

class RMSProp(Optimizer):
    
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_W = []
        self.v_b = []
        self.epsilon = 1e-8

    def update(self, layers):
        if not self.v_W:
            for layer in layers:
                self.v_W.append(np.zeros_like(layer.W))
                self.v_b.append(np.zeros_like(layer.b))
        for i in range(len(layers)):
            self.v_W[i] = self.beta * self.v_W[i] + (1 - self.beta) * (layers[i].dW**2)
            layers[i].W -= (self.learning_rate/((self.v_W[i])**0.5 + self.epsilon)) * (layers[i].dW + self.weight_decay * layers[i].W)
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * (layers[i].db**2)
            layers[i].b -= (self.learning_rate/((self.v_b[i])**0.5 + self.epsilon)) * (layers[i].db)