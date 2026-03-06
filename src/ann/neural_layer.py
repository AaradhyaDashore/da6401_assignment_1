import numpy as np
from ann.activations import *

"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

class NeuralLayer():

    def __init__(self, n_in:int , n_out:int , init_type:str, activation_type:str):

        if (activation_type == "sigmoid"):
            self.activation = Sigmoid()
        elif (activation_type == "tanh"):
            self.activation = TanH()
        elif (activation_type == "relu"):
            self.activation = ReLU()
        elif (activation_type == "softmax"):
            self.activation = Softmax()
        
        self.n_in = n_in
        self.n_out = n_out

        if (init_type == "random"):
            self.W = np.random.normal(0, 1, size = (n_in,n_out))
            self.b = np.zeros((n_out, 1))
        elif(init_type == "xavier"):
            self.W = np.random.normal(0, (2/(n_in + n_out))**0.5, size = (n_in, n_out))
            self.b = np.zeros((n_out, 1))
        
    def forward(self, x):
        # shape of x is (n_in, b)
        # shape of W is (n_in, n_out)
        # shape of b is (n_out, 1)
        # shape of self.logits and self.output is (n_out, b)
        self.batch_size = x.shape[1]
        self.input = x
        self.logits = np.dot((self.W).T, x) + self.b
        self.output = self.activation.forward(self.logits)
        return self.output

    def backward(self, grad_output):
        # shape of grad_ouput is (n_out, b)
        # shape of delta is (n_out, b)
        # shape of dW is (n_in, n_out), and is the averaged gradient over all data of the batch for all weights in the layer
        # shape of db is (n_out, 1), and is the averaged gradient over all data of the batch for all biases in the layer
        # shape of grad_input is (n_in, b), which will be passed to the previous layer in the greater structure of neural network
        self.delta = self.activation.backward(grad_output)
        self.dW = np.einsum('ijk,ikl -> jl', self.input.T[:,:,np.newaxis], self.delta.T[:,np.newaxis,:])/self.batch_size
        # self.dW = np.dot(self.input, self.delta.T)/self.batch_size
        self.db = np.sum(self.delta, axis=1, keepdims=True)/self.batch_size
        self.grad_input = np.dot(self.W, self.delta)
        return self.grad_input
    