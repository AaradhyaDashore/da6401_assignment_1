import numpy as np

"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

class Activation:
    # shape of z is (n_out, b)
    # shape of grad_output is (n_out, b)
    def __init__(self):
        self.state = None


class Sigmoid(Activation):
    def forward(self, z):
        # Computes the sigmoid function of a vector in forward pass
        self.state = 1/(1+np.exp(-z)) 
        return self.state
    
    def backward(self, grad_output):
        # Computes the gradient (with grad_output coming from next layer) to the previous layer in backward pass
        return grad_output * (self.state * (1 - self.state))


class TanH(Activation):

    def forward(self, z):
        # Computes the tanh function of a vector in forward pass
        self.state = np.tanh(z) 
        return self.state
    
    def backward(self, grad_output):
        # Computes the gradient (with grad_output coming from next layer) to the previous layer in backward pass
        return grad_output * (1 - (self.state ** 2))


class ReLU(Activation):

    def forward(self, z):
        # Computes the ReLU function of a vector in forward pass
        self.state = np.maximum(z,0) 
        return self.state
    
    def backward(self, grad_output):
        # Computes the gradient (with grad_output coming from next layer) to the previous layer in backward pass
        return grad_output * ((self.state>0).astype('int32'))

        
class Softmax(Activation):

    def forward(self, z):
        # Computes the softmax function of a vector in forward pass
        self.state =  np.exp(z - np.max(z, axis=0, keepdims=True))/np.sum(np.exp(z - np.max(z, axis=0, keepdims=True)), axis=0, keepdims=True)
        return self.state
    
    def backward(self, grad_output):
        # Computes the gradient (with grad_output coming from next layer) to the previous layer in backward pass
        # The computation for the gradient of softmax (essentially the Jocobian matrix) is not done here as it will get paired up with the Categorical Cross Entropy Loss function
        return grad_output
        