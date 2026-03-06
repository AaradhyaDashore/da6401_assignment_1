import numpy as np

"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

class Loss():
    # shape of y and y_hat: (n_out, b)
    def __init__(self):
        self.loss = None

class MeanSquaredError(Loss):

    def forward(self, y, y_hat):
        # Computes the Loss function given y and y_hat
        self.loss = np.mean((y - y_hat)**2, axis=0)
        return self.loss
    
    def backward(self, y, y_hat):
        # Computes the gradient of Loss function wrt y_hat given y and y_hat
        return (2/len(y)) * (y_hat - y)
    
class CategorialCrossEntropy(Loss):

    def forward(self, y, y_hat):
        # Computes the Loss function give y and y_hat
        self.loss = -np.sum((y * np.log(np.clip(y_hat, 1e-15, 1 - 1e-15))), axis=0)
        return self.loss
    
    def backward(self, y, y_hat):
        # Computes the gradient of Loss function wrt Softmax activation function here, given y and y_hat
        return (y_hat - y)