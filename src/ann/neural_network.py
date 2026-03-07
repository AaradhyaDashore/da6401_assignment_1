import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import *
from ann.optimizers import *

"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers = []
        self.epochs = cli_args.epochs
        self.batch_size = cli_args.batch_size
        self.learning_rate = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay
        self.num_layers = cli_args.num_layers
        self.hidden_size = cli_args.hidden_size
        self.activation = cli_args.activation
        self.weight_init = cli_args.weight_init
        # self.wandb_project = cli_args.wandb_project

        if cli_args.loss == "mean_squared_error":
            self.LossFunction = MeanSquaredError()
        elif cli_args.loss == "cross_entropy":
            self.LossFunction = CategorialCrossEntropy()

        if cli_args.optimizer == 'sgd':
            self.optimizer = SGD(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'momentum':
            self.optimizer = Momentum(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'nag':
            self.optimizer = NAG(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'rmsprop':
            self.optimizer = RMSProp(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)

        dims = [784] + cli_args.hidden_size + [10]
        acts = ([self.activation] * self.num_layers) + ["softmax" if cli_args.loss == "cross_entropy" else "relu"]
        for i in range(len(dims)-1):
            layer = NeuralLayer(dims[i], dims[i+1], init_type = self.weight_init, activation_type = acts[i])
            self.layers.append(layer)
    
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data (shape: (b, n_in))
            
        Returns:
            Output logits (shape: (b, n_out))
        """
        activation = X.T
        for i in range(len(self.layers)):
            activation = self.layers[i].forward(activation)
        self.act_out = activation.T
        return (self.layers[-1].logits).T
            
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels (shape: (b, n_out))
            y_pred: Predicted outputs (shape: (b, n_out))
            
        Returns:
            return grad_w, grad_b
        """
        grad_W_list = []
        grad_b_list = []
        
        self.loss = self.LossFunction.forward(y_true.T, y_pred.T)
        grad_output = self.LossFunction.backward(y_true.T, y_pred.T)
        
        for i in range(len(self.layers) - 1, -1, -1):
            grad_output = self.layers[i].backward(grad_output)
            grad_W_list.append(self.layers[i].dW)
            grad_b_list.append(self.layers[i].db)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b
    

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update(self.layers)
        if (self.optimizer.__class__.__name__ == 'NAG'):
            self.optimizer.get_theta(self.layers)

    
    def train(self, X_train, y_train, epochs=None, batch_size=None):
        """
        Train the network for specified epochs.
        """
        import wandb
        
        # Use CLI defaults if not provided
        epochs = epochs if epochs is not None else self.epochs
        batch_size = batch_size if batch_size is not None else self.batch_size
        n_samples = X_train.shape[0]

        print(f"\nStarting Training: {epochs} epochs, Batch Size: {batch_size}")

        for epoch in range(epochs):

            # 1. Shuffle data at the start of every epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_train_loss = 0

            # 2. Iterate through data in mini-batches
            for i in range(0, n_samples, batch_size):
                current_batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:current_batch_end]
                y_batch = y_shuffled[i:current_batch_end]

                # 3. Batch Forward Pass:
                self.forward(X_batch) 
                y_pred_batch = self.act_out

                # 4. Backward Pass Loop:
                self.backward(y_batch, y_pred_batch)

                # 5. Losses gets accumulated
                batch_loss = self.loss.sum()
                epoch_train_loss += batch_loss

                # 6. Calls optimizer.update
                self.update_weights()
                
            # 7. Logging per epoch (Evaluation happens in train.py)
            avg_train_loss = epoch_train_loss / n_samples
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}\n")
    

    def evaluate(self, X, y):
        """
        Evaluate the network on given data using efficient batch processing.
        Args:
            X: Input features (n_samples, 784)
            y: True one-hot labels (n_samples, 10)
        Returns:
            avg_loss, accuracy
        """
        n_samples = X.shape[0]
        total_loss = 0
        correct = 0

        # 1. Forward Pass to evaluate the predicted output
        self.forward(X) 
        y_hat = self.act_out

        # 2. Evaluating Total Loss
        total_loss = self.LossFunction.forward(y.T, y_hat.T).sum()
            
        # Check accuracy
        correct = (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)).sum()
                
        # Return averaged metrics for logging to W&B
        return total_loss/n_samples, correct/n_samples
    

    def get_weights(self):
            d = {}
            for i, layer in enumerate(self.layers):
                d[f"W{i}"] = layer.W.copy()
                d[f"b{i}"] = layer.b.copy()
            return d


    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

