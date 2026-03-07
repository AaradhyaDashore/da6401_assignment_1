"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import wandb
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_and_preprocess_data

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a modular MLP on MNIST/Fashion-MNIST')
    
    # Dataset & Hyperparameters
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    
    # Model Architecture
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier', choices=['random', 'xavier'])
    
    # Optimization
    parser.add_argument('-o', '--optimizer', type=str, default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'])
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    
    # W&B Configuration
    parser.add_argument('-w_p', '--wandb_project', type=str, default='da6401_assignment_1_final')

    return parser.parse_args()

def log_sample_images(X_train, y_train, class_names):
    """
    Logs one sample image for each of the 10 classes to W&B.
    """
    images = []
    labels = []
    
    # Iterate through all 10 classes
    for i in range(10):
        # Find the index of the first occurrence of this class
        # y_train is assumed to be one-hot, so we check argmax
        idx = np.where(np.argmax(y_train, axis=1) == i)[0][0]
        
        # Reshape the flat 784 vector back to 28x28 for visualization
        img = X_train[idx].reshape(28, 28)
        
        # Add to our list as a wandb.Image object
        images.append(wandb.Image(img, caption=class_names[i]))

    # Log the collection to W&B
    wandb.log({"Sample Data Exploration": images})

def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # 1. Initialize W&B for experiment tracking
    wandb.init(project=args.wandb_project, config=vars(args))
    config = wandb.config

    # 2. Load and split data (Assumes 80/20 or 90/10 split internally)
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(dataset=config.dataset)

    # Logging the sample Images to wandb for Q2.1
    log_sample_images(X_train, y_train, np.arange(10))

    # 3. Initialize the Neural Network engine
    model = NeuralNetwork(config)

    print(f"Model initialized with {config.optimizer} optimizer and {config.activation} activation.")

    # 4. Training Loop
    # We call the model's internal train method which handles batching and updates
    # To satisfy W&B report requirements, we evaluate on validation set every epoch
    for epoch in range(config.epochs):
        # Perform one epoch of training
        model.train(X_train, y_train, epochs=1, batch_size=config.batch_size)
        
        # Evaluate performance
        train_loss, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc = model.evaluate(X_val, y_val)

        # 5. Specialized Logging for Report Questions
        # Q2.4: Log gradient norms of the first hidden layer for Vanishing Gradient Analysis 
        first_layer_grad_norm = np.linalg.norm(model.layers[0].dW)
        
        # Q2.5: Log activation distribution for Dead Neuron Investigation
        # We use the state saved during the val evaluation
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "grad_norm_layer_0": first_layer_grad_norm
        })

        print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

    # 6. Final Evaluation on Test Set for Model Selection
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # --- THIS FOR 2.8 ERROR ANALYSIS ---
    # Get the raw probability outputs for the entire test set
    # Using the forward pass logic of your custom model
    model.forward(X_test) 
    test_probs = model.act_out
    
    # Convert one-hot ground truth back to class integers
    y_true_indices = np.argmax(y_test, axis=1)
    
    # Convert model probabilities to class predictions
    y_pred_indices = np.argmax(test_probs, axis=1)

    # Log the Confusion Matrix to W&B
    wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_indices, 
                preds=y_pred_indices,
                class_names=[str(i) for i in range(10)])})
    
    # 7. Save the Best Model for submission 
    best_weights = model.get_weights()
    np.save("src/best_model.npy", best_weights)
    
    # Save config as JSON for the autograder
    import json
    with open("best_config.json", "w") as f:
        json.dump(vars(args), f)

    wandb.finish()


if __name__ == '__main__':
    main()
