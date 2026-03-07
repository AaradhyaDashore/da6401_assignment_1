"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_and_preprocess_data

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Inference script for MLP Model')
    
    # Model Configuration (Set these to your best found hyperparameters)
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-nhl', '--num_layers', type=int, default=2)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 64])
    parser.add_argument('-a', '--activation', type=str, default='relu')
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy')
    
    # Paths for saved model
    parser.add_argument('--model_path', type=str, default='src/best_model.npy', help='Path to .npy weights')
    
    # Dummy args to maintain CLI consistency with train.py 
    parser.add_argument('-e', '--epochs', type=int, default=15)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-o', '--optimizer', type=str, default='momentum')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-w_p', '--wandb_project', type=str, default='da6401_assignment_1_final')

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    try:
        # allow_pickle is required for loading the dictionary object
        data = np.load(model_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    # 1. Batch Forward Pass to get logits
    model.forward(X_test)
    y_pred_logits = model.act_out
    
    # 2. Convert to class labels
    y_pred_labels = np.argmax(y_pred_logits, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # 3. Calculate Metrics (using macro average for multi-class) [cite: 55]
    metrics = {
        "accuracy": accuracy_score(y_true_labels, y_pred_labels),
        "precision": precision_score(y_true_labels, y_pred_labels, average='macro'),
        "recall": recall_score(y_true_labels, y_pred_labels, average='macro'),
        "f1_score": f1_score(y_true_labels, y_pred_labels, average='macro')
    }
    
    return metrics, y_pred_logits


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    
    # 1. Load Data
    _, _, _, _, X_test, y_test = load_and_preprocess_data(dataset=args.dataset)

    # 2. Reconstruct Model [cite: 74]
    model = NeuralNetwork(args)
    
    # 3. Inject Saved Weights 
    weights = load_model(args.model_path)

    if weights:
        model.set_weights(weights)
        print(f"Model loaded successfully from {args.model_path}")
    else:
        return

    # 4. Run Evaluation
    results, logits = evaluate_model(model, X_test, y_test)
    
    # 5. Output for Autograder/Report [cite: 55]
    print("\n--- Final Test Metrics ---")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return results


if __name__ == '__main__':
    main()
