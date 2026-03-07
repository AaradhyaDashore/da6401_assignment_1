import numpy as np
import argparse

from ann.neural_network import NeuralNetwork

best_config= argparse.Namespace(
            dataset="mnist",
            epochs=15,
            batch_size=32,
            loss="cross_entropy",
            optimizer="momentum",
            weight_decay=0.0,
            learning_rate=0.01,
            num_layers=2,
            hidden_size=[128, 64],
            activation="relu",
            weight_init="xavier"
        )

model = NeuralNetwork(best_config)

weights = np.load("src/best_model.npy", allow_pickle=True).item()

model.set_weights(weights)