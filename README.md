# DA6401 - Assignment 1
**Implementation of a Modular Multi-Layer Perceptron (MLP) from Scratch**

---

* **Name**: Aaradhya Dashore
* **Roll Number**: ME22B089s
* **WandB Report**: https://wandb.ai/aaradhyadashore784-iit-madras/da6401_assignment_1_final/reports/DA6401-Assignment-1-Report--VmlldzoxNjEzMzc0NA?accessToken=g0kq4v1cypsrqknk12kal2srogzfnzp7jqydh9ghj4inc68tbqioxhdjbrizqssd
* **GitHub Repository**: https://github.com/AaradhyaDashore/da6401_assignment_1/

---

## Project Overview
This repository contains a ground-up implementation of a Multi-Layer Perceptron (MLP) using only **NumPy**. The project was developed as part of the DA6401 course to deeply understand the mechanics of backpropagation, gradient flow, and optimization in neural networks.

### Key Deliverables:
1.  **Modular Neural Engine**: Supports arbitrary depth, width, and activation functions.
2.  **Comprehensive Optimizer Suite**: Implementation of SGD, Momentum, NAG, RMSProp, Adam, and Nadam.
3.  **Experimental Analysis**: 150+ runs logged on Weights & Biases (W&B) analyzing vanishing gradients, dead neurons, and initialization strategies.
4.  **Cross-Dataset Validation**: Performance testing on both MNIST (digits) and Fashion-MNIST (clothing).

---

## Features & Architecture
The system is designed with a modular architecture where each layer and optimizer is an independent component:

* **Activations**: ReLU, Sigmoid, and Softmax (Output).
* **Loss Functions**: Cross-Entropy (Primary) and Mean Squared Error.
* **Initialization**: Random and Xavier (Glorot) to ensure stable signal variance.
* **Optimizers**: 
    * Standard Gradient Descent (SGD)
    * Momentum & Nesterov Accelerated Gradient (NAG)
    * Adaptive Methods (RMSProp)

---

## Experimental Highlights
Summarizing the key findings from our investigation:
* **Peak Accuracy**: Achieved **97.9%** on MNIST using a [128, 64] architecture with Momentum optimization.
* **Vanishing Gradients**: Verified that Sigmoid activations stall in deep networks at low learning rates ($10^{-5}$), while ReLU maintains healthy gradient flow.
* **Initialization**: Demonstrated that **Zero Initialization** leads to a total failure of symmetry breaking, whereas **Xavier Initialization** ensures immediate convergence.
* **Fashion-MNIST**: Achieved **~89.5%** accuracy, identifying the increased complexity of texture-based classification compared to digit classification.
