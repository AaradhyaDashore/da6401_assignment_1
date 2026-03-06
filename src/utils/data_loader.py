import numpy as np
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_and_preprocess_data(val_size=0.2, test_size=0.2, dataset = 'mnist'):
    """
    Loads MNIST, normalizes, and splits into Train, Val, and Test sets.
    """
    # 1. Load data from Keras
    if dataset == 'mnist':
        (X_train_full, y_train_full), (X_test_raw, y_test_raw) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train_full, y_train_full), (X_test_raw, y_test_raw) = fashion_mnist.load_data()
        
    # 2. Reshape to 1D vectors (784) and Normalize to [0, 1]
    X_train_full = X_train_full.reshape(-1, 784).astype('float32') / 255.0
    X_test = X_test_raw.reshape(-1, 784).astype('float32') / 255.0

    # 3. One-Hot Encoding using Sklearn
    encoder = OneHotEncoder(sparse_output=False)
    y_train_full_oh = encoder.fit_transform(y_train_full.reshape(-1, 1))
    y_test_oh = encoder.transform(y_test_raw.reshape(-1, 1))

    # 4. Train-Validation Split using Sklearn
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, 
        y_train_full_oh, 
        test_size=val_size, 
        random_state=42,
        stratify=y_train_full 
    )

    return X_train, y_train, X_val, y_val, X_test, y_test_oh