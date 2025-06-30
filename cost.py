import numpy as np

# Funções de custo e derivadas
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]

def categorical_cross_entropy_derivative(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return y_pred - y_true

cost_funcs = {
    'mse':            (mse, mse_derivative),
    'binary_cross_entropy':  (binary_cross_entropy, binary_cross_entropy_derivative),
    'categorical_cross_entropy': (categorical_cross_entropy, categorical_cross_entropy_derivative)
}
