import numpy as np

# Funções de ativação e derivadas
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    sigmoid_a = sigmoid(a)
    return sigmoid_a * (1 - sigmoid_a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(a):
    return np.where(a > 0, 1, 0)

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    exps = np.exp(z)
    return exps / np.sum(exps, axis=-1, keepdims=True)

def softmax_derivative(z):
    return 1

def identity(z):
    return z

def identity_derivative(z):
    return np.ones_like(z)

activation_funcs = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu':    (relu, relu_derivative),
    'softmax': (softmax, softmax_derivative),
    'identity': (identity, identity_derivative)
}
