import numpy as np

# Funções de ativação e derivadas
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(a):
    return (a > 0).astype(float)

def softmax(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_derivative(z):
    return 1

activation_funcs = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu':    (relu, relu_derivative),
    'softmax': (softmax, softmax_derivative)
}
