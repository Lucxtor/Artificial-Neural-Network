from activation import activation_funcs
from cost import cost_funcs

import numpy as np


def backpropagation(layers, y_pred, y_true, cost_function, learningRate):
    newWeights = []
    
    error = None
    nextLayerWeights = None

    for index, layer in enumerate(reversed(layers)):
        _, activation_derivation = activation_funcs[layer["activation_func"]]

        derivation = activation_derivation(layer["output"])

        if index == 0:
            _, cost_derivation = cost_funcs[cost_function]

            error = cost_derivation(y_pred, float(y_true)) * derivation
        else:
            propagated_error = np.dot(error, nextLayerWeights)

            error = propagated_error * derivation

        # print("erro ", error)

        gradient = np.outer(error, layer["input"]) * learningRate
        # print("Gradiente antes do clip:", gradient)
        # gradient = np.round(gradient, decimals=10)

        # max_grad = 1.

        # gradient = np.clip(gradient, -max_grad, max_grad)
        # print("Gradiente depois do clip:", gradient)




        nextLayerWeights = layer["weights"]

        newLayerWeights = nextLayerWeights - gradient

        newWeights.insert(0, newLayerWeights)
    
    return newWeights
