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

            error = cost_derivation(y_pred, y_true) * derivation
        else:
            propagated_error = np.dot(error, nextLayerWeights[:, 1:])

            error = propagated_error * derivation

        gradient = np.outer(error, layer["input"]) * learningRate

        nextLayerWeights = layer["weights"]

        newLayerWeights = nextLayerWeights - gradient

        newWeights.insert(0, newLayerWeights)
    
    return newWeights