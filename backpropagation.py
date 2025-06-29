from activation import activation_funcs
from cost import cost_funcs

import numpy as np

def backpropagation(layers, cost_derivation, learningRate):
    newWeights = []
    
    error = None
    nextLayerWeights = None

    for layer in reversed(layers):
        _, activation_derivation = activation_funcs[layer["activation_func"]]

        derivation = activation_derivation(layer["output"])

        if error is None:
            error = cost_derivation * derivation
        else:
            propagated_error = np.dot(error, nextLayerWeights[:, 1:])

            error = propagated_error * derivation

        gradient = np.outer(error, layer["input"]) * learningRate

        nextLayerWeights = layer["weights"]

        newLayerWeights = nextLayerWeights - gradient

        newWeights.insert(0, newLayerWeights)
    
    return newWeights
