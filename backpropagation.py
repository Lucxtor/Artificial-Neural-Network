from activation import activation_funcs
from cost import cost_funcs

import numpy as np

def backpropagation(layers, cost_derivation, learningRate):
    """
    Executa o algoritmo de backpropagation em uma rede neural artificial.

    Parâmetros
    ----------
    layers : list
        Lista de dicionários representando as camadas da rede após o feedfoward. Cada camada deve conter:
        - "activation_func": nome da função de ativação
        - "output": saída da camada após a ativação
        - "input": entrada recebida pela camada
        - "weights": matriz de pesos da camada
    cost_derivation : np.ndarray
        Derivada da função de custo em relação à saída da rede.
    learningRate : float
        Taxa de aprendizado usada para atualização dos pesos.

    Retorna
    -------
    newWeights : list
        Lista contendo os novos pesos atualizados para cada camada.
    """
    newWeights = []
    
    error = None
    nextLayerWeights = None

    # Percorre as camadas da rede em ordem reversa (do output para o input)
    for layer in reversed(layers):
        # Obtém e calcula a derivada da função de ativação da camada atual
        _, activation_derivation = activation_funcs[layer["activation_func"]]

        derivation = activation_derivation(layer["output"])

        if error is None:
            # Primeira iteração: erro é a derivada do custo vezes a derivada da ativação
            error = cost_derivation * derivation
        else:
            # Para camadas intermediárias: propaga o erro da camada seguinte
            propagated_error = np.dot(error, nextLayerWeights[:, 1:]) # Ignora o viés

            error = propagated_error * derivation

        # Calcula e aplica o gradiente
        gradient = np.outer(error, layer["input"]) * learningRate

        nextLayerWeights = layer["weights"]

        newLayerWeights = nextLayerWeights - gradient

        newWeights.insert(0, newLayerWeights)
    
    return newWeights
