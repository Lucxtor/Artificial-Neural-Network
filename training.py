import numpy as np

from ann import feed_forward
from backpropagation import backpropagation
from cost import cost_funcs


def train(ann, epochs, x, y, learning_rate, cost_func_name):
    """
    Executa o treinamento de uma rede neural artificial (RNA).

    Parâmetros:
    ann : list
        Estrutura da rede neural, lista de dict {weights, activation_func}.
    epochs : int
        Número de épocas de treinamento.
    x : pandas.DataFrame
        Conjunto de dados de entrada.
    y : list or np.ndarray
        Conjunto de saídas desejadas.
    learning_rate : float
        Taxa de aprendizado.
    cost_func_name : string
        Nome da função de custo desejada.

    Retorna:
    stages : list
        Lista contendo os estados da RNA ao final de cada época.
    """

    stages = []  # Armazena o estado da rede após cada época

    # Loop sobre cada época
    for _ in range(epochs):
        # Loop sobre cada observação no conjunto de dados
        for observation_id in range(len(x)):

            observation = np.array(x.iloc[observation_id])

            # Executa a propagação direta
            y_pred, ann = feed_forward(ann, observation)

            y_true = np.array(y[observation_id])
            
            _, cost_derivation_func = cost_funcs[cost_func_name]

            # Calcula da saída
            cost_derivation = cost_derivation_func(y_true=y_true, y_pred=y_pred)

            # Executa a retropropagação para atualizar os pesos
            newWeights = backpropagation(
                layers=ann,
                cost_derivation=cost_derivation,
                learningRate=learning_rate
            )

            for i, layer in enumerate(ann):
                layer["weights"] = newWeights[i]

        # Salva o estado da rede ao final da época
        stages.append(ann)

    return stages[-1]
