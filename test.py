import numpy as np
from ann import initialize_layers, feed_forward
from backpropagation import backpropagation
from prepareData.prepareDataBinaryClassification import prepareDataBinaryClassification


layers = [{'neurons': 6, 'activation_function': 'relu'}, {'neurons': 1, 'activation_function': 'sigmoid'}]

ann_layer = initialize_layers(layers, c_inputs=6)

x_train, x_test, y_train, y_test = prepareDataBinaryClassification()

# Recebe as camadas da rede, número de épocas, entradas, saídas, taxa de aprendizado e função de custo
def train(ann, epochs, x, y, learning_rate, cost_func):

  stages = []

  # Itera sobre o número de épocas
  for epoch in range(epochs):
    # Itera sobre as observações
    for observation_id in range(len(x)):
        # print("rede =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--")
      if epoch == 0 and observation_id == 1 or epoch == 99 and observation_id == 218:
        for i, layer in enumerate(ann):
          print(i, layer)
        # print("fim rede -=-=-=-=-=-=-=-=-=-=-=-")
        # print()

      observation = np.array(x.iloc[observation_id])

      y_pred, ann = feed_forward(ann, observation)

      newWeights = backpropagation(layers=ann, y_pred=y_pred, y_true=y[observation_id], cost_function=cost_func, learningRate=learning_rate)

      for i, layer in enumerate(ann):
          # print(newWeights[i])
          layer["weights"] = newWeights[i]
    stages.append(ann)
  return stages

stages = train(ann=ann_layer, epochs=100, x=x_train, y=y_train, learning_rate=0.002, cost_func='binary_cross_entropy')

def accuracy_func(predictions, labels):
    return int(sum(labels == predictions[0]) / len(labels) * 100)

def evaluate(ann, x, y):
  predictions = []
  for observation_id in range(len(x)):

    input = np.array(x.iloc[observation_id])
    prediction = feed_forward(ann, input)
    predictions.append(prediction[0][0])

  # Obtém métricas de avaliação:
  # print(predictions)
  for pred in predictions:
    print(pred)
  accuracy = accuracy_func(np.round(np.array(predictions)), np.round(np.array(y)))
  print(" accuracy: ", accuracy)

evaluate(stages[-1], x_test, y_test)

# Lembrar de verificar se é preciso denormalizar os dados de saída