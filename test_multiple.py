import numpy as np
from ann import initialize_layers, feed_forward
from backpropagation import backpropagation
from prepareData.prepareDataMultipleClassification import prepareDataMultipleClassification
from cost import cost_funcs
from training import train
from sklearn.metrics import classification_report

input_size = 20

layers = [{'neurons': 10, 'activation_function': 'relu'}, {'neurons': 10, 'activation_function': 'relu'}, {'neurons': 4, 'activation_function': 'softmax'}]

ann_layer = initialize_layers(layers, c_inputs=20)

x_train, x_test, y_train, y_test = prepareDataMultipleClassification()

stages = train(ann=ann_layer, epochs=10, x=x_train, y=np.array(y_train), learning_rate=0.002, cost_func_name='categorical_cross_entropy')

def evaluate(ann, x, y):
  predictions = []
  for observation_id in range(len(x)):

    input = np.array(x.iloc[observation_id])
    prediction, _ = feed_forward(ann, input)
    one_hot_prediction = np.zeros_like(prediction)
    one_hot_prediction[np.argmax(prediction)] = 1
    predictions.append(one_hot_prediction)

  # for pred in predictions:
  #   print(pred)
  print("result:")
  print(classification_report(y, predictions))

evaluate(stages, x_test, y_test)


# Lembrar de verificar se é preciso denormalizar os dados de saída
