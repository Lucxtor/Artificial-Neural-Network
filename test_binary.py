import numpy as np
from ann import initialize_layers, feed_forward
from backpropagation import backpropagation
from prepareData.prepareDataBinaryClassification import prepareDataBinaryClassification
from cost import cost_funcs
from sklearn.metrics import classification_report

from training import train

input_size = 6

layers = [
    {'neurons': 3, 'activation_function': 'relu'},
    {'neurons': 1, 'activation_function': 'sigmoid'}
    ]

ann_layer = initialize_layers(layers, c_inputs=6)

x_train, x_test, y_train, y_test = prepareDataBinaryClassification()

stages = train(ann=ann_layer, epochs=32, x=x_train, y=y_train, learning_rate=0.002, cost_func_name='binary_cross_entropy')

def accuracy_func(predictions, labels):
    return int(sum(labels == predictions) / len(labels) * 100)

def evaluate(ann, x, y):
  predictions = []
  for observation_id in range(len(x)):

    input = np.array(x.iloc[observation_id])
    prediction, _ = feed_forward(ann, input)
    predictions.append(prediction[0])

  predictions = np.round(predictions)
  print("result:")
  print(classification_report(y, predictions))

evaluate(stages, x_test, y_test)

