import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ann import initialize_layers, feed_forward
from backpropagation import backpropagation
from prepareData.prepareDataRegression import prepareDataRegression
from cost import cost_funcs
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import locale

from training import train

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# input_size = 6

layers = [{'neurons': 7, 'activation_function': 'relu'}, {'neurons': 1, 'activation_function': 'identity'}]

ann_layer = initialize_layers(layers, c_inputs=7)

x_train, x_test, y_train, y_test, min_max_house_price = prepareDataRegression()

stages = train(ann=ann_layer, epochs=20, x=x_train, y=y_train, learning_rate=0.002, cost_func_name='mse')

def accuracy_func(predictions, labels):
    return int(sum(labels == predictions) / len(labels) * 100)

def regression_report(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    report = (
        f"Regression Report\n"
        f"------------------------\n"

        f"Mean Squared Error (MSE):       {mse:.4f}\n"
        f"Root Mean Squared Error (RMSE): {locale.currency(rmse, grouping=True)}\n"
        f"Mean Absolute Error (MAE):      {mae:.4f}\n"
        f"R² Score:                       {r2:.4f}"
    )
    return report

def evaluate(ann, x, y, min_max_house_price):
  predictions = []
  for observation_id in range(len(x)):

    input = np.array(x.iloc[observation_id])
    prediction, _ = feed_forward(ann, input)
    predictions.append(prediction)

  scaler = MinMaxScaler(feature_range=min_max_house_price)

  y_denormalized = scaler.fit_transform(y.reshape(-1, 1))
  predictions_denormalized = scaler.fit_transform(predictions)

#   for pred in range(10):
#     print(predictions_denormalized[pred], y_denormalized[pred])

  print("result:")
  print(regression_report(y_denormalized, predictions_denormalized))

evaluate(stages, x_test, y_test, min_max_house_price)


# Lembrar de verificar se é preciso denormalizar os dados de saída
