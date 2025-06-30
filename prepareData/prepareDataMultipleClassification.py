import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

"""
Values:
0 - low cost
1 - medium cost
2 - high cost
3 - very high cost
"""


def prepareDataMultipleClassification():
  data = pd.read_csv('prepareData/mobile_price_multiple_classification.csv')

  # print(data.head())
  
  numerical_cols = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen',  'wifi']

  # Inicializando o MinMaxScaler
  scaler = MinMaxScaler(feature_range=(0, 1))

  # Aplicando a normalização às colunas numéricas
  data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
  
  X = data.drop(columns=['price_range'])

  y = pd.get_dummies(data["price_range"], prefix="price", sparse=False)

  # Divisão dos dados em conjuntos de treino e teste na proporção de 80% para treino e 20% para teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  
  return X_train, X_test, y_train, y_test