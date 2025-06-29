import pandas as pd
from sklearn.model_selection import train_test_split

def prepareDataRegression():
  DATA_PATH = 'prepareData/house_price_regression_dataset.csv'

  data = pd.read_csv(DATA_PATH)

  # print(data.head())

  # Selecionando todas as colunas, exceto a última (features)
  X = data.iloc[:, :-1].values

  # Selecionando a última coluna contendo os alvos (targets)
  y = data.iloc[:, -1].values

  # Divisão dos dados em conjuntos de treino e teste na proporção de 80% para treino e 20% para teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  return X_train, X_test, y_train, y_test

# Ainda é necessário adicionar o bias, mas isso na inicialização da rede neural, e talvez seja necessário normalizar os dados
