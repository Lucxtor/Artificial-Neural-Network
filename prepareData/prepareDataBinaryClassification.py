import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def prepareDataBinaryClassification():
  # print(data.head())
  DATA_PATH = 'prepareData/penguins_binary_classification.csv'

  data = pd.read_csv(DATA_PATH)

  # Realizando o label encoding da coluna 'species'
  data['species'] = data['species'].map({'Adelie': 0, 'Gentoo': 1})

  # Realizando o label encoding da coluna 'island'
  data['island'] = data['island'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})

  numerical_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year']

  # Inicializando o MinMaxScaler
  scaler = MinMaxScaler(feature_range=(0, 1))

  # Aplicando a normalização às colunas numéricas
  data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

  X = data

  y = data.pop('species').values

  # Divisão dos dados em conjuntos de treino e teste na proporção de 80% para treino e 20% para teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  return X_train, X_test, y_train, y_test

# Ainda é necessário adicionar o bias, mas isso na inicialização da rede neural, e talvez seja necessário normalizar os dados
