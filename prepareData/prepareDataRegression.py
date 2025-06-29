import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Do Moodle: https://www.kaggle.com/datasets/prokshitha/home-value-insights
# TODO: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

def prepareDataRegression():
  DATA_PATH = 'prepareData/house_price_regression_dataset.csv'

  data = pd.read_csv(DATA_PATH)

  # print(data.head())

  numerical_cols = ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality', 'House_Price']

  min_max_house_price = (data['House_Price'].min(), data['House_Price'].max())

  # Inicializando o MinMaxScaler
  scaler = MinMaxScaler(feature_range=(0, 1))

  # Aplicando a normalização às colunas numéricas
  data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

  X = data

  y = data.pop('House_Price').values

  
  # Divisão dos dados em conjuntos de treino e teste na proporção de 80% para treino e 20% para teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  return X_train, X_test, y_train, y_test, min_max_house_price