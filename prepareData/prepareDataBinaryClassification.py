import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = 'prepareData/penguins_binary_classification.csv'

data = pd.read_csv(DATA_PATH)

# print(data.head())

# Realizando o label encoding da coluna 'species'
data['species'] = data['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})

X = data

y = data.pop('species').values

# Divisão dos dados em conjuntos de treino e teste na proporção de 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Ainda é necessário adicionar o bias, mas isso na inicialização da rede neural, e talvez seja necessário normalizar os dados
