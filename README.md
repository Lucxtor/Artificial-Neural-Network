# Artificial-Neural-Network

Este repositório apresenta uma implementação de uma Rede Neural Artificial (RNA) desenvolvida do zero em Python, utilizando apenas bibliotecas de baixo nível e recursos fundamentais da linguagem. O objetivo é proporcionar aprendizado prático sobre os conceitos teóricos de redes neurais, incluindo feedforward, backpropagation, funções de ativação, funções de custo e preparação de dados para diferentes tarefas de aprendizado supervisionado.

## Autores

- Vinícius Clemente (21103915)
- Matheus Beilfuss (21104290)
- Luis Felipe Fabiane (21106213)

## Bibliotecas e Dependências

O projeto utiliza as seguintes bibliotecas Python:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib` (apenas para visualização em notebooks)
- (Opcional) `locale` para formatação de valores monetários

Para instalar as dependências, execute:

```sh
pip install numpy pandas scikit-learn matplotlib
```

## Estrutura dos Arquivos

- `ann.py`: implementação das funções principais da rede neural (inicialização, feedforward, etc.)
- `backpropagation.py`: algoritmo de retropropagação para ajuste dos pesos.
- `activation.py`: funções de ativação e suas derivadas.
- `cost.py`: funções de custo e suas derivadas.
- `training.py`: função de treinamento da rede.
- `prepareData/`: scripts para preparação dos dados e datasets utilizados: prepareDataRegression.py, prepareDataMultipleClassification.py, prepareDataBinaryClassification.py
- `test_regression.py`: scripts de teste para tarefas de regressão.
- `test_multiple.py`: script de teste para classificação múltipla.
- `test_binary.py`: script de teste para classificação binária.
- `multiple_classification.ipynb`: Notebook para classificação múltipla (execução local).
- `regression.ipynb`: notebook para regressão (execução local).
- `RNA.ipynb`: notebook completo para execução no Google Colab.

## Como Executar

Para treinar uma rede neural com seus próprios dados ou arquitetura, crie um novo arquivo Python (por exemplo, `meu_teste.py`) e siga o padrão dos arquivos de teste existentes. Um exemplo básico:

```
import numpy as np
from ann import initialize_layers, feed_forward
from backpropagation import backpropagation
from prepareData.prepareDataRegression import prepareDataRegression
from cost import cost_funcs
from training import train

# Defina a arquitetura da rede
layers = [
    {'neurons': 16, 'activation_function': 'relu'},
    {'neurons': 8,  'activation_function': 'relu'},
    {'neurons': 1,  'activation_function': 'identity'}
]

# Inicialize as camadas
ann_layer = initialize_layers(layers, c_inputs=7)

# Carregue os dados (altere para a função de preparação desejada)
x_train, x_test, y_train, y_test, min_max_values = prepareDataRegression()

# Treine a rede
stages = train(
    ann=ann_layer,
    epochs=100,
    x=x_train,
    y=y_train,
    learning_rate=0.002,
    cost_func_name='mse'
)

# Faça predições e avalie
def evaluate(ann, x, y):
    predictions = []
    for observation_id in range(len(x)):
        input = np.array(x.iloc[observation_id])
        prediction, _ = feed_forward(ann, input)
        predictions.append(prediction)
    print("Predições:", predictions[:5])

evaluate(stages, x_test, y_test)
```

- Altere a função de preparação de dados (`prepareDataRegression`, `prepareDataMultipleClassification`, etc.) conforme o problema.
- Modifique a arquitetura da rede (layers) conforme necessário.
- Use as funções de avaliação dos arquivos de teste como referência para métricas mais detalhadas.

## Sobre os Notebooks

- Notebooks para execução local:
  - `multiple_classification.ipynb`: demonstra a aplicação da RNA em um problema de classificação múltipla.
  - `regression.ipynb`: demonstra a aplicação da RNA em um problema de regressão.
  - `binary_classification.ipynb`: demonstra a aplicação da RNA em um problema de classificação binária.
  - Estes notebooks já utilizam os datasets presentes na pasta `prepareData/` e podem ser executados diretamente em ambiente local.
- Notebook completo para Google Colab:
  - `RNA.ipynb`: notebook consolidado com todas as funções, exemplos e explicações, voltado para execução no Google Colab.
  - **Atenção**: Para rodar o `RNA.ipynb` no Colab, será necessário fazer upload manual dos arquivos de dataset (`.csv`) para o ambiente do Colab antes de executar as células de preparação de dados.
