import numpy as np
from activation import activation_funcs
from cost import cost_funcs

def initialize_layers(layers, c_inputs):
  """
  Inicializa as camadas da rede neural com pesos aleatórios.

  Parâmetros
  ----------
  layers : list of dict
      Lista onde cada dicionário contém:
      - 'neurons': número de neurônios na camada
      - 'activation_function': nome da função de ativação usada na camada
  c_inputs : int
      Número de entradas da rede (atributos do dataset).

  Retorna
  -------
  initialized_layers : list
      Lista de camadas com pesos e função de ativação configurados.
  """

  initialized_layers = []

  for i, layer in enumerate(layers):
    # se for a primeira camada, o número de entradas é igual ao número de atributos
    if i == 0:
      input_size = c_inputs
    else:
      input_size = layers[i - 1]['neurons']
    # o número de saídas é igual ao número de neurônios da camada atual
    output_size = layer['neurons']
    
    # Inicializa os pesos com valores aleatórios pequenos e mais um para o viés
    weight_matrix = np.random.uniform(low=-0.33, high=0.33,size=(output_size,input_size + 1))
  
    initialized_layers.append({
        'weights': weight_matrix,
        'activation_func': layer['activation_function']
    })
  return initialized_layers

def feed_forward(layers, inputs):
  """
  Executa o algoritmo de propagação direta (feedforward) em uma rede neural.

  Parâmetros
  ----------
  layers : list
      Lista de camadas já inicializadas com pesos e funções de ativação.
  inputs : np.ndarray
      Vetor de entrada com os atributos de uma observação.

  Retorna
  -------
  activation : np.ndarray
      Saída final da rede após a última camada (predição).
  layers : list
      Lista atualizada das camadas com entradas e saídas armazenadas para uso posterior no backpropagation.
  """
    
  activation = inputs

  # Itera sobre cada camada da rede neural
  for layer in layers:

    # Concatena para o input + viés da matriz de pesos
    activation = np.concatenate(([1], activation))
    
    # salva input na layer
    layer['input'] = activation

    # Calcula a saída da camada atual: z = w' * a + b, onde w' é a transposta da matriz de pesos, a é o input da camada e b é o viés que foi concatenado
    z = np.dot(activation, layer['weights'].T)
    
    # Extrai e calcula a função de ativação do dicionário activation_funcs
    activation_func, _ = activation_funcs[layer['activation_func']]
    
    activation = activation_func(z)
  
    # salva a operacao na layer
    layer['output'] = z

  return activation, layers