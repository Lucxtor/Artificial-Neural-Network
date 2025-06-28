import numpy as np
from activation import activation_funcs
from cost import cost_funcs

# Recebe as layers sendo elas um array de dicionarios contendo o número de neurônios e a sua respectiva função de ativação
# e c_inputs que é o número de entradas da rede neural
def initialize_layers(layers, c_inputs):

  initialized_layers = []

  for i, layer in enumerate(layers):
    # se for a primeira camada, o número de entradas é igual ao número de atributos
    if i == 0:
      input_size = c_inputs
    else:
      input_size = layers[i - 1]['neurons']
    # o número de saídas é igual ao número de neurônios da camada atual
    output_size = layer['neurons']
    
    # Inicializa os pesos com valores aleatórios pequenos e os biases com zeros
    # weight_matrix = (np.random.rand(input_size, output_size) * 0.01)
    weight_matrix = np.random.uniform(low=-0.33, high=0.33,size=(output_size,input_size))
  
    bias_vector = np.ones((1, output_size))
    initialized_layers.append({
        'weights': weight_matrix,
        'biases': bias_vector,
        'activation_func': layer['activation_function']
    })
  return initialized_layers

# Função para realizar o feedforward da rede neural
# Inputs é uma matriz onde cada linha é um exemplo de entrada
def feed_forward(layers, inputs):
    
  # Inicializa a ativação com os inputs iniciais
  activation = inputs

  # Itera sobre cada camada da rede neural
  for layer in layers:
    
    # salva input na layer
    layer['input'] = activation

    # Calcula a saída da camada atual: z = w * a + b, onde w são os pesos, a é a ativação da camada anterior e b são os biases
    z = np.dot(activation, layer['weights'].T) + layer['biases']
    
    # Pega a função de ativação do dicionário activation_funcs
    activation_func, _ = activation_funcs[layer['activation_func']]
    
    # Calcula o resultado da função de ativação
    # print(f"valor de z: {z}")
    activation = activation_func(z)
    # print(f"valor da ativação{activation}")
    # salva a operacao na layer
    layer['output'] = z

  # Retorna o resultado do feedfoward (resultado da rede)
  return activation, layers

# Função para treinar a rede neural usando o algoritmo de backpropagation
def train(self, inputs, targets, epochs, learningRate):
  pass

# Função para avaliar a performance da rede neural
def evaluate(self, inputs, targets):
  pass
