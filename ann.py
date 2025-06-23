import numpy as np

class ANN:
    def __init__(self, numAttributes, numLayers):
        
      # número de dimensões de entrada
      self.numAttributes = numAttributes

        # número de camadas
      self.numLayers = numLayers

    # Método para o usuário escolher o número de neurônios e função de ativação de cada camada
    def chooseNeuronsAndActivation(self):
      self.neurons = []
      self.activationFunctions = []
        
      for i in range(self.numLayers):
          numNeurons = int(input(f"Número de neurônios na camada {i + 1}: "))
          activationFunction = input(f"Função de ativação para a camada {i + 1} (sigmoid, relu, softmax): ")
          self.neurons.append(numNeurons)
          self.activationFunctions.append(activationFunction)
      
      print("Configuração da rede neural:")
      print(f"Número de atributos de entrada: {self.numAttributes}")
      print(f"Número de neurônios em cada camada: {self.neurons}")
      print(f"Funções de ativação de cada camada: {self.activationFunctions}")

    # Função para inicializar os pesos e biases da rede neural de todas as camadas
    def initializeWeightsAndBiases(self):
      self.weights = []
      self.biases = []

      # Inicializa os pesos e biases para cada camada
      for i in range(self.numLayers):
          
          # Se for a primeira camada, o número de entradas é igual ao número de atributos
          if i == 0:
              inputSize = self.numAttributes

          # Caso contrário, o número de entradas é igual ao número de neurônios da camada anterior
          else:
              inputSize = self.neurons[i - 1]

          # O número de saídas é igual ao número de neurônios da camada atual
          outputSize = self.neurons[i]

          # Inicializa os pesos com valores aleatórios pequenos e os biases com zeros
          weightMatrix = np.random.rand(inputSize, outputSize) * 0.01
          biasVector = np.zeros((1, outputSize))

          self.weights.append(weightMatrix)
          self.biases.append(biasVector)

      print("Pesos e biases inicializados.")

    # Função para realizar o feedforward da rede neural
    def feedforward(self, inputs):
        
      # Inicializa a ativação com os inputs iniciais
      activation = inputs

      # Itera sobre cada camada da rede neural
      for i in range(self.numLayers):
          
          # Calcula a saída da camada atual: z = w * a + b, onde w são os pesos, a é a ativação da camada anterior e b são os biases
          z = np.dot(activation, self.weights[i]) + self.biases[i]
          
          # Aplica a função de ativação correspondente
          if self.activationFunctions[i] == 'sigmoid':
              activation = 1 / (1 + np.exp(-z))
          elif self.activationFunctions[i] == 'relu':
              activation = np.maximum(0, z)
          elif self.activationFunctions[i] == 'softmax':
              expZ = np.exp(z - np.max(z))
              activation = expZ / np.sum(expZ, axis=1, keepdims=True)
        
    # Calcula o custo (erro) entre as saídas da rede e os alvos
    def cost(self):
      pass
    
    # Implementação do backpropagation para treinar a rede neural
    def backpropagation(self):
      pass

    # Função para treinar a rede neural usando o algoritmo de backpropagation
    def train(self, inputs, targets, epochs, learningRate):
      pass
    
    # Realiza o feedforward para obter as previsões
    def predict(self, inputs):
      pass
    
    # Função para avaliar a performance da rede neural
    def evaluate(self, inputs, targets):
      pass

