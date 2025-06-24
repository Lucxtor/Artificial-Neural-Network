import numpy as np
from activation import activation_funcs
from cost import cost_funcs

class ANN:
    def __init__(self, numAttributes, numLayers):
        
      # número de dimensões de entrada
      self.numAttributes = numAttributes

      # número de camadas
      self.numLayers = numLayers

      # Lista para armazenar as ativações de cada camada (incluindo a entrada)
      self.activations = []
      # Lista para armazenar os valores de 'z' (saída não ativada) de cada camada
      self.zs = []

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
        
      # Limpa as listas de ativações e zs para uma nova passagem
      self.activations = [inputs] # A primeira ativação é a entrada
      self.zs = []

      activation = inputs

      for i in range(self.numLayers):
          # Calcula a saída da camada atual: z = w * a + b
          z = np.dot(activation, self.weights[i]) + self.biases[i]
          self.zs.append(z) # Armazena o valor de z

          # Pega a função de ativação e sua derivada do dicionário activation_funcs
          activation_func, _ = activation_funcs[self.activationFunctions[i]]
          
          # Calcula o resultado da função de ativação
          activation = activation_func(z)
          self.activations.append(activation) # Armazena a ativação

      # Retorna o resultado do feedfoward (resultado da rede)
      return activation

    # Calcula o custo (erro) entre as saídas da rede e os alvos
    def cost(self, y_true, y_pred, cost_function_name):
      # Pega a função de custo do dicionário cost_funcs
      cost_func, _ = cost_funcs[cost_function_name]
      
      # Calcula e retorna o custo
      return cost_func(y_true, y_pred)
    
    # Implementação do backpropagation para treinar a rede neural
    def backpropagation(self, y_true, cost_function_name):
        # Inicializa listas para armazenar os gradientes dos pesos e biases
        # Serão os gradientes acumulados para um batch, ou para uma única amostra no SGD
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # 1. Erro na camada de saída (delta da última camada)
        # y_pred é a ativação da última camada
        y_pred = self.activations[-1] 
        
        # Pega a derivada da função de custo
        _, cost_derivative = cost_funcs[cost_function_name]
        
        # Pega a derivada da função de ativação da última camada
        _, activation_derivative = activation_funcs[self.activationFunctions[-1]]

        # Calcula o delta (erro) da camada de saída
        delta = cost_derivative(y_true, y_pred) * activation_derivative(self.zs[-1])
        
        # Calcula os gradientes para a última camada
        nabla_b[-1] = np.sum(delta, axis=0, keepdims=True) # Somar para o batch
        nabla_w[-1] = np.dot(self.activations[-2].T, delta) 

        # 2. Propagação reversa do erro pelas camadas ocultas
        # Itera de trás para frente pelas camadas ocultas (excluindo a camada de saída)
        for l in range(2, self.numLayers + 1):
            z = self.zs[-l] # z da camada atual (indo para trás)
            
            # Pega a derivada da função de ativação da camada atual
            _, activation_derivative = activation_funcs[self.activationFunctions[-l]]
            sp = activation_derivative(z) 
            
            # Calcula o delta para a camada atual
            delta = np.dot(delta, self.weights[-l+1].T) * sp
            
            # Calcula os gradientes para a camada atual
            nabla_b[-l] = np.sum(delta, axis=0, keepdims=True) # Somar para o batch
            nabla_w[-l] = np.dot(self.activations[-l-1].T, delta) 
            
        return nabla_w, nabla_b

    # Função para treinar a rede neural usando o algoritmo de backpropagation
    def train(self, X_train, y_train, epochs, learningRate, cost_function_name='mse'):
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Para cada época, itera sobre cada amostra (ou mini-batch)
            # Para simplificar, vamos usar o SGD (Stochastic Gradient Descent) aqui,
            # atualizando os pesos após cada amostra.
            # Para mini-batch, você precisaria dividir X_train e y_train em batches.
            
            total_cost = 0

            for i in range(num_samples):
                # Pega uma única amostra
                x = X_train[i:i+1] # Garante que x_train seja 2D (1, num_features)
                y = y_train[i:i+1] # Garante que y_train seja 2D (1, num_outputs)

                # 1. Feedforward
                y_pred = self.feedforward(x)
                
                # 2. Calcular o custo (opcional para exibição, mas útil para acompanhar o progresso)
                current_cost = self.cost(y, y_pred, cost_function_name)
                total_cost += current_cost

                # 3. Backpropagation para obter os gradientes
                nabla_w, nabla_b = self.backpropagation(y, cost_function_name)

                # 4. Atualizar pesos e biases usando o gradiente descendente
                for l in range(self.numLayers):
                    self.weights[l] -= learningRate * nabla_w[l]
                    self.biases[l] -= learningRate * nabla_b[l]
            
            # Média do custo por amostra na época
            avg_cost = total_cost / num_samples
            print(f"Época {epoch+1}/{epochs}, Custo: {avg_cost:.4f}")

    # Realiza o feedforward para obter as previsões
    def predict(self, inputs):
      return self.feedforward(inputs)
    
    # Função para avaliar a performance da rede neural
    def evaluate(self, inputs, targets, cost_function_name='mse'):
        predictions = self.predict(inputs)
        final_cost = self.cost(targets, predictions, cost_function_name)
        
        # Para classificação, você pode adicionar métricas como acurácia
        if self.activationFunctions[-1] == 'softmax':
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(targets, axis=1)
            accuracy = np.mean(predicted_classes == true_classes)
            print(f"Custo final: {final_cost:.4f}, Acurácia: {accuracy:.4f}")
            return final_cost, accuracy
        else:
            print(f"Custo final: {final_cost:.4f}")
            return final_cost