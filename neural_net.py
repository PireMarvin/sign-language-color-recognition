import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialisation aléatoire
        self.W1 = np.random.rand(self.input_size, self.hidden_size) - 0.5
        self.b1 = np.random.rand(1, self.hidden_size) - 0.5
        self.W2 = np.random.rand(self.hidden_size, self.output_size) - 0.5
        self.b2 = np.random.rand(1, self.output_size) - 0.5

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        # Astuce numérique pour éviter l'explosion (overflow)
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)  # On garde Sigmoïde pour la couche cachée

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # CHANGEMENT ICI : On utilise Softmax pour la sortie
        self.a2 = self.softmax(self.z2)
        return self.a2

   # def forward(self, X):
    #    self.z1 = np.dot(X, self.W1) + self.b1
     #   self.a1 = self.sigmoid(self.z1)
      #  self.z2 = np.dot(self.a1, self.W2) + self.b2
       # self.a2 = self.sigmoid(self.z2)
        #return self.a2

   # def backward(self, X, y, output):
    #    m = X.shape[0]  # Nombre d'exemples

     #   self.output_error = y - output
     #   self.output_delta = self.output_error * self.sigmoid_derivative(output)

       # self.hidden_error = self.output_delta.dot(self.W2.T)
      #  self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)

        # Mise à jour avec division par m (moyenne)
        #self.W1 += (X.T.dot(self.hidden_delta) * self.learning_rate) / m
        #self.b1 += (np.sum(self.hidden_delta, axis=0, keepdims=True) * self.learning_rate) / m
        #self.W2 += (self.a1.T.dot(self.output_delta) * self.learning_rate) / m
        #self.b2 += (np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate) / m

    def backward(self, X, y, output):
        m = X.shape[0]

        # Avec Softmax + CrossEntropy, le delta de sortie est juste (y_pred - y_true)
        # C'est beaucoup plus stable !
        self.output_delta = (output - y)  # Attention : output - y ou y - output selon ta convention, ici (output - y)

        # Erreur cachée (Reste identique)
        self.hidden_error = self.output_delta.dot(self.W2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)

        # Mise à jour (Identique)
        self.W1 -= (X.T.dot(self.hidden_delta) * self.learning_rate) / m  # Note le -= car on descend le gradient
        self.b1 -= (np.sum(self.hidden_delta, axis=0, keepdims=True) * self.learning_rate) / m
        self.W2 -= (self.a1.T.dot(self.output_delta) * self.learning_rate) / m
        self.b2 -= (np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate) / m


    def train(self, X, y):
        output = self.forward(X)
        self.backward(X, y, output)

    def save_weights(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"✅ Modèle sauvegardé dans {filename}")

    def load_weights(self, filename):
        # C'est cette partie qui te manquait !
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"✅ Poids chargés depuis {filename}")