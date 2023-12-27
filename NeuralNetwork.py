import numpy as np
from Cost import *
from ActivationFunction import *
from sklearn.metrics import accuracy_score



class Layer():

	def __init__(self, nb_neurons: int, nb_inputs: int) -> None:
		"""
		nb_neurons: number of neurons of the layer
		nb_inputs: number of inputs of the layer, which is the number neurons of the previous layer
		"""
		
		self.nb_neurons = nb_neurons
		self.W, self.b = self.init_weights(nb_inputs)
		self.A: np.ndarray

	def init_weights(self, nb_inputs: int):
		
		W = np.random.randn(self.nb_neurons, nb_inputs)
		b = np.random.randn(self.nb_neurons, 1)
		# W = W.reshape(len(W), self.nb_neurons)
		return W, b

	def activation(self, X: np.ndarray, activation_function: callable, test: bool = False) -> np.ndarray:

		Z = self.W.dot(X) + self.b
		A = activation_function(Z)

		if not test: self.A = A
		return A
	
	def update_weights(self, dW: np.ndarray, db: np.ndarray, η):
		
		self.W = self.W - η * dW
		self.b = self.b - η * db



class NeuralNetwork():

	def __init__(self, X: np.ndarray, Y: np.ndarray, hidden_layers: list[int], activation_function: callable) -> None:
		"""
		structure: Structure of the neural network. @example: [2, 4] # Layer 1: 2 neurons | Layer 2: 4 neurons
		"""

		self.X = adapt(X)
		self.Y = adapt(Y)
		self.layers = self.init_layers(hidden_layers)
		self.activation_function = activation_function

	
	def init_layers(self, hidden_layers: list[int]) -> list[Layer]:

		layers = []

		for i in range(len(hidden_layers)):
			layers.append(Layer(nb_neurons=hidden_layers[i], nb_inputs=self.X.shape[0] if i==0 else hidden_layers[i-1]))
		
		layers.append(Layer(nb_neurons=self.Y.shape[0], nb_inputs=hidden_layers[-1]))
		# last neuron to aggregate the last hidden layer and to compare to Y

		return layers


	def train(self, epochs: int = 100, η: float = 0.05, X_test: np.ndarray = None, Y_test: np.ndarray = None):

		errors = []
		accuracies = []
		errors_test = []
		accuracies_test = []
		X_test = adapt(X_test)
		Y_test = adapt(Y_test)

		for _ in range(epochs):
			
			output = self.forward(self.X)
			L = LogLoss(output, self.Y)
			errors.append(L.value())
			accuracies.append(self.accuracy(self.X, self.Y))

			if X_test is not None and Y_test is not None:
				L_test = LogLoss(self.forward(X_test, test=True), Y_test)
				errors_test.append(L_test.value())
				accuracies_test.append(self.accuracy(X_test, Y_test))

			self.backward(η)

		return errors, accuracies, errors_test, accuracies_test


	def forward(self, X: np.ndarray, test: bool = False):

		layer_output = X
		
		for layer in self.layers:
			layer_output = layer.activation(layer_output, self.activation_function, test)

		return layer_output
	

	def backward(self, η):
		"""
		Updates the weights of each layer. For a LogLoss cost function
		"""

		m = self.Y.shape[1]
		dZ = self.layers[-1].A - self.Y # final dZ

		for i in reversed(range(len(self.layers))):

			prev_A = self.X if i == 0 else self.layers[i - 1].A # activations of the previous layer
			
			dW = 1/m * dZ.dot(prev_A.T)
			db = 1/m * np.sum(dZ, axis=1, keepdims=True)
			
			self.layers[i].update_weights(dW, db, η)

			dZ = self.layers[i].W.T.dot(dZ) * prev_A * ( 1 - prev_A )


	def predict(self, X, val = None):
		activation = self.forward(X, test=True)
		if val == "numbers":
			return activation.flatten()
		return activation >= 0.5 # this value depends on the activation function


	def accuracy(self, X: np.ndarray, Y: np.ndarray):
		"""
		Returns the accuracy of the model
		"""
		
		Y_pred = self.predict(X)
		return accuracy_score(Y.flatten(), Y_pred.flatten())


def adapt(X: np.ndarray):
	if X.ndim == 1:
		return X.reshape((1, X.shape[0]))
	return X.T