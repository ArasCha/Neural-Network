from Perceptron import Perceptron
from Cost import LogLoss
from ActivationFunction import sigmoid
from utils import *
from NeuralNetwork import NeuralNetwork



def cat_dogs():

	X_images, Y, X_test, y_test = load_data()
	"""
	X_train: images we'll use to train the model. 3d matrix (tensor). images of 64x64px
	y_train: 0 if cat, 1 if dog
	"""

	X = normalize_images(X_images)
	X_test_normalized = normalize_images(X_test)

	perceptron = Perceptron(X, Y, sigmoid, LogLoss)
	errors, accuracies, errors_test, accuracies_test = perceptron.train(10000, 0.01, X_test_normalized, y_test)



def circles():
	from sklearn.datasets import make_circles
	
	X, Y = make_circles(n_samples=1000, noise=0.1, factor=0.3)
	X_test, Y_test = make_circles(n_samples=1000, noise=0.1, factor=0.3)

	nn = NeuralNetwork(X, Y, [3], sigmoid)
	errors, accuracies, errors_test, accuracies_test = nn.train(1200, 0.5, X_test, Y_test)

	analysis(errors, errors_test, accuracies, accuracies_test)



def digits():
	from sklearn.model_selection import train_test_split
	from sklearn.datasets import load_digits

	data = load_digits()
	# show_images(data.images, data.target, nb=13)
	X = data.data
	y = data.target

	Y = []
	for target_value in y:
		arr = [0]*10
		arr[target_value] = 1
		Y.append(arr)
	Y = np.array(Y)
	
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

	nn = NeuralNetwork(X_train, Y_train, [100, 100], sigmoid)

	errors, accuracies, errors_test, accuracies_test = nn.train(700, 0.6, X_test, Y_test)
	
	analysis(errors, errors_test, accuracies, accuracies_test)

	for i in range(40): # test the 40 first numbers
		sample = X_test[i].reshape((X_test[i].shape[0], 1))
		expected_result = Y_test[i]
		prediction = nn.predict(sample)
		np.set_printoptions(precision=2)
		# print(f"expected result: {expected_result}, prediction: {prediction}")
		print(f"expected result: {np.argmax(expected_result)}, prediction: {np.argmax(prediction)}")