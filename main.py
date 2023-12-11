from Perceptron import Perceptron
from Cost import LogLoss
from ActivationFunction import sigmoid
from utils import *
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_circles



if __name__ == "__main__":

    X_images, Y, X_test, y_test = load_data()
    """
    X_train: images we'll use to train the model. 3d matrix (tensor). images of 64x64px
    y_train: 0 if cat, 1 if dog
    """

    X = normalize_images(X_images)
    X_test_normalized = normalize_images(X_test)
    X_test_normalized = X_test_normalized.T
    y_test = y_test.reshape((1, y_test.shape[0]))

    # perceptron = Perceptron(X, Y, sigmoid, LogLoss)
    # errors, accuracies, errors_test, accuracies_test = perceptron.train(10000, 0.01, X_test_normalized, y_test)

    X, Y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)

    X = X.T
    Y = Y.reshape((1, Y.shape[0]))

    nn = NeuralNetwork(X, Y, [8, 16, 32], sigmoid)

    errors, accuracies, errors_test, accuracies_test = nn.train(100, 0.1)
    
    analysis(errors, errors_test, accuracies, accuracies_test)
