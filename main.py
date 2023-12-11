from Perceptron import Perceptron
from Cost import LogLoss
from ActivationFunction import sigmoid
from utils import *
from sklearn.datasets import make_blobs



if __name__ == "__main__":

    X_images, Y, X_test, y_test = load_data()
    """
    X_train: images we'll use to train the model. 3d matrix (tensor). images of 64x64px
    y_train: 0 if cat, 1 if dog
    """

    X = normalize_images(X_images)
    X_test = normalize_images(X_test)
    perceptron = Perceptron(X, Y, sigmoid, LogLoss)
    errors, accuracies, errors_test, accuracies_test = perceptron.train(100, 0.01, X_test, y_test)

    # X, Y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    # Y = Y.reshape((len(Y), 1)) # Making sure that the feature to predict is a vector (1 dimension)
    # perceptron = Perceptron(X, Y, sigmoid, LogLoss)
    # errors, accuracies, errors_test, accuracies_test = perceptron.train(100, 0.01)
    
    analysis(errors, errors_test, accuracies, accuracies_test)
