from sklearn.datasets import make_blobs
from Perceptron import Perceptron
from Cost import LogLoss
from ActivationFunction import sigmoid
import numpy as np
import matplotlib.pyplot as plt
from utils import *




if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_data()
    """
    X: features. n_features = 2 so X has 2 columns
    Y: 1 if the plant is toxic, 0 if not. Feature we try to predict
    """

    print(X_train.shape) # 1000 images of 64x64 px
    print(y_train.shape) # 0 if cat, 1 if dog

    # display_dataset(X, Y)

    # perceptron = Perceptron(X, Y, sigmoid, LogLoss)
    # perceptron.train(500, 0.9)

    # plant = np.array([1, 4]).reshape((1, 2))
    # print(perceptron.predict(plant))

    # print(perceptron.performance)

    # # display_error_graph(perceptron.errors)
    # display_decision_frontier(perceptron.W, perceptron.b)

    # plt.show()
