from Perceptron import Perceptron
from Cost import LogLoss
from ActivationFunction import sigmoid
import numpy as np
import matplotlib.pyplot as plt
from utils import *




if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_data()
    """
    X_train: images we'll use to train the model. 3d matrix (tensor)
    y_train: 0 if cat, 1 if dog
    """

    print(X_train.shape) # 1000 images of 64x64 px
    print(y_train.shape) # 1000x1
    print(np.unique(y_train, return_counts=True)) # 500 cats, 500 dogs, so balanced dataset
    print(X_train)
    # display_dataset(X, Y)

    # perceptron = Perceptron(X, Y, sigmoid, LogLoss)
    # perceptron.train(500, 0.9)

    # plant = np.array([1, 4]).reshape((1, 2))
    # print(perceptron.predict(plant))

    # print(perceptron.performance)

    # # display_error_graph(perceptron.errors)
    # display_decision_frontier(perceptron.W, perceptron.b)

    # plt.show()
