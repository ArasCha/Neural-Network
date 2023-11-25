import numpy as np
import matplotlib.pyplot as plt
import h5py


def display_dataset(X: list, Y: list):
    plt.scatter(X[:,0], X[:,1], c=Y)

def display_error_graph(errors: list):
    plt.plot(errors)

def display_decision_frontier(W, b):

    x1 = np.linspace(0, 5, 100)
    x2 = ( -W[0] * x1 - b ) / W[1]

    plt.plot(x1, x2, c="green", lw=3)


def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test