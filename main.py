from Perceptron import Perceptron
from Cost import LogLoss
from ActivationFunction import sigmoid
from utils import *




if __name__ == "__main__":

    X_images, Y, X_test, y_test = load_data()
    """
    X_train: images we'll use to train the model. 3d matrix (tensor). images of 64x64px
    y_train: 0 if cat, 1 if dog
    """

    X = normalize_images(X_images)

    perceptron = Perceptron(X, Y, sigmoid, LogLoss)
    perceptron.train(100, 2)
    
    print(perceptron.performance)

    image, value = X_test[2], y_test[2]
    show_images(image, value) # show the image we chose
    image_n = normalize_images(image)
    print(perceptron.predict(image_n)) # False: cat, True: dog
    
    # print(perceptron.errors) # nan problem
    # display_error_graph(perceptron.errors)
