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
    X_test_normalized = normalize_images(X_test)

    perceptron = Perceptron(X, Y, sigmoid, LogLoss)
    errors, accuracies, errors_test, accuracies_test = perceptron.train(500, 0.01, X_test_normalized, y_test)
    
    print(perceptron.accuracy())

    # image, value = X_test[6], y_test[6]
    # # show_images(image, value) # show the image we chose
    # image_n = normalize_images(image)
    # print(perceptron.predict(image_n)) # False: cat, True: dog
    
    analysis(errors, errors_test, accuracies, accuracies_test)
