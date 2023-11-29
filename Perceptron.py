import numpy as np
from Cost import CostFunction


class Perceptron:
    def __init__(self, X: np.ndarray, Y: np.ndarray, activation_function: callable, cost_function: CostFunction) -> None:
        """
        X: Matrix of features data
        Y: Vector of the attribute we want to predict
        activation_function: function that will be applied on the model
        cost_function: class that representents the cost function to use
        """
        self.X = X
        self.Y = Y
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.W, self.b = self.init_weights()

    def train(self, epochs: int = 100, η: float = 0.05, X_test: np.ndarray = None, Y_test: np.ndarray = None):

        errors = []
        accuracies = []
        errors_test = []
        accuracies_test = []
        
        for i in range(epochs):
            
            A = self.model(self.X)

            L = self.cost_function(A, self.Y)
            errors.append(L.value())
            accuracies.append(self.accuracy())
            
            if X_test is not None and Y_test is not None:
                L_test = self.cost_function(self.model(X_test), Y_test)
                errors_test.append(L_test.value())
                accuracies_test.append(self.accuracy(X_test, Y_test))

            self.W, self.b = self.calculate_new_weights(η, L)

        return errors, accuracies, errors_test, accuracies_test

    def init_weights(self) -> tuple[np.ndarray, float]:
        W = np.random.randn(self.X.shape[1], 1)
        b = np.random.randn(1)
        W = W.reshape(len(W), 1)
        return W, b

    def calculate_Z(self, X: np.ndarray) -> np.ndarray:
        """
        Returns X·W+b
        """

        assert X.shape[1] == self.W.shape[0], "Given number of features must be equal to number of weights"
        arr = X.dot(self.W) + self.b
        arr = arr.reshape(len(arr), 1)
        return arr
    
    def model(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a vector of each element of a vector Z applied to the activation function f, aka the model
        X: Sample
        """

        Z = self.calculate_Z(X)
        arr = np.array(list(map(self.activation_function, Z)))
        arr = arr.reshape((len(arr), 1))
        return arr

    def calculate_new_weights(self, η: float, C: CostFunction) -> tuple[np.ndarray, float]:
        """
        Does a gradient descent on the convex cost function
        Returns new W and b

        A: Vector of elements that went through the activation function
        η: Learning rate
        C: Instance of class of a convex function
        """

        W = self.W - η * C.derivative_weights(self.X)
        b = self.b - η * C.derivative_bias()

        return W, b

    def predict(self, X: np.ndarray):
        """
        X: Sample of one or many row to predict
        Returns a bool if X is one sample or an ndarray of bools if many samples
        """

        A = self.model(X)
        return A >= 0.5 # should depend on the activation function used
    
    def accuracy(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        """
        Returns the accuracy of the model
        """

        from sklearn.metrics import accuracy_score

        if X is not None:
            Y_pred = self.predict(X) # if we want to test the accuracy
            return accuracy_score(Y, Y_pred)
        else:
            Y_pred = self.predict(self.X)
            return accuracy_score(self.Y, Y_pred)
