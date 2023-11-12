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
        self.errors = []

    def train(self, epochs: int, η: float = 0.05):

        for i in range(epochs):
            
            print(f"Epoch {i} starts")
            
            A = self.model()

            L = self.cost_function(A, self.Y)
            self.errors.append(L.value())

            self.W, self.b = self.calculate_new_weights(η, L)

    def init_weights(self) -> tuple[np.ndarray, float]:
        W = np.random.random_sample(size=self.X.shape[1])
        b = np.random.random_sample()
        W = W.reshape(len(W), 1)
        return W, b

    def calculate_Z(self, X: np.ndarray = None) -> np.ndarray:
        """
        Returns X·W+b
        """
        if X is None: X = self.X # in case we use it to get a prediction

        assert X.shape[1] == self.W.shape[0], "Given number of features must be equal to number of weights"
        arr = X.dot(self.W) + self.b
        arr = arr.reshape(len(arr), 1)
        return arr
    
    def model(self, X: np.ndarray = None) -> np.ndarray:
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

    def predict(self, X: np.ndarray) -> bool:
        """
        X: Sample of one row to predict
        """
        assert X.shape[0] == 1, "Enter only one sample"

        A = self.model(X)
        return A >= 0.5 # should depend on the activation function used