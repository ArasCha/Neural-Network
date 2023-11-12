import numpy as np


class Perceptron:
    def __init__(self, X: np.ndarray, Y: np.ndarray, activation_function: callable, cost_function: object) -> None:
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

    def train(self, epochs: int, η: float = 0.05):

        W, b = self.init_weights()

        for i in range(epochs):
            
            print(f"Epoch {i} starts")
            Z = self.calculate_Z(W, b)
            A = self.calculate_A(Z, self.activation_function)

            L = self.cost_function(A, self.Y)
            cost_value = L.value()
            print(f"Value of error at epoch {i}:", cost_value)

            W, b = self.calculate_new_weights(W, b, η, L)

    def init_weights(self) -> tuple[np.ndarray, float]:
        W = np.random.random_sample(size=self.X.shape[1])
        b = np.random.random_sample()
        W = W.reshape(len(W), 1)
        return W, b

    def calculate_Z(self, W: np.ndarray, b: float) -> np.ndarray:
        """
        Returns X·W+b , aka the model

        W: Vector of weights
        b: bias
        """
        assert self.X.shape[1] == W.shape[0], "Given number of features must be equal to number of weights"
        arr = self.X.dot(W) + b
        arr = arr.reshape(len(arr), 1)
        return arr
    
    def calculate_A(self, Z: np.ndarray, f: callable) -> np.ndarray:
        """
        Returns a vector of each element of a vector Z applied to the activation function f

        Z: Vector of matrix multiplications between X (features data) and W (weights)
        f: Activation function
        """
        arr = np.array(list(map(f, Z)))
        arr = arr.reshape((len(arr), 1))
        return arr

    def calculate_new_weights(self, W: np.ndarray, b: float, η: float, C) -> tuple[np.ndarray, float]:
        """
        Does a gradient descent on the convex cost function
        Returns new W and b

        A: Vector of elements that went through the activation function
        W: Vector of previous weights
        b: Previous bias
        η: Learning rate
        C: Instance of class of a convex function
        """

        _W = W - η * C.derivative_weights(self.X)
        _b = b - η * C.derivative_bias()

        return _W, _b
