import numpy as np
from Cost import logloss, logloss_derivative_bias, logloss_derivative_weights
from ActivationFunction import sigmoid, tanh


class Perceptron:
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = X
        self.Y = Y

    def train(self, epochs: int, η: float = 0.05):

        W, b = self.init_weights()

        for i in range(epochs):
            
            print(f"Epoch {i} starts")
            Z = self.calculate_Z(self.X, W, b)
            A = self.calculate_A(Z, sigmoid)

            cost_value = self.calculate_cost(A, logloss)
            print(f"Value of error at epoch {i}:", cost_value)
            
            W, b = self.calculate_new_weights(self.X, self.Y, A, W, b, η)

    def init_weights(self) -> tuple[np.ndarray, float]:
        W = np.random.random_sample(size=self.X.shape[1])
        b = np.random.random_sample()
        W = W.reshape(len(W), 1)
        return W, b

    def calculate_Z(self, X: np.ndarray, W: np.ndarray, b: float) -> np.ndarray:
        """
        Returns the model, aka the matrix multiplication between X and W, plus b

        X: Matrix of data of each feature
        W: Vector of weights
        b: bias
        """
        assert X.shape[1] == W.shape[0], "Given number of features must be equal to number of weights"
        arr = X.dot(W) + b
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
    
    def calculate_cost(self, A: np.ndarray, f: callable) -> float:
        """
        A: Vector of elements that went through the activation function
        f: Cost function
        """
        return f(A, self.Y)

    def calculate_new_weights(self, X: np.ndarray, Y: np.ndarray, A: np.ndarray, W: np.ndarray, b: float, η: float) -> tuple[np.ndarray, float]:
        """
        Does a gradient descent on the LogLoss function

        X: Matrix of features data
        Y: Vector of the attribute we want to predict
        A: Vector of elements that went through the activation function
        W: Vector of previous weights
        b: Previous bias
        η: Learning rate
        """

        _W = W - η * logloss_derivative_weights(X, A, Y)
        _b = b - η * logloss_derivative_bias(A, Y)

        return _W, _b
