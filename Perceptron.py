import numpy as np
from typing import Sequence
from Cost import logloss, logloss_derivative_bias, logloss_derivative_weights, Vector, Matrix
from ActivationFunction import sigmoid


class Perceptron:
    def __init__(self, X: Matrix, Y: Vector) -> None:
        self.weights: Vector
        self.bias: float
        self.Z: Vector = self.calculate_Z(X)
        self.Y = Y

    def train():
        ...

    def calculate_Z(self, X: Matrix, W: Vector, b: float) -> Vector:
        """
        Returns the matrix multiplication between X and W, plus b

        X: Matrix of data of each feature
        W: Vector of weights
        b: bias
        """
        assert len(X[0]) == len(W), "Given number of features must be equal to number of weights"
        return np.dot(X, W) + b
    
    def calculate_A(self, Z: Vector, f: function) -> Vector:
        """
        Returns a vector of each element of a vector Z applied to the activation function f

        Z: Vector of matrix multiplications between X (features data) and W (weights)
        f: Activation function
        """
        return [f(z) for z in Z]
    
    def calculate_cost(self, A: Vector, f: function) -> float:
        """
        A: Vector of elements that went through the activation function
        f: Cost function
        """
        return f(A, self.Y, self.Z)

    def calculate_new_weights(self, X: Matrix, Y: Vector, A: Vector, W: Vector, b: float, η: float = 0.05) -> tuple[Vector, float]:
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
