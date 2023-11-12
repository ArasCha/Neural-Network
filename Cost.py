import numpy as np


class LogLoss:
    def __init__(self, A: np.ndarray, Y: np.ndarray):
        """
        A: Vector of elements that went through the activation function
        Y: Vector of the attribute we want to predict
        """
        assert A.shape == Y.shape, "A and Y matrixes must have the same size"
        self.A = A
        self.Y = Y


    def value(self) -> float:
        """
        Returns the value of the error
        """
        return -(1/len(self.Y)) * np.sum( [y * np.log(a) + (1-y) * np.log(1-a) for a,y in zip(self.A,self.Y)] )

    def derivative_weights(self, X) -> np.ndarray:
        """
        Returns ∂L/∂W
        """
        assert self.A.shape[0] == X.shape[0], "A and Z must have the same number of lines"
        return (1/len(self.A)) * (X.T).dot(self.A - self.Y)

    def derivative_bias(self) -> float:
        """
        Returns ∂L/∂b
        """
        return (1/len(self.A)) * np.sum(self.A - self.Y)
