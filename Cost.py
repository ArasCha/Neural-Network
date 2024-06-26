import numpy as np


class CostFunction:
    def __init__(self, A: np.ndarray, Y: np.ndarray):
        """
        A: Vector of elements that went through the activation function
        Y: Vector of the attribute we want to predict
        """
        self.A = A
        self.Y = Y
    
    def value(self) -> float:
        """
        Returns the value of the error
        """

    def derivative_weights(self, X) -> np.ndarray:
        """
        Returns ∂L/∂W
        """
    
    def derivative_bias(self) -> float:
        """
        Returns ∂L/∂b
        """

class LogLoss(CostFunction):
    def __init__(self, A: np.ndarray, Y: np.ndarray):
        super().__init__(A, Y)

    def value(self) -> float:
        ε = 1e-15
        return -(1/len(self.Y)) * np.sum( self.Y * np.log(self.A + ε) + (1 - self.Y) * np.log(1 - self.A + ε) )

    def derivative_weights(self, X) -> np.ndarray:
        return (1/len(self.A)) * (X.T).dot(self.A - self.Y)

    def derivative_bias(self) -> float:
        return (1/len(self.A)) * np.sum(self.A - self.Y)



class LeastSquare:
    def __init__(self, A: np.ndarray, Y: np.ndarray):
        super().__init__(A, Y)

    def value(self) -> float:
        pass

    def derivative_weights(self, X) -> np.ndarray:
        pass

    def derivative_bias(self) -> float:
        pass