import numpy as np
from typing import Sequence


Vector = Sequence[float]
Matrix = Sequence[Vector]

class Cost:
    ...


def logloss(A: Vector, Y: Vector) -> float:
    assert len(Y) == len(A)
    return -(1/len(Y)) * np.sum( y * np.log10(a) + (1-y) * np.log10(1-a) for a,y in zip(A,Y))

def logloss_derivative_weights(X: Matrix, A: Vector, Y: Vector) -> Vector:
    """
    Returns ∂L/∂W
    """
    assert len(A) == len(Y) == len(X)
    return (1/len(A)) * np.dot(np.transpose(X), (A - Y))

def logloss_derivative_bias(A: Vector, Y: Vector) -> float:
    """
    Returns ∂L/∂b
    """
    assert len(A) == len(Y)
    return (1/len(A)) * np.sum(A - Y)